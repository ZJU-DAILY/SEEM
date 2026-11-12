from __future__ import annotations
from networks import ForecastFusion
import utils
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from utils import read, write, save
from teacher import solve_repair
from networks import StructuredBehavioralReader
from utils import capture, restore
import math
from utils import bind
from data import digest
from resources import VARIANTS
from resources import CONFIG
from networks import BilinearMatcher, BilinearReader


SPLITS=('train','validation','test')

def query_chunk(h,c):return max(1,min(64,2_000_000//(20*h*c)))

def new_reader(cfg,hp,initial=None,device='cpu'):
    utils.seed_everything(int(os.environ['SEEM_RUN_SEED']))
    fusion=ForecastFusion(cfg['pred_len'],(1,),True)
    old=SimpleNamespace(fusion=fusion,encoder=SimpleNamespace(layers=[SimpleNamespace(in_features=96*cfg['channels'])]),hparams=dict(hp,lambda_pcl=hp.get('lambda_pcl',.1) or .1))
    utils.seed_everything(int(os.environ['SEEM_RUN_SEED']))
    model=StructuredBehavioralReader(old,3*cfg['pred_len']*cfg['channels'],cfg['channels'],8,16,
        128 if cfg['channels']<=32 else 4,cfg['channels']>32).to(device)
    model.hparams['lambda_pcl']=0.
    return model

def bank_chunks(model,ds,chunk):
    device=next(model.parameters()).device
    reps=ds.bank.representatives
    for i in range(0,len(reps),chunk):
        k=ds.keys[reps[i:i+chunk]].to(device)
        yield i,model.encoder(k)

def encode(model,ds,chunk):return torch.cat([v for _,v in bank_chunks(model,ds,chunk)])

@torch.no_grad()
def evaluate(model,ds,labels,chunk,tag,emb=None):
    model.eval();device=next(model.parameters()).device
    if emb is None:emb=encode(model,ds if ds.split.name=='train' else ds.train_data,chunk)
    n=ds.split.count;h=ds.split.pred_len;c=ds.split.channels;step=getattr(model,'runtime_plan',{}).get('query_microbatch',query_chunk(h,c))
    mse=mae=0.
    for i in range(0,n,step):
        rows=np.arange(i,min(i+step,n));p,_=read_forecast(model,ds,rows,emb,labels,device)
        e=(p-ds.split.futures[rows].to(device)).double()
        mse+=float(e.square().sum());mae+=float(e.abs().sum())
        if i==0 or i%(step*32)==0:print('{} | {}/{} queries'.format(tag,i+len(rows),n),flush=True)
    r=dict(mse=mse/(n*h*c),mae=mae/(n*h*c))
    if ds.split.name != 'test':print('{} | MSE {:.10f} MAE {:.10f}'.format(tag,r['mse'],r['mae']),flush=True)
    return r

@torch.no_grad()
def teacher_one(ds,row,args,device,screen_chunk):

    h,b=ds.forecast_inputs(np.asarray([row]),device);y=ds.split.futures[row:row+1].to(device)
    backbone=b-h[:,-1:];residual=y-b;continuation=y-h[:,-1:]
    bank=ds.bank;lo,hi=int(ds.low[row]),int(ds.high[row])
    legal=np.asarray([g for g in range(len(bank.members)) if bank.interval(g,lo,hi)[2]>0 and not lo<=bank.representatives[g]<hi])
    if len(legal)<20:raise ValueError('Fewer than 20 legal Teacher groups')
    gains=[];distances=[]
    for i in range(0,len(legal),screen_chunk):
        v=torch.from_numpy(bank.read_values(legal[i:i+screen_chunk],lo,hi)).to(device)
        d=v-backbone[0];dot=(d*residual[0]).sum((1,2));norm=d.square().sum((1,2))
        strength=(dot/norm.clamp_min(1e-30)).clamp(0,1)
        gains.append((2*strength*dot-strength.square()*norm).cpu())
        distances.append((v-continuation[0]).double().square().mean((1,2)).cpu())
    gain=torch.cat(gains);pos=legal[int(torch.cat(distances).argmin())]
    take=torch.argsort(gain,descending=True,stable=True)[:min(args.teacher_candidates,len(legal))]
    ids=torch.as_tensor(legal[take.numpy()],device=device)
    v=torch.from_numpy(bank.read_values(ids.cpu().numpy(),lo,hi)).to(device)
    solved=solve_repair((v-backbone[0])[None],residual,args.teacher_steps,args.teacher_tolerance)
    u=solved.pop('u');gamma=u.sum(1);order=u.argsort(dim=1,descending=True,stable=True)[:,:20]
    sparse=u.gather(1,order);chosen=ids[order]
    w=torch.where((gamma>0)[:,None],sparse/gamma.clamp_min(torch.finfo(u.dtype).tiny)[:,None],torch.zeros_like(sparse))
    return {k:v.cpu() for k,v in dict(ids=chosen,weights=w,gamma=gamma,
        pcl_positive_mass=(w*(chosen==pos)).sum(1),nonzero_support=(sparse>0).sum(1),
        source_ids=torch.as_tensor(ds.split.source_sample_index[row:row+1],device=device),**solved).items()}


def read_forecast(model, data, rows, bank_embeddings, labels, device):
    query, low, high = data.queries(rows, device)
    result = model.retrieve(query, bank_embeddings, labels, len(data.bank.members), low, high)
    if result["member_columns"] != len(getattr(data.bank, "address_rows", getattr(data.bank, "representatives", data.bank.keys))):
        raise AssertionError("A hidden shortlist replaced full-bank retrieval.")
    values = data.values(rows, result["group_ids"], device)
    history, backbone = data.forecast_inputs(rows, device)
    return model(history, backbone, values, result["weights"]), result


def training_objective(model,ds,rows,result,prediction,teacher,settings,device):
    mse=(prediction-ds.split.futures[rows].to(device)).square().mean()
    kl=mse.new_zeros(());mixture=mse.new_zeros(())
    use_kl=model.options['kl'];use_mixture=model.options['mixture']
    if use_kl or use_mixture:
        if teacher is None:raise ValueError('Enabled Teacher supervision requires authenticated labels')
        ids=teacher['ids'][rows].to(device);w=teacher['weights'][rows].to(device)
        gamma=teacher['gamma'][rows].to(device)
        if use_kl:
            logp=(result['group_scores']/model.hparams['tau_r']).log_softmax(1).gather(1,ids)
            kl=(gamma*(w*(w.clamp_min(1e-30).log()-logp)).sum(1)).mean()
        if use_mixture:
            with torch.no_grad():
                target=(ds.values(rows,ids,device)*w[:,:,None,None]).sum(1)
                values=ds.values(rows,result['group_ids'],device)
            actual=(values*result['weights'][:,:,None,None]).sum(1)
            mixture=(gamma*(actual-target).square().mean((1,2))).mean()
    loss=mse+settings['weight_weight']*kl+settings['effect_weight']*mixture
    return loss,(mse,kl,mixture)


SPLITS=("train","validation","test")

def check_seal(folder):
    folder=Path(folder).resolve()
    for name,sha in read(folder/'complete.json')['files'].items():
        path=(folder/name).resolve()
        if folder not in path.parents or digest(path)!=sha:
            raise ValueError('Changed sealed artifact: '+str(path))


def seal(folder,names):
    folder=Path(folder)
    write(folder/'complete.json',dict(files={n:digest(folder/n) for n in names}))


def loss_settings(variant,original):
    plan=VARIANTS[variant]
    return dict(original,weight_weight=original['weight_weight'] if plan['kl'] else 0.,
        effect_weight=original['effect_weight'] if plan['mixture'] else 0.)


@torch.no_grad()
def score_record(raw,pre,score):
    a=raw.detach().double();z=pre.detach().double();s=score.detach().double()
    return torch.stack([a.new_tensor(a.numel()),a.sum(),a.square().sum(),a.min(),a.max(),
        z.sum(),z.square().sum(),z.min(),z.max(),(s.abs()>.999).double().sum(),
        (s.abs()==1).double().sum(),(1-s.square()).sum(),(a.tanh().abs()>.999).double().sum()])

def merge_records(records):
    if not records:return None
    x=torch.stack(records).double().cpu().numpy();n=float(x[:,0].sum())
    mean=float(x[:,1].sum()/n);zmean=float(x[:,5].sum()/n)
    return dict(pairs=int(n),raw_mean=mean,raw_std=math.sqrt(max(0,float(x[:,2].sum()/n)-mean*mean)),
        raw_min=float(x[:,3].min()),raw_max=float(x[:,4].max()),pre_tanh_mean=zmean,
        pre_tanh_std=math.sqrt(max(0,float(x[:,6].sum()/n)-zmean*zmean)),pre_tanh_min=float(x[:,7].min()),
        pre_tanh_max=float(x[:,8].max()),saturated_fraction=float(x[:,9].sum()/n),
        exact_saturated_fraction=float(x[:,10].sum()/n),mean_tanh_derivative=float(x[:,11].sum()/n),
        same_raw_unscaled_saturated_fraction=float(x[:,12].sum()/n),
        scope='all query-memory score pairs before legal mask; unscaled fraction is counterfactual only')

@torch.no_grad()
def routing_probe(model,ds,labels,resource,device,previous=None):
    rows=np.arange(min(8,ds.split.count));was_training=model.training;model.eval()
    matcher=model.matcher;enabled=matcher.collect_score_diagnostics;records=matcher.score_records
    matcher.collect_score_diagnostics=True;matcher.score_records=[]
    try:
        emb=encode(model,ds,resource['eval_encoding_chunk'])
        query,low,high=ds.queries(rows,device)
        ret=model.retrieve(query,emb,labels,len(ds.bank.members),low,high)
        stats=merge_records(matcher.score_records)
        weights=ret['weights'].detach().double().cpu();ids=ret['group_ids'].detach().cpu()
        entropy=-(weights*weights.clamp_min(1e-30).log()).sum(1)
        out=dict(rows=rows.tolist(),ids=ids.tolist(),weights=weights.tolist(),scores=stats,
            mean_normalized_weight_entropy=float((entropy/math.log(weights.shape[1])).mean()),
            mean_top_weight=float(weights.max(1).values.mean()),top20_set_change_vs_previous=None)
        if previous is not None:
            if previous['rows']!=out['rows']:raise ValueError('Diagnostic probe rows changed')
            out['top20_set_change_vs_previous']=float(np.mean([1-len(set(a)&set(b))/len(a) for a,b in zip(out['ids'],previous['ids'])]))
        return out
    finally:
        matcher.collect_score_diagnostics=enabled;matcher.score_records=records;model.train(was_training)

def print_step(epoch,number,item):
    d=item['score_diagnostics'];g=item['grad_squared']
    print('SCORE epoch%d batch%d | z[%g,%g] saturated %.6f exact %.6f | encoder grad2 %.6g matcher grad2 %.6g'%(
        epoch,number,d['pre_tanh_min'],d['pre_tanh_max'],d['saturated_fraction'],d['exact_saturated_fraction'],g['encoder.'],g['matcher.']),flush=True)

def print_epoch(epoch,probe,grad,steps):
    d=probe['scores']
    print('MATCHER PROBE epoch%d | raw[%g,%g] pre-tanh[%g,%g] saturated %.6f exact %.6f | weight entropy %.6f top20 change %s'%(
        epoch,d['raw_min'],d['raw_max'],d['pre_tanh_min'],d['pre_tanh_max'],d['saturated_fraction'],d['exact_saturated_fraction'],
        probe['mean_normalized_weight_entropy'],probe['top20_set_change_vs_previous']),flush=True)
    for name in ('encoder.','matcher.query_projection.','matcher.memory_projection.','fusion.retrieval_pred.','fusion.linear_pred.'):
        print('GRAD epoch%d | %s nonzero %d/%d squared_sum %.9g'%(epoch,name,grad[name]['nonzero_steps'],steps,grad[name]['squared_sum']),flush=True)


def new_model(cfg, hp, variant, reps, device, initial=None):
    if variant not in ('full', 'bilinear64'):
        raise ValueError('Only the main SEEM model is supported')
    if cfg['seq_len'] != 96 or cfg['pred_len'] not in (96, 192, 336, 720) or hp['top_k'] != 20:
        raise ValueError('Expected input96, horizon96/192/336/720 and Top20')
    scaling = CONFIG['network']['matcher_score_scaling']
    if scaling not in ('none', '7/channels'):
        raise ValueError('Unsupported matcher score scaling')
    scaled = scaling == '7/channels'
    variant = 'bilinear64' if scaled else 'full'
    model = new_reader(cfg, hp, None, 'cpu')
    if initial is not None and initial.get('epoch') != 0:
        raise ValueError('Requires original epoch-zero initialization')
    if scaled and initial is not None:
        model.load_state_dict(initial['state_dict'], strict=True)
    rank = int(cfg.get('bilinear_rank', 64)) if scaled else 64
    if rank not in (32, 64, 128):
        raise ValueError('Unsupported matcher rank')
    scale = 7. / cfg['channels'] if scaled else 1.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(os.environ['SEEM_RUN_SEED']) + 2000)
        matcher = BilinearMatcher(cfg['channels'], rank=rank, score_scale=scale)
    model.matcher = matcher
    model.__class__ = BilinearReader
    if not scaled and initial is not None:
        model.load_state_dict(initial['state_dict'], strict=True)
    model.options = dict(VARIANTS[variant])
    model.hparams = dict(model.hparams if scaled else hp, top_k=20, lambda_pcl=0.)
    model.checkpoint_matching = False
    if scaled:
        model.architecture = dict(model.architecture, variant=variant,
            matcher='tanh((7/C) * ((Wq flatten(E(Key_q))) @ (Wm flatten(E(Key_g))).T /sqrt{})'.format(rank),
            matcher_score_scale=scale, matcher_score_scale_rule='7/C before tanh; C=7 exact original',
            matcher_rank=rank, matcher_bias=False, matcher_seed=int(os.environ['SEEM_RUN_SEED'])+2000,
            matcher_init_std=0.02, original_encoder_unchanged=True,
            score_range=[-1., 1.], cosine_normalization=False, full_bank_GEMM=True,
            views=4, value='mean(Y-X_source_last)', top_k=20, Teacher_KL=True, Teacher_mixture=True)
    else:
        model.architecture = dict(model.architecture, variant=variant, views=4,
            retrieval_feature_dim=4*cfg['channels']*8, matcher='rank64 bilinear tanh(dot/sqrt64)',
            matcher_init_std=0.02, matcher_seed=int(os.environ['SEEM_RUN_SEED'])+2000,
            top_k=20, Teacher_KL=True, Teacher_mixture=True, value='mean(Y_s-X_s_last)')
    model.register_buffer('representative_rows', torch.as_tensor(reps), persistent=False)
    return model.to(device)
