from __future__ import annotations
import torch
import time
import gc
import numpy as np
import json
import fcntl
import utils as source
from utils import save

teacher_batch = None

def simplex_projection(x):
    ordered = x.sort(dim=-1, descending=True).values
    count = torch.arange(1, x.shape[-1]+1, device=x.device, dtype=x.dtype)
    threshold = (ordered.cumsum(-1)-1) / count
    rho = (ordered > threshold).sum(-1, keepdim=True).clamp_min(1)-1
    result = (x-threshold.gather(-1,rho)).clamp_min(0)
    return result / result.sum(-1,keepdim=True)

def quadratic(w, gram, linear):
    return (w * torch.bmm(gram,w.unsqueeze(-1)).squeeze(-1)).sum(-1)-2*(w*linear).sum(-1)

@torch.no_grad()
def solve_simplex(gram, linear, initial, steps=256, tolerance=1e-7):

    w=initial.clone(); z=w.clone(); best=w.clone()
    best_objective=quadratic(best,gram,linear)
    lipschitz=(2*gram.abs().sum(-1).amax(-1,keepdim=True)).clamp_min(1e-12)
    momentum=1.0
    for iteration in range(steps):
        gradient=2*(torch.bmm(gram,z.unsqueeze(-1)).squeeze(-1)-linear)
        candidate=simplex_projection(z-gradient/lipschitz)
        value=quadratic(candidate,gram,linear)
        improve=value<best_objective
        best=torch.where(improve[:,None],candidate,best)
        best_objective=torch.minimum(value,best_objective)
        next_momentum=(1+(1+4*momentum**2)**.5)/2
        z=candidate+((momentum-1)/next_momentum)*(candidate-w)
        w=candidate;momentum=next_momentum
        if (iteration+1)%16==0 or iteration+1==steps:
            grad=2*(torch.bmm(gram,best.unsqueeze(-1)).squeeze(-1)-linear)
            gap=((best*grad).sum(-1)-grad.amin(-1)).clamp_min(0)
            if bool((gap<=tolerance).all()):break
    return best,gap

@torch.no_grad()
def solve_repair(delta, residual, steps=256, tolerance=1e-7, top_k=20):

    d=delta.flatten(2).double(); e=residual.flatten(1).double()
    if not torch.isfinite(d).all() or not torch.isfinite(e).all():
        raise ValueError('Nonfinite Teacher input.')
    batch,n,dim=d.shape
    norm=d.square().sum(-1)
    strength=((d*e[:,None]).sum(-1)/norm.clamp_min(1e-30)).clamp(0,1)
    gain=2*strength*(d*e[:,None]).sum(-1)-strength.square()*norm
    best_id=gain.argmax(1,keepdim=True)
    single=torch.zeros(batch,n+1,device=d.device,dtype=d.dtype)
    single.scatter_(1,best_id,strength.gather(1,best_id))
    single[:,-1]=1-single[:,:n].sum(1)
    augmented=torch.cat((d,torch.zeros_like(d[:,:1])),1)
    gram=augmented@augmented.transpose(1,2)/dim
    linear=(augmented*e[:,None]).sum(-1)/dim
    dense,dense_gap=solve_simplex(gram,linear,single,steps,tolerance)
    ids=dense[:,:n].argsort(dim=1,descending=True,stable=True)[:,:min(top_k,n)]
    support=torch.cat((ids,torch.full((batch,1),n,device=d.device,dtype=torch.long)),1)
    sd=augmented.gather(1,support[:,:,None].expand(-1,-1,dim))
    initial=dense.gather(1,support)

    initial[:,-1]=1-initial[:,:-1].sum(1)
    initial=initial.clamp_min(0);initial=initial/initial.sum(1,keepdim=True)
    refit,gap=solve_simplex(sd@sd.transpose(1,2)/dim,(sd*e[:,None]).sum(-1)/dim,initial,steps,tolerance)
    candidate=torch.zeros_like(single).scatter(1,support,refit)[:,:n].to(delta.dtype)
    single=single[:,:n].to(delta.dtype)
    df=delta.flatten(2);ef=residual.flatten(1)
    def mse(w):return (ef-(w[:,:,None]*df).sum(1)).square().mean(1)
    base=ef.square().mean(1);cl=mse(candidate);sl=mse(single)
    use_single=sl<cl;u=torch.where(use_single[:,None],single,candidate)
    loss=torch.minimum(cl,sl);improve=loss<base
    u=torch.where(improve[:,None],u,torch.zeros_like(u))
    return dict(u=u,base_mse=base,teacher_mse=torch.where(improve,loss,base),dense_gap=dense_gap.float(),refit_gap=gap.float())

def objective(model,ds,rows,result,prediction,teacher,args,device):
    forecast=(prediction-ds.split.futures[rows].to(device)).square().mean()
    ids=teacher['ids'][rows].to(device);w=teacher['weights'][rows].to(device);gamma=teacher['gamma'][rows].to(device)
    logp=(result['group_scores']/model.hparams['tau_r']).log_softmax(1).gather(1,ids)
    kl=(gamma*(w*(w.clamp_min(1e-30).log()-logp)).sum(1)).mean()
    with torch.no_grad():
        target=(ds.values(rows,ids,device)*w[:,:,None,None]).sum(1)
        values=ds.values(rows,result['group_ids'],device)
    mixture=(values*result['weights'][:,:,None,None]).sum(1)
    effect=(gamma*(mixture-target).square().mean((1,2))).mean()
    return forecast+args.weight_weight*kl+args.effect_weight*effect,(forecast,kl,effect)


def cuda_memory_info(device):

    requested = torch.device(device)
    if requested.type != 'cuda':
        raise ValueError('CUDA memory query requires a CUDA device')
    index = requested.index
    if index is None:
        index = torch.cuda.current_device()
    return torch.cuda.mem_get_info(index)

def gpu_simplex_projection(x):
    ordered = x.sort(dim=-1, descending=True).values
    count = torch.arange(1, x.shape[-1]+1, device=x.device, dtype=x.dtype)
    threshold = (ordered.cumsum(-1)-1) / count
    rho = (ordered > threshold).sum(-1, keepdim=True).clamp_min(1)-1
    result = (x-threshold.gather(-1,rho)).clamp_min(0)
    return result / result.sum(-1,keepdim=True)


def gpu_quadratic(w, gram, linear):
    return (w * torch.bmm(gram,w.unsqueeze(-1)).squeeze(-1)).sum(-1)-2*(w*linear).sum(-1)


@torch.no_grad()
def gpu_solve_simplex(gram, linear, initial, steps=256, tolerance=1e-7):

    w=initial.clone(); z=w.clone(); best=w.clone()
    best_objective=gpu_quadratic(best,gram,linear)
    lipschitz=(2*gram.abs().sum(-1).amax(-1,keepdim=True)).clamp_min(1e-12)
    momentum=1.0
    active=torch.ones(len(w),dtype=torch.bool,device=w.device)
    gap=torch.full((len(w),),float("inf"),device=w.device,dtype=w.dtype)
    for iteration in range(steps):
        gradient=2*(torch.bmm(gram,z.unsqueeze(-1)).squeeze(-1)-linear)
        candidate=gpu_simplex_projection(z-gradient/lipschitz)
        value=gpu_quadratic(candidate,gram,linear)
        improve=(value<best_objective) & active
        best=torch.where(improve[:,None],candidate,best)
        best_objective=torch.where(active,torch.minimum(value,best_objective),best_objective)
        next_momentum=(1+(1+4*momentum**2)**.5)/2
        z=candidate+((momentum-1)/next_momentum)*(candidate-w)
        w=candidate;momentum=next_momentum
        if (iteration+1)%16==0 or iteration+1==steps:
            grad=2*(torch.bmm(gram,best.unsqueeze(-1)).squeeze(-1)-linear)
            current=((best*grad).sum(-1)-grad.amin(-1)).clamp_min(0)
            gap=torch.where(active,current,gap)
            active=active & (gap>tolerance)
            if not bool(active.any()):break
    return best,gap


@torch.no_grad()
def gpu_solve_repair(delta, residual, steps=256, tolerance=1e-7, top_k=20):

    d=delta.flatten(2).double(); e=residual.flatten(1).double()
    if not torch.isfinite(d).all() or not torch.isfinite(e).all():
        raise ValueError('Nonfinite Teacher input.')
    batch,n,dim=d.shape
    norm=d.square().sum(-1)
    strength=((d*e[:,None]).sum(-1)/norm.clamp_min(1e-30)).clamp(0,1)
    gain=2*strength*(d*e[:,None]).sum(-1)-strength.square()*norm
    best_id=gain.argmax(1,keepdim=True)
    single=torch.zeros(batch,n+1,device=d.device,dtype=d.dtype)
    single.scatter_(1,best_id,strength.gather(1,best_id))
    single[:,-1]=1-single[:,:n].sum(1)
    augmented=torch.cat((d,torch.zeros_like(d[:,:1])),1)
    gram=augmented@augmented.transpose(1,2)/dim
    linear=(augmented*e[:,None]).sum(-1)/dim
    dense,dense_gap=gpu_solve_simplex(gram,linear,single,steps,tolerance)
    ids=dense[:,:n].argsort(dim=1,descending=True,stable=True)[:,:min(top_k,n)]
    support=torch.cat((ids,torch.full((batch,1),n,device=d.device,dtype=torch.long)),1)
    sd=augmented.gather(1,support[:,:,None].expand(-1,-1,dim))
    initial=dense.gather(1,support)

    initial[:,-1]=1-initial[:,:-1].sum(1)
    initial=initial.clamp_min(0);initial=initial/initial.sum(1,keepdim=True)
    refit,gap=gpu_solve_simplex(sd@sd.transpose(1,2)/dim,(sd*e[:,None]).sum(-1)/dim,initial,steps,tolerance)
    candidate=torch.zeros_like(single).scatter(1,support,refit)[:,:n].to(delta.dtype)
    single=single[:,:n].to(delta.dtype)
    df=delta.flatten(2);ef=residual.flatten(1)
    def mse(w):return (ef-(w[:,:,None]*df).sum(1)).square().mean(1)
    base=ef.square().mean(1);cl=mse(candidate);sl=mse(single)
    use_single=sl<cl;u=torch.where(use_single[:,None],single,candidate)
    loss=torch.minimum(cl,sl);improve=loss<base
    u=torch.where(improve[:,None],u,torch.zeros_like(u))
    return dict(u=u,base_mse=base,teacher_mse=torch.where(improve,loss,base),dense_gap=dense_gap.float(),refit_gap=gap.float())


class GPUTeacher:
    def __init__(self, reference, max_queries=8):
        self.reference=reference;self.max_queries=max_queries;self.means=None;self.bank=None;self.verified=False;self.calls=0
    def release(self):
        self.means=None;self.bank=None
        gc.collect()
    def prepare(self,ds,device):
        if self.bank is not ds.bank or self.means is None:
            self.release();self.bank=ds.bank
            self.means=torch.as_tensor(np.array(ds.bank.shared.means[0],copy=True),device=device)
            print('GPU Teacher | shared means resident | bytes',self.means.numel()*self.means.element_size(),flush=True)
    def screened(self,ds,row,backbone,residual,continuation,args,device):
        bank=ds.bank;lo,hi=int(ds.low[row]),int(ds.high[row])
        intervals=[bank.interval(g,lo,hi) for g in range(len(bank.members))]
        legal=np.asarray([g for g,(first,last,count) in enumerate(intervals) if count>0 and not lo<=bank.representatives[g]<hi],dtype=np.int64)
        if len(legal)<20:raise ValueError('Fewer than 20 legal Teacher groups')
        ids=torch.as_tensor(legal,device=device);v=self.means[ids]
        changed=np.asarray([j for j,g in enumerate(legal) if intervals[g][1]>intervals[g][0]],dtype=np.int64)
        if len(changed):

            patches=bank.read_values(legal[changed],lo,hi)
            v[torch.as_tensor(changed,device=device)]=torch.from_numpy(patches).to(device)
        d=v-backbone;dot=(d*residual).sum((1,2));norm=d.square().sum((1,2))
        strength=(dot/norm.clamp_min(1e-30)).clamp(0,1);gain=2*strength*dot-strength.square()*norm
        pos=ids[(v-continuation).double().square().mean((1,2)).argmin()]
        take=torch.argsort(gain,descending=True,stable=True)[:min(args.teacher_candidates,len(legal))]
        return ids[take],d[take],pos
    @torch.no_grad()
    def block(self,ds,rows,args,device):
        self.prepare(ds,device)
        h,b=ds.forecast_inputs(rows,device);y=ds.split.futures[rows].to(device)
        backbone=b-h[:,-1:];residual=y-b;continuation=y-h[:,-1:]
        screened=[self.screened(ds,int(row),backbone[j],residual[j],continuation[j],args,device) for j,row in enumerate(rows)]

        buckets={}
        for j,(ids,d,pos) in enumerate(screened):buckets.setdefault(len(ids),[]).append(j)
        output=[None]*len(rows)
        for count,indices in buckets.items():
            ids=torch.stack([screened[j][0] for j in indices]);delta=torch.stack([screened[j][1] for j in indices]);positive=torch.stack([screened[j][2] for j in indices])
            solved=gpu_solve_repair(delta,residual[indices],args.teacher_steps,args.teacher_tolerance)
            u=solved.pop('u');gamma=u.sum(1);order=u.argsort(dim=1,descending=True,stable=True)[:,:20]
            sparse=u.gather(1,order);chosen=ids.gather(1,order)
            w=torch.where((gamma>0)[:,None],sparse/gamma.clamp_min(torch.finfo(u.dtype).tiny)[:,None],torch.zeros_like(sparse))
            part=dict(ids=chosen,weights=w,gamma=gamma,pcl_positive_mass=(w*(chosen==positive[:,None])).sum(1),nonzero_support=(sparse>0).sum(1),source_ids=torch.as_tensor(ds.split.source_sample_index[np.asarray(rows)[indices]],device=device),**solved)
            for local,j in enumerate(indices):output[j]={k:v[local:local+1].cpu() for k,v in part.items()}
        return {k:torch.cat([r[k] for r in output]) for k in output[0]}
    def safe_block(self,ds,rows,args,device):
        try:return self.block(ds,rows,args,device)
        except RuntimeError as exc:
            if torch.device(device).type!='cuda' or 'out of memory' not in str(exc).lower():raise
        gc.collect();torch.cuda.empty_cache()
        if len(rows)>1:
            middle=len(rows)//2
            print('Teacher CUDA OOM; retry same queries in',middle,'query batches',flush=True)
            a=self.safe_block(ds,rows[:middle],args,device);b=self.safe_block(ds,rows[middle:],args,device)
            return {k:torch.cat([a[k],b[k]]) for k in a}
        self.release();torch.cuda.empty_cache()
        print('Teacher CUDA OOM at one query; original bounded full-dictionary screening fallback',flush=True)
        return self.reference(ds,rows,args,device)
    @torch.no_grad()
    def __call__(self,ds,rows,args,device):
        if ds.split.name!='train':raise ValueError('Teacher must be Train-only')
        rows=np.asarray(rows,dtype=np.int64)
        if not len(rows):raise ValueError('Empty Teacher request')
        start=time.perf_counter();size=self.max_queries
        if torch.device(device).type=='cuda':
            free,_=cuda_memory_info(device)
            per=max(1,args.teacher_candidates*ds.split.pred_len*ds.split.channels*8*8)
            size=max(1,min(size,int(free*.25)//per))
        pieces=[self.safe_block(ds,rows[i:i+size],args,device) for i in range(0,len(rows),size)]
        result={k:torch.cat([p[k] for p in pieces]) for k in pieces[0]}

        if not self.verified:
            n=min(4,len(rows));expected=self.reference(ds,rows[:n],args,device)
            for k,v in expected.items():
                if v.is_floating_point():torch.testing.assert_close(result[k][:n],v,atol=1e-6,rtol=1e-5,msg='GPU Teacher equivalence: '+k)
                else:torch.testing.assert_close(result[k][:n],v,atol=0,rtol=0,msg='GPU Teacher exact ids/support: '+k)
            self.verified=True;print('GPU Teacher initial equivalence PASS | original IDs/support exact; numeric atol1e-6 rtol1e-5',flush=True)
        self.calls+=1
        print('GPU Teacher | queries %d | solve batch <=%d | %.3fs | original support20/steps/tolerance retained'%(len(rows),size,time.perf_counter()-start),flush=True)
        return result


def build_teacher_cache(ds,args,out,device,batch_size):
    folder=out/'teacher';folder.mkdir(parents=True,exist_ok=True)
    parts=[];began=time.perf_counter()
    for first in range(0,ds.split.count,batch_size):
        rows=np.arange(first,min(first+batch_size,ds.split.count));path=folder/('%06d.pt'%first)
        if path.exists():part=torch.load(path,map_location='cpu',weights_only=True)
        else:part=teacher_batch(ds,rows,args,device);save(path,part)
        if not torch.equal(part['source_ids'],torch.as_tensor(ds.split.source_sample_index[rows])):raise ValueError('Teacher row mismatch.')
        parts.append(part)
        done=int(rows[-1])+1;elapsed=time.perf_counter()-began
        print('Static Teacher | {}/{} | elapsed {:.1f}s | ETA {:.1f}s'.format(done,ds.split.count,elapsed,elapsed/done*(ds.split.count-done)),flush=True)
    teacher={k:torch.cat([p[k] for p in parts]) for k in parts[0]}
    gamma=teacher['gamma'];active=gamma>0
    report=dict(train_queries=ds.split.count,static=True,train_only=True,oof=False,
        base_backbone_mse=float(teacher['base_mse'].mean()),teacher_geometric_mse=float(teacher['teacher_mse'].mean()),
        improved_ratio=float((teacher['teacher_mse']<teacher['base_mse']).float().mean()),
        gamma_quantiles=dict(zip(('min','p10','p25','median','p75','p90','max'),torch.quantile(gamma,torch.tensor([0.,.1,.25,.5,.75,.9,1.])).tolist())),
        gamma_near_zero_ratio=float((gamma<=1e-6).float().mean()),gamma_near_one_ratio=float((gamma>=1-1e-6).float().mean()),
        mean_nonzero_support=float(teacher['nonzero_support'].float().mean()),
        normalized_teacher_entropy=float((-(teacher['weights']*teacher['weights'].clamp_min(1e-30).log()).sum(1))[active].mean()) if active.any() else 0.,
        pcl_positive_mass_mean=float(teacher['pcl_positive_mass'].mean()),pcl_diagnostic_only=True,
        dense_gap_mean=float(teacher['dense_gap'].mean()),refit_gap_mean=float(teacher['refit_gap'].mean()),
        sparse_global_optimum_claimed=False,teacher_candidates=args.teacher_candidates,
        note='Geometric backbone/Memory mixture, not actual Fusion forecast. Gamma is a heuristic supervision weight, not calibrated confidence or a deployment gate.')
    source.write(out/'teacher_report.json',report)
    return teacher


def shared_teacher(ds,args,out,device):
    import shutil
    cache=args.teacher_cache_dir or (out/'static_teacher_cache')
    cache.mkdir(parents=True,exist_ok=True)
    with (cache/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=cache/'signature.json'
        if path.exists() and json.loads(path.read_text())!=args.teacher_signature:
            raise ValueError('Shared Teacher sources/settings changed; use a new sweep directory.')
        source.write(path,args.teacher_signature)
        teacher=build_teacher_cache(ds,args,cache,device,args.teacher_batch_size)
    shutil.copy2(cache/'teacher_report.json',out/'teacher_report.json')
    source.write(out/'teacher_cache.json',dict(path=str(cache.resolve()),shared_across_seeds=True))
    return teacher

