from __future__ import annotations
from framework import read_forecast
import teacher as teacher_api
import utils
import os
import fcntl
import math
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from framework import SPLITS, read, write, save, seal, check_seal, digest, loss_settings
from framework import training_objective as objective
from framework import encode, bank_chunks, evaluate
from utils import capture, restore
from resources import retry_backward
from resources import framework_plan, evaluate_splits, streamed_teacher_batch


GRAD_PREFIXES=('encoder.','matcher.','matcher.query_projection.','matcher.memory_projection.',
    'fusion.retrieval_pred.','fusion.linear_pred.')

def sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)

def build_teacher(model,data,settings,cache_folder,consumer,binding,device):
    if not (model.options['kl'] or model.options['mixture']):return None
    cache_folder=Path(cache_folder);cache_folder.mkdir(parents=True,exist_ok=True)
    consumer=Path(consumer);consumer.mkdir(parents=True,exist_ok=True)
    names=('teacher_candidates','teacher_steps','teacher_tolerance','teacher_batch_size')
    signature=dict(geometry='absolute_future' if model.options['future_value'] else 'original_legal_continuation_means',settings={k:settings[k] for k in names},
        input_binding=binding,teacher_support=20,version='unchanged_static_simplex_refit_v1',
        teacher_code=digest(Path(teacher_api.objective.__code__.co_filename)))
    args=SimpleNamespace(**settings,teacher_cache_dir=cache_folder,teacher_signature=signature)


    with (cache_folder/'.teacher_lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (cache_folder/'complete.json').exists():check_seal(cache_folder)
        original=teacher_api.teacher_batch
        teacher_api.teacher_batch=streamed_teacher_batch
        try:teacher=teacher_api.shared_teacher(data['train'],args,consumer,device)
        finally:teacher_api.teacher_batch=original
        if not (cache_folder/'complete.json').exists():
            names=['signature.json','teacher_report.json']+[str(p.relative_to(cache_folder)) for p in sorted((cache_folder/'teacher').glob('*.pt'))]
            seal(cache_folder,names)
        check_seal(cache_folder)
    write(consumer/'definition.json',dict(original_teacher_unchanged=True,support=20,
        student_top_k=model.hparams['top_k'],student_loss_weights=loss_settings(model.architecture['variant'],settings),
        teacher_cache=str(cache_folder.resolve()),teacher_complete_sha256=digest(cache_folder/'complete.json')))
    return teacher

def batch_step(model,ds,rows,labels,teacher,settings,opt,device,encoding=128,profile=False):
    resource=getattr(model,'runtime_plan',None)
    if resource is None:
        resource=framework_plan(ds.split.pred_len,ds.split.channels)
        resource['encoding_chunk']=min(encoding,resource['encoding_chunk'])
    if resource['logical_batch'] not in (64,128):raise ValueError('Invalid Full-matched batch')
    if len(rows)>resource['logical_batch']:raise ValueError('Queries exceed the actual batch')
    if profile:sync(device)
    began=time.perf_counter()
    def work(p):
        timings={};total=np.zeros(4,dtype=np.float64);query_norm=0.
        def add(name,mark):
            if profile:
                sync(device);timings[name]=timings.get(name,0.)+time.perf_counter()-mark
        mark=time.perf_counter()
        direct=p['query_microbatch']>=len(rows) and p['encoding_chunk']>=len(ds.bank.representatives)
        if direct:emb=encode(model,ds,p['encoding_chunk'])
        else:
            with torch.no_grad():emb=encode(model,ds,p['encoding_chunk'])
        learned=any(p.requires_grad for p in model.encoder.parameters())
        if learned:
            if direct:emb.retain_grad()
            else:emb.requires_grad_(True)
        add('bank_encoder_seconds',mark)
        for first in range(0,len(rows),p['query_microbatch']):
            rr=rows[first:first+p['query_microbatch']];mark=time.perf_counter()
            model._profile_matcher=profile
            try:prediction,ret=read_forecast(model,ds,rr,emb,labels,device)
            finally:model._profile_matcher=False
            if learned:ret['query_embeddings'].retain_grad()
            add('native_read_seconds',mark)
            if profile:timings['matcher_seconds']=timings.get('matcher_seconds',0.)+model.last_matcher_seconds
            mark=time.perf_counter()
            loss,parts=objective(model,ds,rr,ret,prediction,teacher,settings,device)
            if not torch.isfinite(loss):raise ValueError('Nonfinite framework objective')
            fraction=len(rr)/len(rows);(loss*fraction).backward()
            add('loss_and_backward_seconds',mark)
            total+=np.asarray([float(v.detach()) for v in (loss,)+parts])*fraction
            qg=ret['query_embeddings'].grad if learned else None
            if qg is not None:query_norm+=float(qg.detach().double().square().sum())
            del prediction,ret,loss,parts,qg
        mark=time.perf_counter()
        bank_norm=0.
        if learned:
            if emb.grad is None:raise ValueError('Missing bank Encoder gradient path')
            upstream=emb.grad.detach();bank_norm=float(upstream.double().square().sum())
            if not direct:
                for first,chunk in bank_chunks(model,ds,p['encoding_chunk']):
                    torch.autograd.backward(chunk,upstream[first:first+len(chunk)])
        add('bank_encoder_VJP_seconds',mark);mark=time.perf_counter()
        gradients={}
        for prefix in GRAD_PREFIXES:
            gs=[v.grad for name,v in model.named_parameters() if name.startswith(prefix) and v.grad is not None]
            if any(not torch.isfinite(g).all() for g in gs):raise ValueError('Nonfinite gradient in '+prefix)
            gradients[prefix]=sum(float(g.detach().double().square().sum()) for g in gs)
        gradients.update(bank_embedding=bank_norm,query_embedding=query_norm)
        add('gradient_diagnostics_seconds',mark)
        return dict(total=float(total[0]),mse=float(total[1]),teacher_kl=float(total[2]),teacher_mixture=float(total[3]),
            grad_squared=gradients,queries=len(rows),timings=timings)
    result,retries=retry_backward(work,opt,resource,device)
    mark=time.perf_counter();opt.step()
    if profile:
        sync(device);result['timings'].update(Adam_seconds=time.perf_counter()-mark,
            full_step_seconds=time.perf_counter()-began,native_read_includes_matcher=True)
    result.update(runtime_plan=dict(resource),oom_retries=retries)
    return result

def train_r0(model,data,labels,cfg,original_settings,teacher,folder,device,epochs=10,encoding=128,initial=None):
    device=torch.device(device)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        check_seal(folder)
        ck=torch.load(folder/'checkpoint_validation_best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(ck['state_dict'],strict=True);model.eval()
        result=read(folder/'result.json');model.runtime_plan=dict(result['runtime_plan']);return result
    resource=getattr(model,'runtime_plan',None) or framework_plan(cfg['pred_len'],cfg['channels'])
    resource['encoding_chunk']=min(encoding,resource['encoding_chunk'])
    model.runtime_plan=resource
    settings=loss_settings(model.architecture['variant'],original_settings)
    if (model.options['kl'] or model.options['mixture']) and teacher is None:
        raise ValueError('Teacher must be loaded for an unfinished supervised run')
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    generator=torch.Generator().manual_seed(int(os.environ['SEEM_RUN_SEED'])+1000)
    loader=DataLoader(torch.arange(data['train'].split.count),batch_size=cfg['batch_size'],shuffle=True,
        generator=generator,num_workers=0,drop_last=False)
    curve=[];best=float('inf');start=0;steps=0
    last=folder/'checkpoint_last.pt'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=True)
        if ck['epochs']!=epochs or ck['loss_settings']!=settings or ck['config']!=cfg:raise ValueError('Resume budget/loss settings changed')
        model.load_state_dict(ck['state_dict'],strict=True);opt.load_state_dict(ck['optimizer'])
        generator.set_state(ck['generator']);restore(ck['rng_state'])
        curve=ck['curve'];best=ck['best'];start=ck['epoch']+1;steps=ck['steps']
        if ck['runtime_plan']!=resource:raise ValueError('Resume physical execution changed')
        resource=dict(ck['runtime_plan']);model.runtime_plan=resource
    else:
        if initial is not None:
            if initial['epoch']!=0 or initial['optimizer']['state']:raise ValueError('Requires original untrained initialization')
            model.load_state_dict(initial['state_dict'],strict=True)
            opt.load_state_dict(initial['optimizer'])
            for group in opt.param_groups:
                group['lr']=cfg['learning_rate'];group['weight_decay']=cfg['weight_decay']
            generator.set_state(initial['generator']);restore(initial['rng_state'])
        save(folder/'checkpoint_initial.pt',dict(epoch=0,state_dict=model.state_dict(),hparams=model.hparams,
            architecture=model.architecture,optimizer=opt.state_dict(),generator=generator.get_state(),rng_state=capture()))
    for epoch in range(start,epochs+1):
        sync(device);began=time.perf_counter();logs=[];model.train()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        if epoch:
            for number,index in enumerate(loader,1):
                item=batch_step(model,data['train'],index.numpy(),labels,teacher,settings,opt,device,encoding,profile=steps<3)
                logs.append(item);steps+=1
                if item['timings']:
                    print('ACTUAL STEP PROFILE {} | {}'.format(steps,item['timings']),flush=True)
                if number==1 or number%16==0 or number==len(loader):
                    print('R0 {} Top{} epoch {} batch {}/{} | MSE {:.9f} KL {:.9f} mixture {:.9f} total {:.9f}'.format(
                        model.architecture.get('configuration',model.architecture['variant']),model.hparams['top_k'],epoch,number,len(loader),
                        item['mse'],item['teacher_kl'],item['teacher_mixture'],item['total']),flush=True)
        if logs and sum(x['queries'] for x in logs)!=data['train'].split.count:
            raise ValueError('Not every Train query was used exactly once')
        sync(device);train_seconds=time.perf_counter()-began
        boundary=capture()
        index_start=time.perf_counter()
        scores,eval_retries=evaluate_splits(model,data,labels,resource,'R0 epoch {}'.format(epoch),evaluate,encode)
        sync(device);index_seconds=0.;evaluation_seconds=dict(all_splits=time.perf_counter()-index_start)
        restore(boundary)
        count=sum(x['queries'] for x in logs)
        terms={key:sum(x['queries']*x[key] for x in logs)/count if count else 0.
            for key in ('total','mse','teacher_kl','teacher_mixture')}
        terms.update(weighted_teacher_kl=settings['weight_weight']*terms['teacher_kl'],
            weighted_teacher_mixture=settings['effect_weight']*terms['teacher_mixture'])
        grad={prefix:dict(squared_sum=sum(x['grad_squared'][prefix] for x in logs),
            nonzero_steps=sum(x['grad_squared'][prefix]>0 for x in logs))
            for prefix in GRAD_PREFIXES+('bank_embedding','query_embedding')}
        row=dict(epoch=epoch,steps=steps,**scores,train_loss=terms,gradients=grad,
            seconds=time.perf_counter()-began,train_seconds=train_seconds,index_seconds=index_seconds,
            evaluation_seconds=evaluation_seconds,
            train_queries_per_second=count/train_seconds if count and train_seconds else 0.,
            actual_step_profiles=[dict(global_step=steps-len(logs)+i+1,**x['timings'])
                for i,x in enumerate(logs) if x['timings']],
            peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            runtime_plan=dict(resource),evaluation_oom_retries=eval_retries)
        print('ENCODER GRAD epoch%d | query nonzero %d/%d | bank nonzero %d/%d | encoder grad2 %.9g'%(epoch,grad['query_embedding']['nonzero_steps'],len(logs),grad['bank_embedding']['nonzero_steps'],len(logs),grad['encoder.']['squared_sum']),flush=True)
        curve.append(row)
        checkpoint=dict(epoch=epoch,epochs=epochs,steps=steps,state_dict=model.state_dict(),optimizer=opt.state_dict(),
            generator=generator.get_state(),rng_state=boundary,config=cfg,hparams=model.hparams,
            architecture=model.architecture,loss_settings=settings,metrics=scores,runtime_plan=dict(resource))
        if epoch>=1 and scores['validation']['mse']<best:
            best=scores['validation']['mse'];save(folder/'checkpoint_validation_best.pt',checkpoint)
        save(folder/('checkpoint_epoch%02d.pt'%epoch),checkpoint)
        save(last,dict(checkpoint,best=best,curve=curve));write(folder/'curve.json',curve)
        write(folder/'resource_plan.json',dict(resource,logical_batch=cfg['batch_size'],optimizer='one Adam step after full query and bank VJP',
            matcher='all2048 full-bank GEMM, no pairwise channel tensor',Teacher='exact all-legal screening, original simplex/refit'))
        write(folder/'step_profiles.json',[p for row in curve for p in row['actual_step_profiles']])
        print('EPOCH {} {} Top{} | Train MSE {:.10f} MAE {:.10f} | Validation MSE {:.10f} MAE {:.10f}'.format(
            epoch,model.architecture.get('configuration',model.architecture['variant']),model.hparams['top_k'],scores['train']['mse'],scores['train']['mae'],
            scores['validation']['mse'],scores['validation']['mae']),flush=True)
        print('COST epoch {} | train {:.3f}s | index {:.3f}s | evaluation {} | query/s {:.2f} | peak CUDA {} bytes'.format(
            epoch,train_seconds,index_seconds,evaluation_seconds,row['train_queries_per_second'],row['peak_cuda_bytes']),flush=True)
        print('PHYSICAL chunks | train query{} bank{} | evaluation query{} bank{} | logical{}'.format(
            resource['query_microbatch'],resource['encoding_chunk'],resource['eval_query_microbatch'],resource['eval_encoding_chunk'],cfg['batch_size']),flush=True)
    expected=epochs*math.ceil(data['train'].split.count/cfg['batch_size'])
    if steps!=expected:raise ValueError('Incorrect Adam update budget')
    selected=torch.load(folder/'checkpoint_validation_best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected['state_dict'],strict=True);model.eval()
    result=dict(epoch=selected['epoch'],selected=selected['metrics'],epoch10={s:curve[-1][s] for s in SPLITS},
        epoch10_epoch=epochs,runtime_plan=dict(resource),steps=steps,epochs=epochs,parameters=sum(p.numel() for p in model.parameters()),
        retriever_trainable_parameters=sum(p.numel() for m in (model.encoder,model.matcher) for p in m.parameters()),
        Teacher_used=model.options['kl'] or model.options['mixture'],teacher_support=20 if teacher is not None else None,
        loss_weights=settings,top_k=model.hparams['top_k'],selection='Validation MSE best epoch1..{}, strict comparison, earlier ties'.format(epochs))
    result['actual_end']=result['epoch10'];result['actual_end_epoch']=epochs
    result['parameter_counts']={name:sum(p.numel() for p in module.parameters()) for name,module in (
        ('encoder',model.encoder),('matcher',model.matcher),('Projection_Fusion',model.fusion))}
    result['step_profiles']=[p for row in curve for p in row['actual_step_profiles']]
    write(folder/'result.json',result)
    seal(folder,['checkpoint_initial.pt','checkpoint_validation_best.pt','checkpoint_last.pt','curve.json','result.json','step_profiles.json','resource_plan.json']+['checkpoint_epoch%02d.pt'%e for e in range(epochs+1)])
    return result
