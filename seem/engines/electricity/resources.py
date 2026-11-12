from __future__ import annotations
import json
import os
from pathlib import Path
import gc
import torch
from utils import capture, restore

CONFIG=json.loads(Path(os.environ['SEEM_CONFIG']).read_text())
HORIZONS=(96,192,336,720)
PREFIX='seem'
VARIANTS={'full':dict(history_only=False,future_value=False,cosine=False,kl=True,mixture=True,top_k=20)}
VARIANTS['bilinear64']=VARIANTS['full']
def task(ds=None,h=None):
    return CONFIG['horizons'][str(h or os.environ['SEEM_HORIZON'])]
def allowed(a,w,al,wl,h=None):
    ev=task(h=h)['evolution']
    return (a,w,al,wl)==(ev['A_steps'],ev['Writer_steps'],ev['A_lr'],ev['Writer_lr'])


def plan(h,c,logical=64):
    return dict(logical_batch=logical,
        query_microbatch=max(1,min(logical,32 if c<=32 else 16,32_000_000//(20*h*c))),
        encoding_chunk=max(1,min(128,32_000_000//((96+3*h)*c))),
        member_chunk=128 if c<=32 else 32,
        teacher_screen_chunk=max(1,32_000_000//(h*c)))


def is_cuda_oom(exc,device):
    return torch.device(device).type=='cuda' and isinstance(exc,RuntimeError) and 'out of memory' in str(exc).lower()


def shrink(p):


    for key in ('query_microbatch','encoding_chunk'):
        if p[key]>1:
            p[key]=max(1,p[key]//2);return key
    return None


def retry_backward(work,optimizer,p,device):

    state=capture();retries=0
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            value=work(p)
            return value,retries
        except RuntimeError as exc:
            if not is_cuda_oom(exc,device):raise
            error=str(exc)

        key=shrink(p)
        if key is None:raise RuntimeError('CUDA OOM even at minimum physical chunks: '+error)
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();restore(state)
        retries+=1
        print('CUDA OOM before Adam; retry SAME logical batch; {} -> {}; optimizer not stepped'.format(key,p[key]),flush=True)


def kmeans_threads(groups,h,c,requested):


    center_bytes=groups*h*c*4
    return max(1,min(requested,(16*1024**3)//max(1,center_bytes)))


def framework_plan(h,c,logical=64):
    return dict(logical_batch=logical,query_microbatch=logical,encoding_chunk=2048,teacher_screen_chunk=2048,eval_query_microbatch=64,eval_encoding_chunk=2048,direct_bank=True,recipe='full_physical64_full2048_direct_backward')


def evaluate_splits(model,data,labels,p,tag,evaluate,encode):

    device=next(model.parameters()).device;boundary=capture()
    previous=getattr(model,'runtime_plan',None)
    ep=dict(query_microbatch=p['eval_query_microbatch'],encoding_chunk=p['eval_encoding_chunk'])
    retries=0
    def attempt():
        model.runtime_plan=ep;model.eval()
        with torch.no_grad():emb=encode(model,data['train'],ep['encoding_chunk'])
        return {sp:evaluate(model,ds,labels,ep['encoding_chunk'],tag+' '+sp,emb) for sp,ds in data.items()}
    try:
        while True:
            try:
                result=attempt();break
            except RuntimeError as exc:
                if not is_cuda_oom(exc,device):raise
                message=str(exc)
            key=shrink(ep)
            if key is None:raise RuntimeError('Native evaluation OOM at minimum chunks: '+message)
            gc.collect();torch.cuda.empty_cache();restore(boundary);retries+=1
            print('Evaluation OOM; eval-only {} -> {}'.format(key,ep[key]),flush=True)
        p.update(eval_query_microbatch=ep['query_microbatch'],eval_encoding_chunk=ep['encoding_chunk'])
        return result,retries
    finally:
        restore(boundary);model.runtime_plan=previous


def streamed_teacher_batch(ds,rows,args,device):

    from framework import teacher_one
    h,c=ds.split.pred_len,ds.split.channels
    chunk=2048
    def attempt(row):return teacher_one(ds,int(row),args,device,chunk)
    parts=[]
    for row in rows:
        while True:
            try:
                part=attempt(row);break
            except RuntimeError as exc:
                if not is_cuda_oom(exc,device) or chunk==1:raise
            chunk=max(1,chunk//2);gc.collect();torch.cuda.empty_cache()
            print('Teacher screening OOM; same full dictionary, screen chunk{}'.format(chunk),flush=True)
        parts.append(part)
    return {k:torch.cat([part[k] for part in parts]) for k in parts[0]}
