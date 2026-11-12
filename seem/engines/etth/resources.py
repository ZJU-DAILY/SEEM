from __future__ import annotations
import json
import os
from pathlib import Path
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

    optimizer.zero_grad(set_to_none=True)
    return work(p),0


def kmeans_threads(groups,h,c,requested):


    center_bytes=groups*h*c*4
    return max(1,min(requested,(16*1024**3)//max(1,center_bytes)))


def framework_plan(h,c,logical=None):
    expected=task(h=h)['framework']['batch_size']
    if logical is not None and logical!=expected:raise ValueError('Batch must match this horizon Full')
    return dict(logical_batch=expected,query_microbatch=expected,encoding_chunk=2048,
        eval_query_microbatch=expected,eval_encoding_chunk=2048,recipe='ETTh_Full_matched_full_query_full2048_graph')


def evaluate_splits(model,data,labels,p,tag,evaluate,encode):
    device=next(model.parameters()).device;boundary=capture()
    previous=getattr(model,'runtime_plan',None)
    ep=dict(query_microbatch=p['eval_query_microbatch'],encoding_chunk=p['eval_encoding_chunk'])
    try:
        model.runtime_plan=ep;model.eval()
        with torch.no_grad():emb=encode(model,data['train'],ep['encoding_chunk'])
        return {sp:evaluate(model,ds,labels,ep['encoding_chunk'],tag+' '+sp,emb) for sp,ds in data.items()},0
    finally:
        restore(boundary);model.runtime_plan=previous


def original_teacher_batch(ds,rows,args,device):
    from framework import teacher_one
    parts=[teacher_one(ds,int(row),args,device,2048) for row in rows]
    return {key:torch.cat([part[key] for part in parts]) for key in parts[0]}

_GPU_TEACHER=None
def streamed_teacher_batch(ds,rows,args,device):
    global _GPU_TEACHER
    if torch.device(device).type!='cuda':return original_teacher_batch(ds,rows,args,device)
    if _GPU_TEACHER is None:
        from teacher import GPUTeacher
        _GPU_TEACHER=GPUTeacher(original_teacher_batch,max_queries=64)
    return _GPU_TEACHER(ds,rows,args,device)

def release_teacher():
    global _GPU_TEACHER
    if _GPU_TEACHER is not None:_GPU_TEACHER.release();_GPU_TEACHER=None
