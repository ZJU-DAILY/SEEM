from __future__ import annotations
import os
from pathlib import Path
from data import digest
import numpy as np
import torch
from results import read, write, save


def load_cpu(path):
    path=str(path)
    try:return torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    except TypeError:return torch.load(path,map_location='cpu',weights_only=True)

def check_part(part,ds,binding):
    if part['binding']!=binding:raise ValueError('M0 cache binding differs')
    source_ids=torch.as_tensor(ds.split.source_sample_index)
    if not torch.equal(part['source_ids'],source_ids):raise ValueError('M0 cache row identities differ')
    cache=part['cache'];shape=tuple(ds.split.futures.shape)
    if any(tuple(cache[k].shape)!=shape for k in ('prediction','target')):
        raise ValueError('M0 cache prediction/target shape differs')
    if tuple(cache['ids'].shape)!=(len(source_ids),20) or cache['weights'].shape!=cache['ids'].shape:
        raise ValueError('M0 cache Top20 shape differs')
    return cache

@torch.no_grad()
def prepare(model,data,labels,args,device,binding,out,reader):
    out=Path(out);global_path=out/'frozen_cache.pt';global_meta=out/'frozen_cache.json'
    if global_path.exists():
        if not global_meta.exists() or read(global_meta)['binding']!=binding:
            raise ValueError('Existing original frozen cache has no matching binding')
        print('REUSE original completed frozen_cache.pt | no model read or affine probe',flush=True)
        caches=load_cpu(global_path)['caches']
        for sp,ds in data.items():
            if tuple(caches[sp]['prediction'].shape)!=tuple(ds.split.futures.shape):raise ValueError('Existing cache shape differs')
        return caches
    folder=out/'m0_split_cache';folder.mkdir(exist_ok=True);caches={};missing=[]
    for sp,ds in data.items():
        path=folder/(sp+'.pt');meta_path=folder/(sp+'.json')
        if path.exists():
            part=load_cpu(path);caches[sp]=check_part(part,ds,binding)
            if meta_path.exists():
                meta=read(meta_path)
                if meta['binding']!=binding or path.stat().st_size!=meta['bytes']:
                    raise ValueError('Completed split cache metadata differs')
            print('REUSE completed M0 cache split',sp,flush=True)
        else:missing.append(sp)
    if not missing:return caches
    keys=data['train'].keys[data['train'].bank.representatives].to(device)
    embeddings=model.encode_bank(keys,args.encoding_chunk)
    del keys
    for sp in missing:
        ds=data[sp];parts={k:[] for k in ('prediction','target','ids','weights')}
        batch=getattr(args,'query_batch',64)
        for first in range(0,ds.split.count,batch):
            rows=np.arange(first,min(first+batch,ds.split.count))
            prediction,result=reader(model,ds,rows,embeddings,labels,device)
            for key,value in [('prediction',prediction),('target',ds.split.futures[rows]),
                              ('ids',result['group_ids']),('weights',result['weights'])]:
                parts[key].append(value.detach().cpu())
            if first%2048==0 or first+len(rows)==ds.split.count:
                print('M0 cache | %s | %d/%d | no affine/FP64 audit'%(sp,first+len(rows),ds.split.count),flush=True)
            del prediction,result
        part={k:torch.cat(values) for k,values in parts.items()};del parts
        path=folder/(sp+'.pt')
        save(path,dict(binding=binding,source_ids=torch.as_tensor(ds.split.source_sample_index),cache=part))
        write(folder/(sp+'.json'),dict(binding=binding,queries=ds.split.count,bytes=path.stat().st_size,
            affine_probe=False,FP64_diagnostic=False))
        caches[sp]=part
        print('SAVED M0 cache split %s | later resumes skip this split'%sp,flush=True)
    return caches

def cached_metrics(caches,chunk=64):
    result={}
    for sp,cache in caches.items():
        squared=absolute=0.;n=cache['target'].numel()
        for first in range(0,len(cache['target']),chunk):
            error=cache['prediction'][first:first+chunk].double()-cache['target'][first:first+chunk].double()
            squared+=float(error.square().sum());absolute+=float(error.abs().sum())
        result[sp]=dict(mse=squared/n,mae=absolute/n)
    return result

def open_observables(folder,split,shape):
    return np.lib.format.open_memmap(str(folder/(split+'_observables.tmp.npy')),mode='w+',dtype=np.float32,shape=shape)

def finish_observables(folder,split,matrix):
    matrix.flush(); shape=list(matrix.shape)
    target=folder/(split+'_observables.npy')
    os.replace(str(folder/(split+'_observables.tmp.npy')),str(target))
    return dict(file=target.name,shape=shape,sha256=digest(target))

def load_observables(folder,record):
    path=folder/record['file']
    if digest(path)!=record['sha256']:raise ValueError('Observable cache changed')
    value=np.load(path,mmap_mode='c')
    if list(value.shape)!=record['shape'] or value.dtype!=np.float32:raise ValueError('Observable shape/dtype changed')
    return torch.from_numpy(value)
