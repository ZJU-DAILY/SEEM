from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from utils import read, write
from data import digest


KMEANS=dict(init='k-means++',n_init=3,max_iter=50,tol=1e-4,random_state=20260915,algorithm='lloyd')

def fit_mapping(vectors,groups):
    import sklearn
    from sklearn.cluster import KMeans
    start=time.perf_counter()
    model=KMeans(n_clusters=groups,**KMEANS).fit(vectors)
    return model.labels_,dict(backend='sklearn.cluster.KMeans',version=sklearn.__version__,seconds=time.perf_counter()-start,n_iter=int(model.n_iter_),inertia=float(model.inertia_))

def array_hash(x):
    a=np.ascontiguousarray(x)
    h=hashlib.sha256(str((a.shape,a.dtype.str)).encode())
    view=memoryview(a).cast('B')
    for i in range(0,len(view),8*1024*1024):h.update(view[i:i+8*1024*1024])
    return h.hexdigest()

class GlobalData:

    def values(self, rows, ids, device):
        chosen = ids.detach().cpu().numpy()
        result = np.stack([self.bank.read_values(chosen[j], self.low[row], self.high[row]) for j, row in enumerate(rows)])
        return torch.from_numpy(result).to(device)

    def forecast_inputs(self, rows, device):
        index = torch.from_numpy(np.asarray(rows, dtype=np.int64))
        return (self.split.histories.index_select(0, index).to(device), self.split.backbone_prediction.index_select(0, index).to(device))

class Data(GlobalData):

    def queries(self, rows, device):
        rows = np.asarray(rows, dtype=np.int64)
        index = torch.as_tensor(rows, dtype=torch.long, device=self.keys.device)
        return (self.keys.index_select(0, index).to(device), torch.from_numpy(self.low[rows]).to(device), torch.from_numpy(self.high[rows]).to(device))

def mmap_array(path, shape, fill):
    path=Path(path)
    if not path.exists():
        tmp=Path(str(path)+'.partial')
        a=np.lib.format.open_memmap(tmp, mode='w+', dtype='float32', shape=shape)
        fill(a);a.flush();del a;tmp.replace(path)
    a=np.load(path,mmap_mode='c')
    if a.shape!=tuple(shape) or a.dtype!=np.float32:raise ValueError('Wrong cached shape: '+str(path))
    return a

class Bank:

    def __init__(self,keys,values,labels,groups,folder):
        self.keys=np.asarray(keys);self.labels=np.asarray(labels,dtype=np.int64)
        self.members=[np.flatnonzero(self.labels==g) for g in range(groups)]
        self.counts=np.asarray([len(m) for m in self.members])
        if (self.counts==0).any():raise ValueError('Empty memory group')
        folder=Path(folder);folder.mkdir(exist_ok=True)
        n,h,c=values.shape;offset=np.r_[0,np.cumsum(self.counts+1)]
        pp=folder/'prefix.npy';ss=folder/'sums.npy';mm=folder/'means.npy'
        seal=folder/'bank.json'
        if not seal.exists():
            prefix=np.lib.format.open_memmap(str(pp)+'.partial',mode='w+',dtype='float64',shape=(n+groups,h,c))
            sums=np.lib.format.open_memmap(str(ss)+'.partial',mode='w+',dtype='float64',shape=(groups,h,c))
            means=np.lib.format.open_memmap(str(mm)+'.partial',mode='w+',dtype='float32',shape=(groups,h,c))
            for g,m in enumerate(self.members):
                lo,hi=offset[g:g+2];prefix[lo]=0

                np.cumsum(values[m],axis=0,dtype=np.float64,out=prefix[lo+1:hi])
                sums[g]=prefix[hi-1];means[g]=sums[g]/len(m)
            for a in (prefix,sums,means):a.flush()
            del prefix,sums,means
            for p in (pp,ss,mm):Path(str(p)+'.partial').replace(p)
            write(seal,dict(files={p.name:digest(p) for p in (pp,ss,mm)}))
        else:
            for name,sha in read(seal)['files'].items():
                if digest(folder/name)!=sha:raise ValueError('Bank changed: '+name)
        self.offset=offset
        self.prefix=np.load(pp,mmap_mode='r')
        self.value_prefix=[self.prefix[offset[g]:offset[g+1]] for g in range(groups)]
        self.shared=SimpleNamespace(values=values[None],sums=[np.load(ss,mmap_mode='r')],
            means=[np.load(mm,mmap_mode='r')],group_ids=[self.labels],members=[self.members])

    def interval(self,g,low,high):
        first,last=np.searchsorted(self.members[g],[low,high])
        return first,last,int(self.counts[g]-(last-first))

    def read_values(self,groups,low,high):
        result=[]
        for g in groups:
            first,last,count=self.interval(g,low,high)
            if count<1:raise ValueError('Empty legal Value')
            v=self.shared.means[0][g]
            if last>first:
                off=self.offset[g]
                v=((self.shared.sums[0][g]-(self.prefix[off+last]-self.prefix[off+first]))/count).astype('float32')
            result.append(v)
        return np.stack(result)

def build_bank(train,keys,folder,groups=2048):
    folder=Path(folder);folder.mkdir(exist_ok=True)
    n,h,c=train.futures.shape
    def fill(a):
        for i in range(0,n,64):a[i:i+64]=(train.futures[i:i+64]-train.histories[i:i+64,-1:]).numpy()
    values=mmap_array(folder/'continuations.npy',(n,h,c),fill)
    mapping=folder/'l_continuation_group_mapping.npz';seal=folder/'regrouping.json'
    binding=dict(continuation_sha256=digest(folder/'continuations.npy'),groups=groups,kmeans=KMEANS)
    if seal.exists():
        info=read(seal)
        if info['binding']!=binding or digest(mapping)!=info['mapping_sha256']:raise ValueError('Changed grouping')
        with np.load(mapping) as f:
            np.testing.assert_array_equal(f['canonical_source_idx'],train.source_sample_index);labels=f['group_ids'].copy()
    else:
        print('Fresh exact KMeans | {} x {} -> {} groups'.format(n,h*c,groups),flush=True)
        labels,info=fit_mapping(values.reshape(n,-1),groups)
        if not np.array_equal(np.unique(labels),np.arange(groups)):raise ValueError('KMeans empty group')
        with (folder/'mapping.partial').open('wb') as f:np.savez_compressed(f,group_ids=labels,canonical_source_idx=train.source_sample_index)
        (folder/'mapping.partial').replace(mapping)
        write(seal,dict(binding=binding,mapping_sha256=digest(mapping),fit=info))
    return Bank(keys,values,labels,groups,folder/'bank')

def select(keys, members, history_size):
    keys=np.asarray(keys,dtype=np.float32)
    if not np.isfinite(keys).all():raise ValueError('Nonfinite Key')
    rest=keys.shape[1]-history_size
    if history_size<=0 or rest<=0 or rest%3:raise ValueError('Wrong four-view dimensions')
    cuts=[0,history_size]+[history_size+rest//3*i for i in (1,2,3)]
    reps=[]
    for members_g in members:
        rows=np.sort(np.asarray(members_g,dtype=np.int64))
        if not len(rows):raise ValueError('Empty group')
        distance=np.zeros(len(rows),dtype=np.float64)
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            part=keys[rows,lo:hi].astype('float64')
            part/=np.maximum(np.linalg.norm(part,axis=1,keepdims=True),1e-12)
            distance+=((part-part.mean(0))**2).sum(1)/4
        reps.append(int(rows[np.argmin(distance)]))
    return np.asarray(reps,dtype=np.int64)

def configure(model,bank,data,out):
    keys=data['train'].keys.detach().cpu().numpy()
    reps=select(keys,bank.members,bank.keys.shape[1])
    bank.representatives=reps

    model.register_buffer('representative_rows',torch.as_tensor(reps,device=next(model.parameters()).device),persistent=False)
    manifest=dict(rule='nearest real member to four independently L2-normalized view centroids; equal weights; row-order ties',
        groups=len(reps),source_members=len(keys),packed_keys_sha256=array_hash(keys),
        representatives_sha256=array_hash(reps),representative_rows=reps.tolist(),
        value_sources_unchanged=True,representative_overlap_masks_entire_group=True)
    path=Path(out)/'single_key_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Single-Key binding changed')
    path.write_text(json.dumps(manifest,indent=2))
    summary={}
    for name,ds in data.items():
        blocked=((reps[None]>=ds.low[:,None]) & (reps[None]<ds.high[:,None])).sum(1)
        legal=len(reps)-blocked
        if legal.min()<20:raise ValueError('Fewer than 20 legal representative groups: '+name)
        summary[name]=dict(queries=len(legal),mean_blocked_groups=float(blocked.mean()),minimum_legal_groups=int(legal.min()))
    (Path(out)/'single_key_legality.json').write_text(json.dumps(summary,indent=2))
    print('Single Key | {} groups | {} scored addresses | {} original content sources'.format(len(reps),len(reps),len(keys)),flush=True)
