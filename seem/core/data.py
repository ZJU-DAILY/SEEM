from __future__ import annotations
import csv
import hashlib
from pathlib import Path
import numpy as np
import utils
from torch.nn import functional as F
from types import SimpleNamespace
import torch
from utils import read, write


CATALOG = {
    'ETTh1': ('ETT-small/ETTh1.csv', 7, 'hour'),
    'ETTh2': ('ETT-small/ETTh2.csv', 7, 'hour'),
    'ETTm1': ('ETT-small/ETTm1.csv', 7, 'minute'),
    'ETTm2': ('ETT-small/ETTm2.csv', 7, 'minute'),
    'Electricity': ('electricity/electricity.csv', 321, 'custom'),
    'Weather': ('weather/weather.csv', 21, 'custom'),
    'Traffic': ('traffic/traffic.csv', 862, 'custom'),
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def borders(name,n,seq_len=96):
    kind=CATALOG[name][2]
    if kind!='custom':
        factor=4 if kind=='minute' else 1
        train=12*30*24*factor;valid=4*30*24*factor;test=valid
        if n<train+valid+test:raise ValueError('ETT file too short')
    else:
        train=int(n*.7);test=int(n*.2);valid=n-train-test
    return {'train':(0,train),'validation':(train-seq_len,train+valid),
            'test':(train+valid-seq_len,train+valid+test)}


def load(root,name,horizon=96):
    rel,c,_=CATALOG[name];path=Path(root)/'dataset'/rel
    if not path.is_file():
        flat=Path(root)/'dataset'/Path(rel).name
        if flat.is_file():path=flat
        else:raise FileNotFoundError(str(path))
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.reader(f);header=next(reader)
        if header[0]!='date' or len(header)!=c+1:raise ValueError('Unexpected CSV columns: '+str(path))
        rows=[];dates=[]
        for row in reader:
            if len(row)!=len(header):raise ValueError('Ragged CSV')
            dates.append(row[0]);rows.append([float(v) for v in row[1:]])
    x=np.asarray(rows,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Nonfinite CSV value')
    bounds=borders(name,len(x));end=bounds['train'][1]
    mean=x[:end].mean(0);scale=x[:end].std(0);scale[scale<1e-12]=1.
    z=((x-mean)/scale).astype('float32')
    return z,dict(dataset=name,csv=str(path.resolve()),csv_sha256=digest(path),rows=len(x),channels=c,
        columns=header[1:],borders=bounds,scaler_fit_rows=[0,end],mean=mean.tolist(),scale=scale.tolist(),
        duplicate_timestamps=len(dates)-len(set(dates)),row_order_preserved=True,features='M',seq_len=96,pred_len=horizon)


def windows(values,border,length=96,horizon=96):
    lo,hi=border;ids=np.arange(lo,hi-length-horizon+1,dtype=np.int64)
    if not len(ids):raise ValueError('Empty split')

    histories=np.lib.stride_tricks.sliding_window_view(values,length,axis=0).transpose(0,2,1)
    futures=np.lib.stride_tricks.sliding_window_view(values,horizon,axis=0).transpose(0,2,1)
    return histories[lo:hi-length-horizon+1],futures[lo+length:hi-horizon+1],ids


def prepare(z,meta,model_name,out,device,record):
    from backbone import load_backbone, calendar, marks
    out.mkdir(parents=True,exist_ok=True);c=meta['channels'];horizon=meta['pred_len']
    if record['config']['pred_len']!=horizon:raise ValueError('Backbone horizon mismatch')
    model,record=load_backbone(record,device)
    before=utils.state_fingerprint(model)
    stamps=calendar(meta,record);batch_size=record['inference_batch_size']
    splits={};keys={};scores={}
    for name,border in meta['borders'].items():
        x,y,ids=windows(z,border,horizon=horizon);n=len(ids)
        paths={k:out/(name+'_'+k+'.npy') for k in ('prediction','keys')};seal=out/(name+'.json')
        binding=dict(csv_sha256=meta['csv_sha256'],checkpoint_sha256=record['sha256'],model=model_name,config=record['config'],inference_batch_size=batch_size,count=n,channels=c,marks='original calendar fixed across input probes')
        if seal.exists():
            cached=read(seal)
            if cached['binding']!=binding:raise ValueError('Backbone cache binding changed')
            for k,p in paths.items():
                if digest(p)!=cached['files'][k]:raise ValueError('Changed backbone cache '+str(p))
        else:
            pred=np.lib.format.open_memmap(str(paths['prediction'])+'.partial',mode='w+',dtype='float32',shape=(n,horizon,c))
            key=np.lib.format.open_memmap(str(paths['keys'])+'.partial',mode='w+',dtype='float32',shape=(n,(96+3*horizon)*c))
            with torch.no_grad():
                for start in range(0,n,batch_size):
                    stop=min(start+batch_size,n);h=torch.from_numpy(x[start:stop].copy()).to(device)
                    hm,fm=marks(stamps,ids[start:stop],record,device);b=model(h,hm,fm)
                    hist=utils.normalized_flattened(utils.multigrain_centered(h,(1,)))[0]
                    response=torch.stack([model(v,hm,fm)-b for v in transformed_histories(h)],1)
                    pred[start:stop]=b.cpu().numpy();key[start:stop]=torch.cat((hist,response.flatten(1)),1).cpu().numpy()
                    if start%4096==0:print('Frozen '+model_name+' / Keys | {} | {}/{}'.format(name,stop,n),flush=True)
            pred.flush();key.flush();del pred,key
            for p in paths.values():Path(str(p)+'.partial').replace(p)
            write(seal,dict(binding=binding,files={k:digest(p) for k,p in paths.items()}))

        pred=torch.from_numpy(np.load(paths['prediction'],mmap_mode='c'))
        histories=torch.from_numpy(x.copy());futures=torch.from_numpy(y.copy())
        splits[name]=SimpleNamespace(name=name,histories=histories,futures=futures,backbone_prediction=pred,
            source_sample_index=ids,count=n,seq_len=96,pred_len=horizon,channels=c)
        keys[name]=torch.from_numpy(np.load(paths['keys'],mmap_mode='c'))
        mse=mae=0.
        for i in range(0,n,64):
            error=(pred[i:i+64]-futures[i:i+64]).double();mse+=float(error.square().sum());mae+=float(error.abs().sum())
        scores[name]=dict(mse=mse/(n*horizon*c),mae=mae/(n*horizon*c),queries=n)
    if utils.state_fingerprint(model)!=before or any(v.grad is not None for v in model.parameters()):raise ValueError('Backbone state changed')
    write(out/'backbone_metrics_unverified.json',scores)
    write(out/'backbone_verification.json',dict(frozen=True,strict_checkpoint_load=True,checkpoint_sha256=record['sha256']))
    write(out/'backbone_metrics.json',scores)
    return splits,keys


def make_data(splits,keys,bank):
    from memory import Data
    data={}
    for name,s in splits.items():
        ds=Data.__new__(Data);ds.split=s;ds.keys=keys[name];ds.bank=bank
        ds.low,ds.high=utils.forbidden_bounds(splits['train'].source_sample_index,s.source_sample_index,s.seq_len,s.pred_len,name=='train')
        data[name]=ds
    return data

def transformed_histories(history):

    if history.ndim != 3 or history.shape[1] < 9:
        raise ValueError("Expected history [N,T,C] with at least nine steps.")
    recent = history.clone()
    recent[:, -8:] = history[:, -9:-8].expand(-1, 8, -1)
    padded = torch.cat((history[:, :1].expand(-1, 2, -1), history,
                        history[:, -1:].expand(-1, 2, -1)), dim=1)
    smooth = F.avg_pool1d(padded.transpose(1, 2), kernel_size=5, stride=1).transpose(1, 2)
    scale = history[:, -1:] + 1.05 * (history - history[:, -1:])
    return recent, smooth, scale
