from __future__ import annotations
import torch
from data import digest
from networks import ResidualMLP
import fcntl
from data import windows
from utils import read, write, save, bind
from utils import capture, restore
from utils import seed_everything


def load_backbone(record,device):
    if digest(record['path'])!=record['sha256']:raise ValueError('MLP checkpoint changed')
    m=ResidualMLP(**record['config']).to(device)
    m.load_state_dict(torch.load(record['path'],map_location='cpu',weights_only=True),strict=True)
    m.eval().requires_grad_(False)
    return m,record


def calendar(meta,record):return None


def marks(stamps,source_ids,record,device):return None,None


@torch.no_grad()
def evaluate(model,x,y,device,batch):
    model.eval();se=0.
    for i in range(0,len(x),batch):
        err=model(torch.tensor(x[i:i+batch],device=device))-torch.tensor(y[i:i+batch],device=device)
        se+=float(err.double().square().sum())
    return se/y.size

def train(z,meta,cfg,folder,device,resume):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        config={k:cfg[k] for k in ['seq_len','hidden','dropout','eps']};config['pred_len']=meta['pred_len']
        protocol=dict(config=config,training=cfg,csv_sha256=meta['csv_sha256'])
        if (folder/'protocol.json').exists() and read(folder/'protocol.json')!=protocol:raise ValueError('Changed backbone configuration')
        write(folder/'protocol.json',protocol)
        if (folder/'complete.json').exists():return read(folder/'complete.json')
        if (folder/'last.pt').exists() and not resume:raise ValueError('Backbone training exists: use --resume')
        seed_everything(cfg['seed']);model=ResidualMLP(**config).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        x,y,_=windows(z,meta['borders']['train'],horizon=meta['pred_len']);vx,vy,_=windows(z,meta['borders']['validation'],horizon=meta['pred_len'])
        best=float('inf');best_epoch=0;bad=0;start=1;history=[]
        if (folder/'last.pt').exists():
            ck=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore(ck['rng'])
            best=ck['best'];best_epoch=ck['best_epoch'];bad=ck['bad'];start=ck['epoch']+1;history=ck['history']
        for ep in range(start,cfg['epochs']+1):
            if bad>=cfg['patience']:break
            model.train();order=torch.randperm(len(x)).numpy();total=0.
            for i in range(0,len(x),cfg['batch_size']):
                rows=order[i:i+cfg['batch_size']];h=torch.tensor(x[rows],device=device);target=torch.tensor(y[rows],device=device)
                opt.zero_grad(set_to_none=True);loss=(model(h)-target).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite backbone loss')
                loss.backward();opt.step();total+=float(loss.detach())*len(rows)
            val=evaluate(model,vx,vy,device,cfg['batch_size'])
            if val<best:best=val;best_epoch=ep;bad=0;save(folder/'best.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
            else:bad+=1
            history.append(dict(epoch=ep,train_online_mse=total/len(x),validation_mse=val))
            save(folder/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=capture(),best=best,best_epoch=best_epoch,bad=bad,epoch=ep,history=history))
            write(folder/'history.json',history);print('MLP epoch',ep,'Validation MSE',val,flush=True)
        record=dict(path=str((folder/'best.pt').resolve()),sha256=digest(folder/'best.pt'),config=config,inference_batch_size=64,selected_epoch=best_epoch)
        write(folder/'complete.json',record);return record
