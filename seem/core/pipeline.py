from __future__ import annotations
from framework import read_forecast
import utils
import fcntl
import inspect
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from utils import read, write, save, bind
from data import load, digest
from backbone import train
from framework import new_model
from data import prepare as prepare_backbone, make_data
from memory import build_bank
from memory import configure
from training import build_teacher, train_r0
from resources import framework_plan
from horizons import expected_counts, role_split
from cache import prepare as prepare_cache


def finish(out):
    from results import persist_results, compact_complete
    from evolution import read_endpoint
    evo=out/'evolution';tr=evo/'trajectory';selection=read(tr/'selection.json')
    loadpt=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    geometry=loadpt(evo/'frozen_export_geometry.pt');binding=geometry['binding']
    if not (evo/'finalization.json').exists():
        selected=read_endpoint(tr/('round%d/actual_end.pt'%selection['round']),binding)
        persist_results(evo,selected,loadpt(tr/'states.pt'),read(tr/'history.json'),selection,geometry)
    protocol=read(out/'protocol.json')
    write(out/'summary.json',dict(dataset=protocol['dataset'],pred_len=protocol['pred_len'],seed=protocol['seed'],framework=read(out/'framework/result.json'),selection=selection))
    (out/'report.md').write_text((evo/'report.md').read_text())
    compact_complete(out)
    print_final(out)

def print_final(out):
    selection=read(out/'evolution/selection_report.json')
    metrics=selection['metrics']['test']
    print('FINAL round-end Validation MSE selection | R%d | Test MSE %.10f MAE %.10f'%(selection['round'],metrics['mse'],metrics['mae']),flush=True)

def execute(a,config):
    out=(a.output_root/a.dataset/str(a.horizon)/str(a.seed)).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    with (out.parent/(out.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'complete.json').exists():
            print('Already complete:',out);print_final(out);return
        if (out/'evolution/trajectory/selection.json').exists():finish(out);return
        device=utils.resolve_device(a.device);torch.set_num_threads(a.cpu_threads)
        if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        t=config['horizons'][str(a.horizon)];fw=t['framework'];ev=t['evolution']
        z,meta=load(a.data_root,a.dataset,a.horizon)
        shared=(a.output_root/'shared'/a.dataset/str(a.horizon)).resolve();shared.mkdir(parents=True,exist_ok=True)
        if a.backbone_checkpoint:
            path=a.backbone_checkpoint.resolve()
            weight=dict(path=str(path),sha256=digest(path),config=dict(config['backbone'],pred_len=a.horizon),inference_batch_size=64)
        else:weight=train(z,meta,config['backbone'],shared/'mlp',device,a.resume)
        cfg=dict(seq_len=96,pred_len=a.horizon,channels=meta['channels'],seed=a.seed,seeds=[a.seed],train_epochs=fw['epochs'],batch_size=fw['batch_size'],learning_rate=fw['learning_rate'],weight_decay=fw['weight_decay'],bilinear_rank=64)
        hp=config['hparams'];settings=config['teacher'];roles=role_split(expected_counts(a.horizon)['train'],a.horizon)
        code_root=Path(__file__).resolve().parents[1]
        code_digest=hashlib.sha256()
        for code in sorted(code_root.rglob('*.py')):
            code_digest.update(str(code.relative_to(code_root)).encode());code_digest.update(code.read_bytes())
        protocol=dict(network=config['network'],code_sha256=code_digest.hexdigest(),dataset=a.dataset,pred_len=a.horizon,seed=a.seed,config=cfg,hparams=hp,framework=fw,evolution=ev,roles=roles,backbone=weight,csv_sha256=meta['csv_sha256'],teacher=settings)
        bind(out,protocol,a.resume)
        with (shared/'prepare.lock').open('a') as shared_lock:
            fcntl.flock(shared_lock,fcntl.LOCK_EX)
            splits,keys=prepare_backbone(z,meta,'ResidualMLP',shared/'backbone',device,weight)
            bank=build_bank(splits['train'],keys['train'][:,:96*meta['channels']].numpy(),shared/'memory',config['groups'])
            data=make_data(splits,keys,bank)
            model=new_model(cfg,hp,'bilinear64',np.arange(config['groups']),device)
            configure(model,bank,data,shared/'memory')
            labels=torch.as_tensor(bank.labels,dtype=torch.long,device=device)
            kwargs={'logical':fw['batch_size']} if 'logical' in inspect.signature(framework_plan).parameters else {}
            model.runtime_plan=framework_plan(a.horizon,meta['channels'],**kwargs)
            teacher=None if (out/'framework/complete.json').exists() else build_teacher(model,data,settings,shared/'teacher',out/'teacher',dict(csv_sha256=meta['csv_sha256'],backbone=weight['sha256'],horizon=a.horizon,groups=config['groups']),device)
        result=train_r0(model,data,labels,cfg,settings,teacher,out/'framework',device,epochs=fw['epochs'],encoding=2048)
        del teacher
        ck=out/'framework_selected.pt'
        if not ck.exists():save(ck,dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,hparams=hp,architecture=model.architecture,group_labels=labels.cpu(),representative_rows=torch.as_tensor(bank.representatives),metrics=result['selected'],epoch=result['epoch']))
        model.eval().requires_grad_(False)
        for parameter in model.parameters():parameter.grad=None
        args=SimpleNamespace(encoding_chunk=2048,query_batch=64,seed=a.seed,arm='static')
        binding=dict(protocol_sha256=digest(out/'protocol.json'),framework_sha256=digest(ck),pred_len=a.horizon)
        if a.dataset=='Traffic':
            import runtime as execution
            execution.configure(out/'evolution',requested_rows=128)
        original=prepare_cache(model,data,labels,args,device,binding,out,read_forecast)
        values=np.load(shared/'memory/continuations.npy',mmap_mode='c')
        if a.dataset=='Traffic':
            scale=execution.channel_std(torch.from_numpy(values),len(values))
            if a.horizon==96:scale=scale.float()
        else:scale=torch.from_numpy(values).reshape(-1,meta['channels']).std(0,unbiased=False)
        scale=scale.clamp_min(1e-5).to(device)
        from evolution import execute_dynamic
        execute_dynamic(model,data,labels,args,original,model.fusion.retrieval_pred[0].weight.detach(),model.fusion.linear_pred.weight[:,a.horizon:].detach(),scale,out/'evolution',device,binding,A_steps=ev['A_steps'],W_steps=ev['Writer_steps'],A_lr=ev['A_lr'],W_lr=ev['Writer_lr'],rounds=5,a_end=roles['A'][1],b_start=roles['B'][0])
        finish(out)
