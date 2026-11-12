#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATASETS=('ETTh1','ETTh2','ETTm1','ETTm2','Weather','Electricity','Traffic')
def configure(dataset,horizon,seed,config=None):
    path=Path(config).resolve() if config else ROOT/'configs'/(dataset+'.json')
    cfg=json.loads(path.read_text())
    if cfg['dataset']!=dataset:raise ValueError('Configuration dataset mismatch')
    engine=cfg['engine']
    os.environ.update(SEEM_CONFIG=str(path),SEEM_HORIZON=str(horizon),SEEM_RUN_SEED=str(seed))
    for prefix in ('ETTH','ETTM','WEATHER','ECL','TRAFFIC'):
        os.environ[prefix+'_ABLATION_DATASET']=dataset
        os.environ[prefix+'_ABLATION_HORIZON']=str(horizon)
    sys.path[:0]=[str(ROOT/'seem/engines'/engine),str(ROOT/'seem/core')]
    return cfg,engine

def main():
    p=argparse.ArgumentParser(description='Train MLP + SEEM on one benchmark/horizon/seed.')
    p.add_argument('--dataset',choices=DATASETS,required=True)
    p.add_argument('--horizon',type=int,choices=[96,192,336,720],required=True)
    p.add_argument('--seed',type=int,choices=[42],default=42)
    p.add_argument('--config',type=Path)
    p.add_argument('--data-root',type=Path,default=ROOT)
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    p.add_argument('--device',default='cuda');p.add_argument('--cpu-threads',type=int,default=8)
    p.add_argument('--backbone-checkpoint',type=Path,help='Optional pretrained ResidualMLP state_dict; otherwise train once per dataset/horizon.')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();cfg,engine=configure(a.dataset,a.horizon,a.seed,a.config)
    if a.dry_run:
        print(json.dumps(dict(dataset=a.dataset,horizon=a.horizon,seed=a.seed,engine=engine,task=cfg['horizons'][str(a.horizon)],output=str(a.output_root/a.dataset/str(a.horizon)/str(a.seed))),indent=2));return
    from pipeline import execute
    execute(a,cfg)
if __name__=='__main__':main()
