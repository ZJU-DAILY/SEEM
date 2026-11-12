#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
DATASETS=('ETTh1','ETTh2','ETTm1','ETTm2','Weather','Electricity','Traffic')
SEEDS=(42,)
def main():
    p=argparse.ArgumentParser(description='Summarize saved validation-selected results; never run model evaluation.')
    p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent/'outputs')
    a=p.parse_args();rows=[];done=0
    for ds in DATASETS:
        for h in (96,192,336,720):
            scores=[]
            for seed in SEEDS:
                folder=a.output_root/ds/str(h)/str(seed)
                path=folder/'evolution/selection_report.json'
                if not (folder/'complete.json').exists() or not path.exists():continue
                obj=json.loads(path.read_text())
                scores.append(obj['metrics']['test']);done+=1
            def fmt(key):
                if not scores:return '—'
                return '%.6f'%scores[0][key]
            rows.append((ds,h,len(scores),fmt('mse'),fmt('mae')))
    print('Completed: %d/28; seed: 42; selection: round_end Validation MSE. Missing results are not zero.'%done)
    print('| Dataset | H | Seeds | Test MSE | Test MAE |')
    print('|---|---:|---:|---|---|')
    for ds,h,n,mse,mae in rows:print('| %s | %d | %d/1 | %s | %s |'%(ds,h,n,mse,mae))
if __name__=='__main__':main()
