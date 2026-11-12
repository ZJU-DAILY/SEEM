from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import torch


def read(path):return json.loads(Path(path).read_text())
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(str(temp),str(path))
def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.partial');torch.save(obj,temp);os.replace(str(temp),str(path))
def pair(m):return '%.10f / %.10f'%(m['mse'],m['mae'])
def check_selection(states,history,selection):
 settings=selection['settings'];width=settings['A_steps']+settings['Writer_steps'];total=width*5
 if settings['rounds']!=5 or selection['formal_write_count']!=5 or selection['actual_final_endpoint']!=total:raise ValueError('Five completed writes required')
 if len(history)!=total+1 or [x['global_step'] for x in history]!=list(range(total+1)):raise ValueError('All saved updates required')
 eligible=[width*i for i in range(1,6)]
 if selection['eligible_global_steps']!=eligible:raise ValueError('Wrong round-end selection scope')
 best=history[width]
 for i in eligible[1:]:
  if history[i]['metrics']['validation']['mse']<best['metrics']['validation']['mse']-1e-12:best=history[i]
 if (best['global_step'],best['metrics'])!=(selection['global_step'],selection['metrics']):raise ValueError('Saved round selection differs from saved Validation')
 if states['RoundEnd_Validation_selected']['metrics']!=best['metrics']:raise ValueError('Saved metric states differ')

def artifact(state,base):
 keys=('format','query_mapping','scale','response_scale','projection','fusion','group_features','horizon','binding')
 result={k:base[k] for k in keys}
 result.update(content=state['content'],applicability=state['applicability'],round=state['round'],phase=state['phase'],phase_step=state['phase_step'],global_step=state['global_step'],metrics=state['metrics'],Writer_served=False,labels_in_serving=False,second_order=False)
 return result

def render(states,selection):
 lines=['# SEEM 96->%d: saved five-round training results'%selection['settings']['horizon'],'',
  'All metrics below were already evaluated and saved during training. The round-end checkpoint is selected by Validation MSE; no new training or inference. Full native replay was deliberately skipped, not marked as passed.','',
  '| State | Step | Train MSE / MAE | Validation MSE / MAE | Test MSE / MAE | A MSE / MAE | B MSE / MAE |','|---|---:|---|---|---|---|---|']
 for name,state in states.items():lines.append('|%s|%d|%s|'%(name,state['global_step'],'|'.join(pair(state['metrics'][s]) for s in ('train','validation','test','A','B'))))
 lines+=['','Settings: '+str(selection['settings']),
  'Round-end selection: R%d Writer%d global%d'%(selection['round'],selection['phase_step'],selection['global_step'])]
 return '\n'.join(lines)+'\n'

def persist_results(evo,selected,states,history,selection,base):
 evo=Path(evo);check_selection(states,history,selection)
 if selected['metrics']!=selection['metrics'] or selected['global_step']!=selection['global_step']:raise ValueError('Selected weight/metric position mismatch')
 if selected['binding']!=base['binding']:raise ValueError('Selected state and frozen framework binding differ')
 print('WRITE saved round-end selected state | global%d | no forward/evaluation'%selected['global_step'],flush=True)
 save(evo/'memory_selected.pt',artifact(selected,base))
 write(evo/'selection_report.json',selection)
 (evo/'report.md').write_text(render(states,selection))
 roles=('train','validation','test','A','B');columns=[sp+'_'+k for sp in roles for k in ('mse','mae')]
 with (evo/'metrics.csv').open('w',newline='') as handle:
  writer=csv.writer(handle);writer.writerow(['state','global_step']+columns)
  for name,row in states.items():writer.writerow([name,row['global_step']]+[row['metrics'][sp][k] for sp in roles for k in ('mse','mae')])
 with (evo/'learning_curve.csv').open('w',newline='') as handle:
  writer=csv.writer(handle);writer.writerow(['global_step','round','phase','phase_step']+columns)
  for row in history:writer.writerow([row[k] for k in ('global_step','round','phase','phase_step')]+[row['metrics'][sp][k] for sp in roles for k in ('mse','mae')])
 write(evo/'finalization.json',dict(mode='saved_training_results_only',training_repeated=False,model_forward_calls=0,optimizer_updates=0,native_replay_performed=False,native_replay_verified=False,metrics_source='original saved per-update training evaluations',round_end_global_step=selected['global_step'],input_bank_Teacher_observables_reloaded=False))

def compact_complete(folder,names=None):

 folder=Path(folder);summary=read(folder/'summary.json');summary['post_training_checks']=read(folder/'evolution/finalization.json');write(folder/'summary.json',summary)
 names=['protocol.json','summary.json','report.md','evolution/selection_report.json','evolution/finalization.json']
 write(folder/'complete.json',dict(files={n:digest(folder/n) for n in names},completion_scope='training_and_selected_weight_export',native_replay_verified=False,large_data_cache_rehash_performed=False))
