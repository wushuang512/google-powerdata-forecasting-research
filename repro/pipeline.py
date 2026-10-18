from pathlib import Path
import hashlib,json,subprocess,sys,os
from .support import read,save,sha

def google(work,cpu,gpu,source_root=None,include_legacy=False):
 launcher=Path(__file__).resolve().parents[1]/'reproduce.py'
 lock=read(launcher.parent/'repro/assets.lock.json')
 for row in lock['google_data']:
  if not (work/row['path']).is_file() or sha(work/row['path'])!=row['sha256']:raise RuntimeError('Raw input missing or changed: '+row['path'])
 for model in lock['models']:
  for row in model['files']:
   p=work/model['location']/row['name']
   if not p.is_file() or sha(p)!=row['sha256']:raise RuntimeError('Pinned model asset missing or changed: '+model['model_id']+'/'+row['name'])
 for runtime,version,cuda in [(cpu,'2.8.0+cpu',False),(gpu,'2.11.0+cu128',True)]:
  code='import torch; assert torch.__version__ == '+repr(version)+'; '+('assert torch.cuda.is_available()' if cuda else 'print(torch.__version__)')
  probe=subprocess.run([str(runtime),'-B','-c',code],capture_output=True,text=True)
  if probe.returncode:raise RuntimeError('Reproduction runtime does not match the fixed profile: '+str(runtime)+'\n'+probe.stderr[-1500:])
 tasks=[(cpu,['google-prepare']),(cpu,['google-baselines'])]
 members=['chronos2_long','tirex2','timesfm25','toto2_joint','toto2_single','timesfm30','timesfm30_long','timesfm30_uni','local24','patchtstfm_long','local7']
 for phase in ['dev','test']:
  tasks += [(cpu,['google-member',m,'--phase',phase]) for m in members]
  tasks += [(gpu,['google-adapt',f,'--recipe','r46','--phase',phase]) for f in ['timesfm3','toto2','patchtst']]
  tasks += [(gpu,['google-adapt','patchtst','--recipe','r47curve','--phase',phase])]
 tasks += [(cpu,['google-score'])]
 if include_legacy:
  tasks += [(cpu if f=='tirex2' else gpu,['legacy-bank',f]) for f in ['chronos2','tirex2','timesfm3']]
  tasks += [(cpu,['legacy-calibrator']),(gpu,['legacy-r8-train'])]
 ledger=work/'reproduction/pipeline_ledger.json';done=read(ledger) if ledger.exists() else{}
 for i,(runtime,args) in enumerate(tasks,1):
  fingerprint=hashlib.sha256((sha(work/'reproduction_workspace.json')+str(runtime)+json.dumps(args)).encode()).hexdigest()
  previous=done.get(fingerprint)
  if previous and previous['exit_code']==0:
   outputs=previous.get('output_sha256',{})
   if all((work/p).is_file() and sha(work/p)==h for p,h in outputs.items()):continue
   raise RuntimeError('Previously completed job output changed; use a fresh work directory')
  cmd=[str(runtime),'-B',str(launcher),'--work',str(work)]
  if source_root:cmd += ['--source-root',str(source_root)]
  cmd+=args;log=work/'reproduction'/f'job_{i:02}_{args[0]}.log';print('REPRODUCTION_JOB',i,len(tasks),args,flush=True)
  with log.open('w',encoding='utf-8') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  outputs={}
  if args[0]=='google-member':
   names=['reproduction/'+args[-1]+'/'+args[1]+ext for ext in ['.npz','.json']]
   if args[1]=='local24':names+=['experiment/round44/results/local_cut'+('14' if args[-1]=='dev' else '22')+'.joblib']
  elif args[0]=='google-adapt':
   names=['reproduction/'+args[-1]+'/'+args[1]+'_'+args[3]+'_delta'+ext for ext in ['.npz','.json']]
   cut='14' if args[-1]=='dev' else'22'
   names += ['experiment/'+('round47/models/patchtst_curve_cut'+cut+'.pt' if args[3]=='r47curve' else 'round46/models/'+args[1]+'_tail_s40_cut'+cut+'.pt')]
  elif args[0]=='google-prepare':
   names=['experiment/results/data.npz','experiment/round44/results/source_dataset.joblib','reproduction/results/google_data_audit.json']
  elif args[0]=='google-baselines':names=['reproduction/results/google_baselines.csv']
  elif args[0]=='google-score':names=['reproduction/results/google_R48_all_metrics.csv']+['reproduction/results/google_R48_H'+str(h)+suffix for h in [1,7] for suffix in ['.npz','_comparison.json']]
  elif args[0]=='legacy-bank':
   stem={'chronos2':'chronos','tirex2':'tirex','timesfm3':'tf3'}[args[1]];names=['experiment/round3/results/'+stem+'_'+split+ext for split in ['train','dev'] for ext in ['.npz','.reproduction.json']]
  elif args[0]=='legacy-calibrator':names=['experiment/round5/models/residual_ridge100000_final.pkl','experiment/round5/models/residual_ridge100000_final.reproduction.json']
  elif args[0]=='legacy-r8-train':names=['experiment/round8b/models/best_tail_lr1e6_teacher01_s100_final'+ext for ext in ['.pt','.reproduction.json']]
  else:raise RuntimeError('No output contract for pipeline job')
  for name in names:
   p=work/name;outputs[name]=sha(p) if p.is_file() else'missing'
  if result.returncode==0 and 'missing' in outputs.values():raise RuntimeError('Successful process did not create its required outputs; see '+str(log))
  done[fingerprint]={'command':args,'runtime':str(runtime),'exit_code':result.returncode,'log':str(log.relative_to(work)),'output_sha256':outputs};save(ledger,done)
  if result.returncode:raise RuntimeError('Reproduction job failed; see '+str(log))
 save(work/'reproduction/results/pipeline_complete.json',{'completed':True,'frozen_recipe_retrained':True,'legacy_nlr_source_models_included':include_legacy,'independent_blind_test':False})
