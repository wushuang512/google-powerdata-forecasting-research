"""Use the original strict 14-scheme protocol in an isolated generated work tree."""
from pathlib import Path
import os,subprocess,sys
from .support import REPO,read,save,sha

def execute(work,script,*args,runtime=None):
 folder=work/'experiment/nlr_strict_protocol_20261013'
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2')
 paths=[work/'experiment/round44/reference/PowerData_Prediction',work/'third_party/granite-tsfm',work/'experiment/round35/code/official_uni2ts/src']
 env['PYTHONPATH']=os.pathsep.join(map(str,paths))+os.pathsep+env.get('PYTHONPATH','')
 cp=subprocess.run([str(runtime or sys.executable),str(folder/'code'/script),str(work),*args],env=env)
 if cp.returncode:raise SystemExit(cp.returncode)

def run(work,command,cpu=None):
 F=work/'experiment/nlr_strict_protocol_20261013'
 for sub in ['models','data','cache','results','docs']:(F/sub).mkdir(parents=True,exist_ok=True)
 if command=='nlr-prepare':execute(work,'prepare.py');return
 # No fallback to an 8-model table or old predictions when a source model is missing.
 required=['experiment/round44/results/local_cut22.joblib','experiment/round5/models/residual_ridge100000_final.pkl','experiment/round8b/models/best_tail_lr1e6_teacher01_s100_final.pt','experiment/round47/models/patchtst_curve_cut22.pt']
 required += ['experiment/round46/models/'+f+'_tail_s40_cut22.pt' for f in ['timesfm3','toto2','patchtst']]
 missing=[p for p in required if not (work/p).exists()]
 if missing:raise RuntimeError('Full 14-scheme NLR run requires Google source fitting first. Missing: '+', '.join(missing))
 jobs=[('register_historical.py',[]),('evaluate.py',['audit']),('verify_raw_stream.py',[]),('statistical.py',['validation'])]
 jobs += [('foundation.py',[f,'validation']) for f in ['timesfm3','toto2','moirai2']]
 jobs += [('historical_foundation.py',[f,'validation']) for f in ['chronos2','tirex2','timesfm25','patchtst','timesfm3','toto2']]
 jobs += [('historical_assemble.py',['validation']),('evaluate.py',['freeze']),('statistical.py',['final'])]
 jobs += [('foundation.py',[f,'test']) for f in ['timesfm3','toto2','moirai2']]
 jobs += [('historical_foundation.py',[f,'test']) for f in ['chronos2','tirex2','timesfm25','patchtst','timesfm3','toto2']]
 jobs += [('historical_assemble.py',['test']),('evaluate.py',['score']),('evaluate.py',['audit']),('complete_tables.py',[])]
 for script,args in jobs:
  # The original preparation creates a new protocol and binds actual generated
  # source artifacts. Original historical JSON freezes are never rewritten.
  print('NLR_STRICT',script,*args,flush=True);execute(work,script,*args,runtime=None if script in ['foundation.py','historical_foundation.py'] else cpu)
 save(F/'results/reproduction_pipeline_complete.json',{'all_fourteen_schemes':True,'new_target_preprocessing_training_inference':True,'previous_test_exposure':True,'summary_sha256':sha(F/'results/summary.json')})
