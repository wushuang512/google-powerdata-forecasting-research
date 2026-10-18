"""Only the R5/R8 source artifacts required by the six NLR transfer controls."""
import sys
from .support import REPO,read,save,source_path,sha

def paths(work):
 for rel in ['experiment/code','experiment/round2/code','experiment/round3/code','experiment/round5/code','google-clusterdata-2019/code']:
  p=work/rel
  if rel.startswith('google-'):
   p.mkdir(parents=True,exist_ok=True)
   import shutil
   for n in ['finetune_best_foundation.py']:
    shutil.copyfile(REPO/rel/n,p/n)
  sys.path.insert(0,str(p))
 for sub in ['round3/results','round5/models','round5/results','round8b/models','round8b/results']:(work/'experiment'/sub).mkdir(parents=True,exist_ok=True)

def bank(work,family):
 paths(work)
 if family=='tirex2':
  # This import intentionally requests the original CPU implementation before torch.
  import tirex_run as tr
  from tirex2 import load_model
  model=load_model(str(work/'experiment/models/tirex2-base'),device='cpu',use_flex_attention=False,compile=False)
  from multiscale import forecast,CONFIGS
  fn=lambda d,oo,split:forecast(model,d,oo,CONFIGS['native'],True) if split=='train' else tr.predict_tirex(model,d,oo,'uni',96,1344)
  stem='tirex';key='native'
 elif family=='chronos2':
  from deep_models import base_chronos
  from round2_lib import predict
  from multiscale import forecast,CONFIGS
  model=base_chronos()
  fn=lambda d,oo,split:forecast(model,d,oo,CONFIGS['native'],False) if split=='train' else predict(model,d,oo,'uni',96,1344)
  stem='chronos';key='native'
 elif family=='timesfm3':
  from timesfm3 import TimesFM3Forecaster
  from timesfm3_run import forecast,CONFIGS
  model=TimesFM3Forecaster.from_pretrained(str(work/'experiment/models/timesfm3-base'),device='cuda',per_core_batch_size=4,local_files_only=True)
  fn=lambda d,oo,split:forecast(model,d,oo,CONFIGS['tf3_cellprod']);stem='tf3';key='tf3_cellprod'
 else:raise ValueError(family)
 import numpy as np
 with np.load(work/'experiment/results/data.npz') as z:d={k:z[k] for k in z.files}
 for split,oo in [('train',list(range(7,14))),('dev',list(range(14,22)))]:
  p=work/f'experiment/round3/results/{stem}_{split}.npz'
  if p.exists():continue
  np.savez_compressed(p,**{key:fn(d,oo,split)})
  save(p.with_suffix('.reproduction.json'),{'source':'actual native inference','family':family,'origins':oo,'npz_sha256':sha(p)})

def fit_source_calibrator(work):
 import numpy as np,pickle
 paths(work)
 from calibration import fit,bank
 with np.load(work/'experiment/results/data.npz') as z:d={k:z[k] for k in z.files}
 model,features=fit(d,bank(['train','dev']),list(range(7,22)),100000)
 p=work/'experiment/round5/models/residual_ridge100000_final.pkl'
 with p.open('wb') as f:pickle.dump(model,f)
 save(p.with_suffix('.reproduction.json'),{'fit_from_google_only':True,'training_origins':list(range(7,22)),'features':features,'model_sha256':sha(p)})

def train_r8(work):
 import numpy as np
 paths(work)
 import finetune_best_foundation as old
 with np.load(work/'experiment/results/data.npz') as z:d={k:z[k] for k in z.files}
 cfg=old.CONFIGS['best_tail_lr1e6_teacher01'];f,h=old.train(d,22,cfg,steps=100)
 path=work/'experiment/round8b/models/best_tail_lr1e6_teacher01_s100_final.pt'
 old.ft.save_delta(f,path,cfg,22,100)
 save(path.with_suffix('.reproduction.json'),{'actual_gradient_steps':100,'cutoff':22,'configuration':cfg,'seed':20261001,'schedule_denominator':600,'history':h,'model_sha256':sha(path),'redistribution_prohibited_by_base_license':True})
