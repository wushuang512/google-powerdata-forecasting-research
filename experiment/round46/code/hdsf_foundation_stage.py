"""Actual foundation weight training, isolated one-family jobs for Round46."""
from pathlib import Path
import argparse, datetime, gc, hashlib, json, os, sys, time, traceback
sys.dont_write_bytecode=True
import joblib,numpy as np,torch
from hdsf_foundation_adapters import Adapter,FAMILIES,load_training_data,load_loss

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def np_save(p,arrays):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
 tmp.replace(p)

def initialize(root):
 global T,B,R,Q,A,data,d,canon,natural,protocol
 T=Path(root);B=T/'experiment';R=B/'round46';Q=R/'results';A=B/'round44'
 sys.path[:0]=[str(A/'reference/PowerData_Prediction'),str(B/'code'),str(B/'round29/code')]
 global common,evaluate
 import common as cm
 from aligned_metrics import evaluate as ev
 common,evaluate=cm,ev;data=load_training_data(T);d=common.load()
 canon=np.array([data.series_keys.index(str(n)) for n in d['names']]);natural=np.array([list(map(str,d['names'])).index(n) for n in data.series_keys])
 protocol=read(R/'docs/protocol.json')
 assert protocol['script_sha256']==sha(__file__) and protocol['adapter_sha256']==sha(Path(__file__).with_name('hdsf_foundation_adapters.py'))
 assert protocol['canonical_evaluation_sha256']==sha(B/'round29/code/aligned_metrics.py')
 assert np.array_equal(data.target_valid[canon],d['valid'])
 assert sha(A/'results/predictions.npz')==protocol['source_predictions_sha256']
 assert sha(A/'results/selection.json')==protocol['source_selection_sha256']
 for sub in ['models','results']:(R/sub).mkdir(exist_ok=True)

def origins(phase,H):return ([14,16,18,20] if H==1 else [14,15]) if phase=='dev' else (list(range(22,31)) if H==1 else[22,23,24])
def member_tasks(member):return [H for H,t in [(1,'24h'),(7,'7d')] if member in read(A/'results/selection.json')['configs'][t]['methods']]
def context(member,origin):
 maxdays=28 if member=='chronos2_long' else 21 if member in ['timesfm30_long','timesfm30_uni','patchtstfm_long'] else 14
 days=min(origin,maxdays);return np.ascontiguousarray(data.target[:,origin-days:origin].reshape(57,-1),dtype=np.float32)

def predict_member(adapter,member,origin,H):
 x=context(member,origin);h=H*96
 if member.startswith('timesfm30') and member!='timesfm30_uni':return adapter.predict(x,h,'joint')
 if member=='toto2_joint':
  from power_forecast.multiscale_experiment import toto_groups
  groups=toto_groups(data);out=np.empty((57,h))
  for group in dict.fromkeys(groups):
   ix=np.array([i for i,g in enumerate(groups) if g==group]);out[ix]=adapter.predict(x[ix],h,'joint')
  return out
 out=[];batch=4 if adapter.family!='patchtst' else 2
 for j in range(0,57,batch):out.append(adapter.predict(x[j:j+batch],h,'uni'))
 result=np.concatenate(out);assert result.shape==(57,h) and np.isfinite(result).all()
 return result

def collect(adapter,family,phase,label):
 path=Q/f'{family}_{phase}_{label}.npz';meta=path.with_suffix('.json')
 if path.exists() and meta.exists():
  info=read(meta);assert info['npz_sha256']==sha(path) and info['adapter_sha256']==protocol['adapter_sha256']
  return {k:v for k,v in np.load(path).items()}
 arrays={};inputs={}
 for member in FAMILIES[family]:
  for H in member_tasks(member):
   parts=[]
   for o in origins(phase,H):
    print('INFER',family,label,member,'H',H,'origin',o,flush=True)
    p=predict_member(adapter,member,o,H);assert np.isfinite(p).all();parts.append(p[canon])
    inputs[f'{member}_H{H}_o{o}']=hashlib.sha256(context(member,o).tobytes()).hexdigest()
   arrays[f'{member}_H{H}']=np.stack(parts,axis=1)
 np_save(path,arrays);save(meta,{'utc':utc(),'label':label,'phase':phase,'family':family,
    'npz_sha256':sha(path),'adapter_sha256':protocol['adapter_sha256'],'input_context_digests':inputs,
    'trainable_weights_sha256':adapter.hash_trainable(),'no_future_true_workload_or_power_input':True})
 return arrays

def sample_batches(cut,family,steps=80):
 rng=np.random.default_rng(20261012);sequence=[]
 horizons=[96] if family in ['chronos2','tirex2','timesfm25'] else([672] if family=='patchtst' else[96,672])
 cells=data.meta['cell'].to_numpy();types=data.meta['is_mvpp'].to_numpy()
 for step in range(steps):
  H=horizons[step%len(horizons)];validorigins=np.arange(3*96,cut*96-H+1,48)
  origin=int(rng.choice(validorigins));ctx=min(int(rng.choice([3,7,14]))*96,origin);ctx=ctx//32*32
  mode='joint' if family in ['timesfm3','toto2'] and step%4<2 else'uni'
  if family=='toto2' and mode=='joint':
   anchor=int(rng.integers(57));pool=np.flatnonzero((cells==cells[anchor])&(types==types[anchor]));ids=rng.choice(pool,min(2,len(pool)),replace=False)
  else:ids=rng.choice(57,2,replace=False)
  sequence.append({'origin_tick':origin,'context_ticks':ctx,'horizon_ticks':H,'series_ids':ids.tolist(),'mode':mode,
                   'last_label_tick':origin+H-1,'cutoff_tick':cut*96})
 return sequence

def train(adapter,family,scope,cut,steps,phase,control,callback=True):
 count=adapter.set_scope(scope);initial=adapter.trainable_state();initial_hash=adapter.hash_trainable()
 initial_on_device={n:v.to(adapter.device) for n,v in initial.items()};params=[p for p in adapter.model.parameters() if p.requires_grad]
 lr=3e-5 if scope=='head' else 3e-6;optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
 adapter.model.eval();flat=data.target[:,:cut].reshape(57,-1);valid=data.target_valid[:,:cut].reshape(57,-1)
 seq=sample_batches(cut,family,80);teacher=[]
 for i,b in enumerate(seq[:steps]):
  ix=b['series_ids'];o=b['origin_tick'];ctx=b['context_ticks'];H=b['horizon_ticks']
  assert o+H<=cut*96
  x=flat[ix,o-ctx:o]
  with torch.no_grad():_,p=adapter.tensor_forecast(x,H,b['mode'],False)
  teacher.append(p.detach().float().cpu())
  if (i+1)%20==0:print('TEACHER',family,scope,cut,i+1,flush=True)
 history=[];checkpoint_scores=[];changed_tail=False
 for i,b in enumerate(seq[:steps],1):
  ix=b['series_ids'];o=b['origin_tick'];ctx=b['context_ticks'];H=b['horizon_ticks'];x=flat[ix,o-ctx:o];y=flat[ix,o:o+H];m=valid[ix,o:o+H]
  q,p=adapter.tensor_forecast(x,H,b['mode'],True);q=q.float();p=p.float()
  assert q.requires_grad and p.requires_grad,(family,scope,'no gradient')
  loss,info=load_loss(adapter,q,p,y,m,teacher[i-1].to(adapter.device),teacher_weight=.1)
  prox=torch.stack([((dict(adapter.model.named_parameters())[n]-ref)**2).mean()/ref.square().mean().clamp_min(1e-4) for n,ref in initial_on_device.items()]).mean()
  loss=loss+.05*prox;assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward()
  gradients={n:float(v.grad.abs().max()) for n,v in adapter.model.named_parameters() if v.requires_grad and v.grad is not None}
  assert gradients and max(gradients.values())>0 and all(np.isfinite(v) for v in gradients.values())
  if scope=='tail':
   headprefix={'chronos2':'output_patch_embedding.','tirex2':'output_patch_embedding.',
      'timesfm25':'output_projection_','timesfm3':'output_head.','toto2':'output_head.','patchtst':'backbone.out_layer.'}[family]
   changed_tail|=any(not n.startswith(headprefix) and g>0 for n,g in gradients.items())
  torch.nn.utils.clip_grad_norm_(params,1);optimizer.step()
  for group in optimizer.param_groups:group['lr']=lr*(.2+.8*(1-i/80))
  history.append({'step':i,**info,'prox':float(prox.detach()),'max_gradient':max(gradients.values()),'UTC':utc()})
  if i%10==0:print('TRAIN',family,scope,'cut',cut,history[-1],flush=True)
  if (phase=='dev' and i in [40,80]) or (phase=='test' and i==steps):
   label=f'{scope}_s{i}';state=adapter.trainable_state();weight_hash=adapter.hash_trainable();assert weight_hash!=initial_hash
   modelpath=R/'models'/f'{family}_{label}_cut{cut}.pt';torch.save(state,modelpath)
   modelinfo={'family':family,'scope':scope,'step':i,'cutoff':cut,'trainable_parameters':count,'trainable_tensor_names':adapter.train_names,
      'initial_trainable_sha256':initial_hash,'updated_trainable_sha256':weight_hash,'model_file_sha256':sha(modelpath),
      'real_optimizer_updates':i,'tail_gradient_observed':changed_tail if scope=='tail' else None,'CUDA':adapter.device.type=='cuda',
      'training_last_label_max_tick':max(b['last_label_tick'] for b in seq[:i]),'training_cutoff_tick':cut*96,
      'training_batches_sha256':hashlib.sha256(json.dumps(seq[:i],sort_keys=True).encode()).hexdigest(),'utc':utc(),
      'training_script_sha256':protocol['script_sha256'],'adapter_sha256':protocol['adapter_sha256']}
   if scope=='tail':assert changed_tail,(family,'tail parameters did not receive gradients')
   save(modelpath.with_suffix('.json'),modelinfo);save(Q/f'{family}_{label}_cut{cut}_training.json',history)
   raw=collect(adapter,family,phase,label);scores=[]
   with np.load(A/'results/predictions.npz') as z:
    for key,pred in raw.items():
     member,H=key.rsplit('_H',1);H=int(H);prior=z[('dev_' if phase=='dev' else '')+key]
     bridged=prior+pred-control[key];score=evaluate(common,d,origins(phase,H),bridged,H)
     scores.append({'member':member,'H':H,'model':family+'_'+label,**score})
   checkpoint_scores.extend(scores);save(Q/f'{family}_{phase}_scores.json',checkpoint_scores)
   del state,raw;gc.collect()
 return initial,history

def run_family(family,phase):
 device='cpu' if family=='tirex2' else'cuda'
 if phase=='test':
  selection=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text().strip()
  selected=selection['families'][family];scope,ss=selected['config'].split('_s');steps=int(ss);cut=22
 else:cut=14
 status=Q/f'{family}_{phase}_status.json';save(status,{'state':'running','pid':os.getpid(),'family':family,'phase':phase,'started_utc':utc()})
 marker=Q/f'{family}_{phase}_complete.json'
 if marker.exists():print('already complete',family,phase,flush=True);return
 torch.manual_seed(20261012);adapter=Adapter(T,family,device);control=collect(adapter,family,phase,'original_control')
 if phase=='dev':
  for scope in ['head','tail']:
   checkpoint=R/'models'/f'{family}_{scope}_s80_cut14.pt'
   if checkpoint.exists() and (Q/f'{family}_dev_{scope}_s80.npz').exists():continue
   initial,history=train(adapter,family,scope,14,80,phase,control)
   adapter.load_trainable_state(initial)
 else:train(adapter,family,scope,22,steps,phase,control)
 save(marker,{'completed_utc':utc(),'family':family,'phase':phase,'all_passed':True,'real_finetuning':True,
               'script_sha256':protocol['script_sha256'],'adapter_sha256':protocol['adapter_sha256']})
 save(status,{'state':'complete','family':family,'phase':phase,'completed_utc':utc()})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('family',choices=list(FAMILIES));ap.add_argument('--phase',choices=['dev','test'],required=True);args=ap.parse_args()
 initialize(args.root)
 try:run_family(args.family,args.phase)
 except BaseException as e:
  save(Q/f'{args.family}_{args.phase}_status.json',{'state':'failed','error':repr(e),'traceback':traceback.format_exc(),'utc':utc()});raise
