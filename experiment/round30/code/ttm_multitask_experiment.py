from pathlib import Path
import sys,json,hashlib,datetime,argparse,numpy as np
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',required=True);args=ap.parse_args();T=Path(args.root);B=T/'experiment';R=B/'round30';Q=R/'results';C=R/'code';P=B/'round29';sys.path.insert(0,str(P/'cpu_runtime'));sys.path.insert(0,str(P/'code'));sys.path.insert(0,str(B/'code'));import common;from aligned_metrics import evaluate
import torch
from ttm_official.modeling_tinytimemixer import TinyTimeMixerForPrediction
torch.set_num_threads(2);torch.manual_seed(20261007);torch.use_deterministic_algorithms(True);d=common.load();flat=d['y'].reshape(57,-1);valid=d['valid'].reshape(57,-1);proto=json.loads((R/'docs/protocol.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(Q/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def load(n):return json.loads((Q/n).read_text(encoding='utf-8'))
def z(path,key):
 with np.load(B/path) as a:return a[key].astype('float64')
def prior(phase,H):return z(f'round26/results/{"dev" if phase=="dev" else "test"}_predictions.npz',f'selected{H}')[:,[0,2,4,6]] if phase=='dev' and H==1 else z(f'round26/results/{"dev" if phase=="dev" else "test"}_predictions.npz',f'selected{H}')
def fit(mode,cut):
 torch.manual_seed(20261007);model=TinyTimeMixerForPrediction.from_pretrained(str(P/'models/512-720-r2'),local_files_only=True)
 for n,p in model.named_parameters():p.requires_grad_(mode=='all' or n.startswith('head.') or n.startswith('decoder.'))
 params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=1e-5 if mode=='all' else 1e-4,weight_decay=.01)
 mean=flat[:,:cut*96].mean(-1);std=flat[:,:cut*96].std(-1).clip(1e-4);starts=list(range(512,cut*96-95,24));count=np.zeros(cut*96)
 for s in starts:count[s:min(s+672,cut*96)]+=1
 rows=[(i,s) for s in starts for i in range(57)];rng=np.random.default_rng(20261007);trace=[]
 for epoch in range(4):
  model.train();losses=[]
  for first in range(0,len(rows),16):
   if first==0:order=rng.permutation(len(rows))
   batch=[rows[j] for j in order[first:first+16]];xs=[];ys=[];ms=[];ws=[];ii=[]
   for i,s in batch:
    xs.append((flat[i,s-512:s]-mean[i])/std[i]);end=min(s+672,cut*96);length=end-s;yy=np.zeros(672);mm=np.zeros(672,dtype=bool);ww=np.zeros(672);yy[:length]=flat[i,s:end];mm[:length]=valid[i,s:end];ww[:length]=1/count[s:end];ys.append(yy);ms.append(mm);ws.append(ww);ii.append(i)
   x=torch.tensor(np.stack(xs),dtype=torch.float32)[:,:,None];y=torch.tensor(np.stack(ys),dtype=torch.float32);m=torch.tensor(np.stack(ms));w=torch.tensor(np.stack(ws),dtype=torch.float32)*m;mi=torch.tensor(mean[ii],dtype=torch.float32)[:,None];sd=torch.tensor(std[ii],dtype=torch.float32)[:,None];norm=torch.tensor(mean[ii].clip(.05),dtype=torch.float32)[:,None]
   pred=model(past_values=x).prediction_outputs[:,:672,0]*sd+mi;rel=(pred-y)/norm
   curve=(torch.nn.functional.smooth_l1_loss(rel,torch.zeros_like(rel),reduction='none',beta=.01)*w).sum()/w.sum().clamp_min(1)
   yy=y.reshape(-1,7,96);pp=pred.reshape(-1,7,96);mm=m.reshape(-1,7,96);ok=mm.all(-1);dd=(pp.mean(-1)-yy.mean(-1))/norm;pk=(pp.max(-1).values-yy.max(-1).values)/norm
   daily=(abs(dd)*ok).sum()/ok.sum().clamp_min(1);peak=(abs(pk)*ok).sum()/ok.sum().clamp_min(1);loss=curve+.25*daily+.25*peak
   optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1);optimizer.step();losses.append(float(loss.detach()))
  trace.append(dict(epoch=epoch+1,loss=float(np.mean(losses))));print('train',mode,cut,epoch+1,trace[-1]['loss'],flush=True)
 model.eval();mp=R/f'models/{mode}_cut{cut}';model.save_pretrained(mp,safe_serialization=True);np.savez_compressed(mp/'training_scaler.npz',mean=mean,std=std);save(f'train_{mode}_cut{cut}.json',dict(trace=trace,training_rows=len(rows),last_label_index_exclusive=cut*96,earliest_input=0,trainable_parameters=sum(p.numel() for p in params),epochs=4,seed=20261007,weights_sha256=sha(mp/'model.safetensors')));return model,mean,std
def infer(model,mean,std,orig):
 out=[]
 with torch.inference_mode():
  for o in orig:
   x=(flat[:,o*96-512:o*96]-mean[:,None])/std[:,None];ps=[]
   for i in range(0,57,4):ps.append(model(past_values=torch.tensor(x[i:i+4],dtype=torch.float32)[:,:,None]).prediction_outputs[:,:672,0].numpy())
   out.append(np.clip(np.concatenate(ps)*std[:,None]+mean[:,None],0,1))
 return np.stack(out,axis=1)
def run(phase):
 orig=sorted(set(proto['dev24']+proto['dev7'])) if phase=='dev' else proto['test24'];scores={};out={};cut=14 if phase=='dev' else 22
 for mode in ['head','all']:
  model,mean,std=fit(mode,cut);p=infer(model,mean,std,orig);name=f'TTM512 multitask {mode}';scores[name]={}
  for H in [1,7]:
   oo=proto[f'{"dev" if phase=="dev" else "test"}{24 if H==1 else 7}'];pp=p[:,[orig.index(o) for o in oo],:H*96];scores[name][str(H)]=evaluate(common,d,oo,pp,H);out[f'{name}_H{H}']=pp
  del model
 np.savez_compressed(Q/f'{phase}_predictions.npz',**out);return scores
stage=args.stage
if stage=='develop':
 scores=run('dev');controls={str(H):evaluate(common,d,proto[f'dev{24 if H==1 else 7}'],prior('dev',H),H) for H in [1,7]};items=[]
 for n,s in scores.items():
  ratios=[s['1']['wape']/controls['1']['wape'],s['1']['peak_mape']/controls['1']['peak_mape'],s['7']['wape']/controls['7']['wape'],s['7']['daily_wape']/controls['7']['daily_wape']];items.append(dict(model=n,joint_cost=float(np.mean(ratios)),ratios=ratios,eligible=bool(np.mean(ratios)<=.98 and max(ratios)<=1.02)))
 items.sort(key=lambda r:(r['joint_cost'],r['model']));selected=next((x['model'] for x in items if x['eligible']),None);assert not (Q/'selection.json').exists();save('selection.json',dict(protocol_sha256=sha(R/'docs/protocol.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),scores=scores,controls=controls,ranking=items,selected_new_model=selected,retain_controls=selected is None,no_test_used=True));(Q/'selection.sha256').write_text(sha(Q/'selection.json'));print(json.dumps(dict(ranking=items,selected=selected),ensure_ascii=False),flush=True)
elif stage=='test':
 sel=load('selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text() and sel['protocol_sha256']==sha(R/'docs/protocol.json');scores=run('test');save('summary.json',dict(scores=scores,controls={str(H):evaluate(common,d,proto[f'test{24 if H==1 else 7}'],prior('test',H),H) for H in [1,7]},selection=sel,selection_sha256=sha(Q/'selection.json'),test_predictions_sha256=sha(Q/'test_predictions.npz'),independent_confirmation=False));print(json.dumps(dict(scores=scores,retained=sel['retain_controls']),ensure_ascii=False),flush=True)
elif stage=='deploy':
 bank={}
 for mode in ['head','all']:
  mp=R/f'models/{mode}_cut22';model=TinyTimeMixerForPrediction.from_pretrained(mp,local_files_only=True).eval()
  with np.load(mp/'training_scaler.npz') as z:mean=z['mean'];std=z['std']
  bank[f'TTM512 multitask {mode}']=infer(model,mean,std,[31])[:,0];del model
 np.savez_compressed(Q/'deployment.npz',**bank);print('deployment saved',flush=True)
else:raise ValueError(stage)
