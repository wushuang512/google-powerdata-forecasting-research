from pathlib import Path
import sys,json,hashlib,datetime,argparse,numpy as np
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',required=True);args=ap.parse_args();T=Path(args.root);B=T/'experiment';R=B/'round29';C=R/'code';Q=R/'results';sys.path.insert(0,str(C));sys.path.insert(0,str(B/'code'));import common;from aligned_metrics import evaluate
d=common.load();proto=json.loads((R/'docs/protocol.json').read_text(encoding='utf-8'));stage=args.stage
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(Q/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def load(n):return json.loads((Q/n).read_text(encoding='utf-8'))
def arr(path,key):
 with np.load(B/path) as z:return z[key].astype('float64')
def prior(phase,H):
 if phase=='dev':
  if H==1:return arr('round26/results/dev_predictions.npz','selected1')[:,[0,2,4,6]]
  return arr('round26/results/dev_predictions.npz','selected7')
 return arr('round26/results/test_predictions.npz',f'selected{H}')
def ttm(phase):
 sys.path.insert(0,str(R/'cpu_runtime'))
 import torch
 from ttm_official.modeling_tinytimemixer import TinyTimeMixerForPrediction
 torch.set_num_threads(2);orig=sorted(set(proto['dev24']+proto['dev7'])) if phase=='dev' else ([31] if phase=='deploy' else proto['test24']);cut=14 if phase=='dev' else 22
 flat=d['y'].reshape(57,-1);mean=flat[:,:cut*96].mean(-1);std=flat[:,:cut*96].std(-1).clip(1e-4);bank={};meta={}
 for context in [512,1024]:
  name=f'TTM{context}-720-r2';model=TinyTimeMixerForPrediction.from_pretrained(str(R/f'models/{context}-720-r2'),local_files_only=True).eval();meta[name]=dict(parameters=sum(p.numel() for p in model.parameters()),device='CPU',context=context,forecast_native=720,cropped=672,standardization_cutoff=cut)
  for o in orig:
   x=(flat[:,o*96-context:o*96]-mean[:,None])/std[:,None];assert x.shape==(57,context);out=[]
   with torch.inference_mode():
    for k in range(0,57,4):
     pred=model(past_values=torch.from_numpy(x[k:k+4].astype('float32'))[:,:,None]).prediction_outputs.detach().numpy()[:,:,0];out.append(pred[:,:672])
   p=np.concatenate(out)*std[:,None]+mean[:,None];assert np.isfinite(p).all();bank[f'{name}_o{o}']=np.clip(p,0,1);print(phase,name,o,flush=True)
  del model
 np.savez_compressed(Q/f'ttm_{phase}.npz',**bank);save(f'ttm_{phase}_metadata.json',meta)
# The multivariate basis is fitted on train-only daily variations. Predict future daily Fourier coefficients directly.
grid=np.arange(96);basis=np.stack([np.ones(96)]+[f(2*np.pi*j*grid/96) for j in range(1,7) for f in [np.cos,np.sin]],axis=1);pinv=np.linalg.pinv(basis)
def fit_factor(cut,rank):
 levels=d['y'][:,:cut].mean(-1).T;mu=levels.mean(0);sd=levels.std(0).clip(.002);z=(levels-mu)/sd;u,s,v=np.linalg.svd(z,full_matrices=False);return mu,sd,v[:rank]
def features(o,h,factor):
 y=d['y'];mu,sd,v=factor;hist=y[:,o-7:o];recent=hist[:,-3:].mean(1);norm=recent.mean(-1).clip(.05);coeff=recent@pinv.T;last=hist[:,-1]@pinv.T;week=hist.mean(1)@pinv.T;lev=hist.mean(-1);z=(lev.T-mu)/sd;fac=(z@v.T);shared=np.concatenate([fac[-1],fac[-3:].mean(0),fac[-1]-fac[-3]],axis=0)
 own=np.column_stack([lev[:,-1]/norm-1,lev[:,-2]/norm-1,lev[:,-3]/norm-1,lev.mean(1)/norm-1,(lev[:,-1]-lev[:,-3])/norm,lev.std(1)/norm,hist[:,-1].std(-1)/norm,norm])
 cal=np.array([h/7,(h/7)**2,np.sin(2*np.pi*((o+h-1)%7)/7),np.cos(2*np.pi*((o+h-1)%7)/7)])
 X=np.concatenate([own,(last-coeff)/norm[:,None],(week-coeff)/norm[:,None],np.tile(shared,(57,1)),np.tile(cal,(57,1))],axis=1)
 # Keep the past mean3 high-frequency remainder to avoid invented high-frequency detail.
 return X,norm,coeff,recent-coeff@basis.T
def fit(cut,rank,alpha):
 factor=fit_factor(cut,rank);xx=[];yy=[]
 for o in range(7,cut):
  for h in range(1,min(7,cut-o)+1):
   X,norm,cf,hi=features(o,h,factor);j=o+h-1;ok=d['valid'][:,j].all(-1);target=(d['y'][:,j]@pinv.T-cf)/norm[:,None];xx.append(X[ok]);yy.append(target[ok])
 x=np.concatenate(xx);y=np.concatenate(yy);xm=x.mean(0);xs=x.std(0).clip(.01);z=(x-xm)/xs;z=np.column_stack([np.ones(len(z)),z]);reg=np.eye(z.shape[1])*alpha;reg[0,0]=.01;coef=np.linalg.solve(z.T@z+reg,z.T@y)
 return dict(factor=factor,xm=xm,xs=xs,coef=coef,n_samples=len(x),cutoff=cut,rank=rank,alpha=alpha)
def predict(model,orig):
 out=[]
 for o in orig:
  days=[]
  for h in range(1,8):
   X,norm,cf,hi=features(o,h,model['factor']);z=np.column_stack([np.ones(57),(X-model['xm'])/model['xs']]);future=cf+(z@model['coef'])*norm[:,None];days.append(np.clip(future@basis.T+hi,0,1))
  out.append(np.concatenate(days,axis=-1))
 return np.stack(out,axis=1)
def candidates(phase):
 cut=14 if phase=='dev' else 22;orig=sorted(set(proto['dev24']+proto['dev7'])) if phase=='dev' else proto['test24'];bank={};mods={}
 with np.load(Q/f'ttm_{phase}.npz') as z:
  for c in [512,1024]:bank[f'TTM{c}-720-r2']=np.stack([z[f'TTM{c}-720-r2_o{o}'] for o in orig],axis=1)
 for rank in [2,4]:
  for alpha in [10,100]:
   name=f'JointFactorFourier rank{rank} ridge{alpha}';model=fit(cut,rank,alpha);bank[name]=predict(model,orig);mods[name]=dict(training_samples=model['n_samples'],cutoff=cut,rank=rank,alpha=alpha)
   np.savez_compressed(R/f'models/factor_{phase}_r{rank}_a{alpha}.npz',mean=model['factor'][0],scale=model['factor'][1],components=model['factor'][2],xm=model['xm'],xs=model['xs'],coef=model['coef'])
 return bank,orig,mods
def scored(phase,bank,orig):
 scores={};out={}
 for name,p in bank.items():
  scores[name]={}
  for H in [1,7]:
   oo=proto[f'{phase if phase=="dev" else "test"}{24 if H==1 else 7}'];a=p[:,[orig.index(o) for o in oo],:H*96];out[f'{name}_H{H}']=a;scores[name][str(H)]=evaluate(common,d,oo,a,H)
 return scores,out
if stage.startswith('ttm_'):ttm(stage.split('_')[1])
elif stage=='develop':
 bank,orig,mods=candidates('dev');scores,out=scored('dev',bank,orig);controls={str(H):evaluate(common,d,proto[f'dev{24 if H==1 else 7}'],prior('dev',H),H) for H in [1,7]};items=[]
 for n,s in scores.items():
  ratios=[s['1']['wape']/controls['1']['wape'],s['1']['peak_mape']/controls['1']['peak_mape'],s['7']['wape']/controls['7']['wape'],s['7']['daily_wape']/controls['7']['daily_wape']];items.append(dict(model=n,joint_cost=float(np.mean(ratios)),ratios=ratios,eligible=bool(np.mean(ratios)<=.98 and max(ratios)<=1.02)))
 ordered=sorted(items,key=lambda r:(r['joint_cost'],r['model']));selected=next((x['model'] for x in ordered if x['eligible']),None);v=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(R/'docs/protocol.json'),candidate=ordered[0]['model'],selected_new_model=selected,retain_controls=selected is None,scores=scores,controls=controls,ranking=ordered,models=mods,dev_prediction_sha256=None,no_test_used=True)
 np.savez_compressed(Q/'dev_predictions.npz',**out);v['dev_prediction_sha256']=sha(Q/'dev_predictions.npz');assert not (Q/'selection.json').exists();save('selection.json',v);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');print(json.dumps(dict(ranking=ordered,selected=selected),ensure_ascii=False),flush=True)
elif stage=='test':
 sel=load('selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();assert sel['protocol_sha256']==sha(R/'docs/protocol.json');bank,orig,mods=candidates('test');scores,out=scored('test',bank,orig);controls={str(H):evaluate(common,d,proto[f'test{24 if H==1 else 7}'],prior('test',H),H) for H in [1,7]};np.savez_compressed(Q/'test_predictions.npz',**out);save('summary.json',dict(scores=scores,controls=controls,selection=sel,selection_sha256=sha(Q/'selection.json'),protocol_sha256=sha(R/'docs/protocol.json'),test_predictions_sha256=sha(Q/'test_predictions.npz'),new_recommended=sel['selected_new_model'],independent_confirmation=False));print(json.dumps(dict(scores={n:{H:{k:s[k] for k in ['wape','daily_wape','peak_mape']} for H,s in v.items()} for n,v in scores.items()},retained=sel['retain_controls']),ensure_ascii=False),flush=True)
elif stage=='deploy':
 # All candidates remain research predictions; the selected recommendation is determined by DEV.
 import pandas as pd
 bank={};meta={}
 for rank in [2,4]:
  for alpha in [10,100]:bank[f'JointFactorFourier rank{rank} ridge{alpha}']=predict(fit(22,rank,alpha),[31])[:,0]
 # TTM has its own guarded inference stage for origin31 and final train scaler.
 np.savez_compressed(Q/'factor_deployment.npz',**bank);save('deployment_status.json',dict(factor_curves=len(bank),origin=31,horizon=672,research_only=load('selection.json')['retain_controls'],not_current_live_forecast=True));print('factor deployment saved',flush=True)
else:raise ValueError(stage)
