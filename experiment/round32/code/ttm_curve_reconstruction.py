from pathlib import Path
import sys,json,hashlib,datetime,argparse,numpy as np
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',required=True);args=ap.parse_args();T=Path(args.root);B=T/'experiment';R=B/'round32';Q=R/'results';sys.path.insert(0,str(B/'round29/code'));sys.path.insert(0,str(B/'code'));import common;from aligned_metrics import evaluate
d=common.load();proto=json.loads((R/'docs/protocol.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):(Q/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def z(p,k):
 with np.load(B/p) as a:return a[k].astype('float64')
def controls(phase,H):
 a=z(f'round26/results/{"dev" if phase=="dev" else "test"}_predictions.npz',f'selected{H}');return a[:,[0,2,4,6]] if phase=='dev' and H==1 else a
def expert(phase,H,mode):return z(f'round30/results/{phase}_predictions.npz',f'TTM512 multitask {mode}_H{H}')
def reconstruct(prior,pred,kind,strength=None):
 shape=prior.shape;dd=prior.reshape(-1,96);tt=pred.reshape(-1,96);mean=dd.mean(-1);peak=dd.max(-1);strength=(.1 if kind in ["direct10","shape10"] else .5) if strength is None else strength;newpeak=np.clip((1-strength)*peak+strength*tt.max(-1),mean,1)
 if kind=='direct10':out=(1-strength)*dd+strength*tt
 elif kind=='shape10':
  raw=(1-strength)*dd+strength*(tt-tt.mean(-1)[:,None]+mean[:,None]);out=project(raw,dd.sum(-1),np.ones(len(raw)),None)
 elif kind=='sparsepeak50':
  out=np.minimum(dd,newpeak[:,None]);index=dd.argmax(-1);out[np.arange(len(dd)),index]=newpeak
 elif kind=='sumpeak50':out=project(dd,dd.sum(-1),newpeak,dd.argmax(-1))
 else:raise ValueError(kind)
 assert np.isfinite(out).all();return np.clip(out,0,1).reshape(shape)
def project(raw,total,cap,anchor):
 low=np.full(len(raw),-1.);high=np.full(len(raw),1.)
 for i in range(55):
  shift=(low+high)/2;out=np.clip(raw+shift[:,None],0,cap[:,None])
  if anchor is not None:out[np.arange(len(raw)),anchor]=cap
  too=out.sum(-1)>total;high=np.where(too,shift,high);low=np.where(too,low,shift)
 shift=(low+high)/2;out=np.clip(raw+shift[:,None],0,cap[:,None])
 if anchor is not None:out[np.arange(len(raw)),anchor]=cap
 assert np.max(abs(out.sum(-1)-total))<1e-9
 return out
def score(H,orig,p):return evaluate(common,d,orig,p,H)
def assess(s,b):
 groups=[['wape'],['daily_wape','daily_mae','daily_rmse'],['peak_mape'],['p95_ape','hit2_percent','hit5_percent'],['rmse']];wins=[]
 for ks in groups:
  wins.append(any((s[k]>b[k]+1e-10 if k.startswith('hit') else s[k]<b[k]-1e-10) for k in ks))
 tolerance={'wape':.005,'daily_wape':.005,'peak_mape':.005,'p95_ape':.01,'hit2_percent':.05,'hit5_percent':.05,'rmse':b['rmse']*.002,'mae':b['mae']*.002,'daily_mae':b['daily_mae']*.005,'daily_rmse':b['daily_rmse']*.005,'cella5_wape':.005,'other52_wape':.005,'absolute_bias_percent':.005,'peak_time_mae_minutes':5}
 allsmall=all((s[k]>=b[k]-v-1e-12 if k.startswith('hit') else s[k]<=b[k]+v+1e-12) for k,v in tolerance.items());categories=sum(wins);cost=float(np.mean([s[k]/b[k] for k in ['wape','daily_wape','peak_mape','p95_ape']]))
 return dict(categories_improved=categories,all_tradeoffs_within_limits=allsmall,eligible=bool(allsmall and categories>=2 and cost<1),joint_cost=cost)
stage=args.stage
if stage=='develop':
 bank={};selection={};scores={}
 for H in [1,7]:
  orig=proto[f'dev{24 if H==1 else 7}'];base=controls('dev',H);bs=score(H,orig,base);rr=[];scores[str(H)]={'prior':bs}
  if H==7:
   base=z('round31/results/dev_predictions.npz','TTM_head_shape10_H7');bs=score(H,orig,base);scores[str(H)]['prior']=bs
  original=controls('dev',H);mode='head';tt=expert('dev',H,mode);kind='sparsepeak50' if H==1 else 'shape10'
  for strength in ([.05,.10,.25,.40] if H==1 else [.05,.15,.25,.35]):
   name=f'TTM_head_{kind}_strength{strength}';p=reconstruct(original,tt,kind,strength);bank[f'{name}_H{H}']=p;sc=score(H,orig,p);scores[str(H)][name]=sc;ev=assess(sc,bs);rr.append(dict(model=name,expert=mode,kind=kind,strength=strength,**ev))
  rr.sort(key=lambda r:(not r['eligible'],r['joint_cost'],r['strength']));best=next((x for x in rr if x['eligible']),None);selected=min([x for x in rr if x['eligible'] and x['joint_cost']<=best['joint_cost']+.0001],key=lambda x:x['strength']) if best else None;selection[str(H)]=dict(selected=selected,retain_prior=selected is None,ranking=rr,diagnostic=rr[0]);bank[f'prior_H{H}']=base
 np.savez_compressed(Q/'dev_predictions.npz',**bank);assert not (Q/'selection.json').exists();save('selection.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(R/'docs/protocol.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),horizons=selection,scores=scores,no_test_used=True));(Q/'selection.sha256').write_text(sha(Q/'selection.json'));print(json.dumps(selection,ensure_ascii=False),flush=True)
elif stage=='test':
 sel=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text() and sha(R/'docs/protocol.json')==sel['protocol_sha256'];bank={};scores={}
 for H in [1,7]:
  orig=proto[f'test{24 if H==1 else 7}'];original=controls('test',H);base=z('round31/results/test_predictions.npz','selected_H7') if H==7 else original;scores[str(H)]={'prior':score(H,orig,base)};sh=sel['horizons'][str(H)];item=sh['selected'] or sh['diagnostic'];p=reconstruct(original,expert('test',H,item['expert']),item['kind'],item['strength']);bank[f'diagnostic_H{H}']=p;bank[f'selected_H{H}']=base if sh['retain_prior'] else p;scores[str(H)]['diagnostic']=score(H,orig,p);scores[str(H)]['selected']=score(H,orig,bank[f'selected_H{H}']);scores[str(H)]['test_tradeoff_audit']=assess(scores[str(H)]['diagnostic'],scores[str(H)]['prior'])
 np.savez_compressed(Q/'test_predictions.npz',**bank);save('summary.json',dict(selection=sel,selection_sha256=sha(Q/'selection.json'),scores=scores,test_predictions_sha256=sha(Q/'test_predictions.npz'),independent_confirmation=False,all_candidate_test_sweep=False));print(json.dumps(dict(scores=scores,selection={H:sel['horizons'][H]['selected'] for H in ['1','7']}),ensure_ascii=False),flush=True)
elif stage=='deploy':
 sel=read(Q/'selection.json');bank={}
 with np.load(B/'round26/results/exports/deployment_origin31.npz') as a:original7=a['selected7'].copy()
 with np.load(B/'round31/results/deployment.npz') as a:base7=a['selected_H7'].copy()
 import pandas as pd
 df=pd.read_csv(B/'round26/results/exports/selected_1day_curve.csv');base1=df['forecast_normalized_power'].to_numpy().reshape(57,1,96)
 with np.load(B/'round30/results/deployment.npz') as a:
  for H,base in [(1,base1),(7,base7)]:
   sh=sel['horizons'][str(H)];it=sh['selected'] or sh['diagnostic'];tt=a[f'TTM512 multitask {it["expert"]}'][:,None,:H*96];cp=reconstruct(original7 if H==7 else base,tt,it['kind'],it['strength']);bank[f'diagnostic_H{H}']=cp;bank[f'selected_H{H}']=base if sh['retain_prior'] else cp
 np.savez_compressed(Q/'deployment.npz',**bank);print('deployment saved')
else:raise ValueError(stage)
