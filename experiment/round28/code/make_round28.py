from pathlib import Path
import sys,ast
T=Path(sys.argv[1]);S=Path(__file__).parent;R=T/'experiment/round28';(R/'code').mkdir(parents=True,exist_ok=True)
v=(S/'robust_point_experiment.py').read_text(encoding='utf-8').replace("R=B/'round27'","R=B/'round28'").replace("round27/models/interval","round28/models/interval")
start=v.index('MODELS=');end=v.index('ALPHAS=',start)
v=v[:start]+"MODELS=[dict(id='lgb_regression_l1_leaf15',loss='regression_l1',leaves=15,iterations=80)]\nCONFIGS=[dict(id=f'level_shape{int(s*100)}_cap{int(c*10000)}',model='lgb_regression_l1_leaf15',shape_gain=s,cap=c) for s in [0.,.25] for c in [.0025,.005]]\n"+v[end:]
start=v.index('def candidate(');end=v.index('def prefix_check',start)
v=v[:start]+'''def candidate(d,base,ref,delta,cfg,origins):
 out=base.copy();raw=base.copy();H=base.shape[-1]//96
 for j,o in enumerate(origins):
  _,scale=features(d,o,0,ref[:,j,:96])
  correction=delta[:,j].reshape(57,H,96);mean=correction.mean(-1,keepdims=True);change=mean+cfg['shape_gain']*(correction-mean);change=np.clip(change,-cfg['cap']*scale[:,None,None],cfg['cap']*scale[:,None,None]);raw[:,j]=np.clip(base[:,j]+change.reshape(57,-1),0,1)
 for j in range(base.shape[1]):
  for cell in np.unique(d['cells']):
   ix=d['cells']==cell
   if cell=='a':out[ix,j]=base[ix,j];continue
   target=base[ix,j].sum();v=raw[ix,j];lo=-2.;hi=2.
   for _ in range(55):
    mid=(lo+hi)/2
    if np.clip(v+mid,0,1).sum()>target:hi=mid
    else:lo=mid
   out[ix,j]=np.clip(v+(lo+hi)/2,0,1)
   assert abs(out[ix,j].sum()-target)<1e-8
 return out
'''+v[end:]
v=v.replace("models=fit(d,14)","models=pickle.load((B/'round27/models/dev_heads.pkl').open('rb'))")
v=v.replace("models=fit(d,22,{v['candidate']['model'] for v in s['horizons'].values()})","models=pickle.load((B/'round27/models/final_heads.pkl').open('rb'))")
v=v.replace("candidate(d,base,ref,deltas[cfg['model']],cfg)","candidate(d,base,ref,deltas[cfg['model']],cfg,origins)")
v=v.replace("candidate(d,base,ref,residual(d,origins,ref,models[cfg['candidate']['model']],H),cfg['candidate'])","candidate(d,base,ref,residual(d,origins,ref,models[cfg['candidate']['model']],H),cfg['candidate'],origins)")
v=v.replace("candidate(d,base,ref,residual(d,[31],ref,models[cfg['candidate']['model']],H),cfg['candidate'])","candidate(d,base,ref,residual(d,[31],ref,models[cfg['candidate']['model']],H),cfg['candidate'],[31])")
v=v.replace("'round3/results/tf3_test.npz']","'round3/results/tf3_test.npz','round27/results/selection.json','round27/models/dev_heads.pkl','round27/models/final_heads.pkl']")
v=v.replace('Four LightGBM Huber/L1 heads, no neural reload.','Reuse the LightGBM L1 15leaf model fixed by R27 DEV for both horizons; no new fitting or neural reload.')
v=v.replace('prior plus robust normalized point residual, optionally subtract half of already-existing prior minus native calibration to avoid double correction.','Robust LightGBM residual decomposed to daily level plus0or25%zero-mean shape. Cap each raw residual to0.25or0.5%of historical7day mean. Fixed normalization uses only pre-origin history.')
v=v.replace('rank8 configs','rank4 configs')
v=v.replace('Never TEST rerank; if none alpha0 keepsR26.','Never TEST rerank; if none alpha0 keepsR26. Recommendation replacement additionally requires selected TEST within registered minor-tradeoff limits; if this exploratory diagnostic fails, withhold replacement without changing the frozen experimental alpha or searching TEST alternatives.')
v=v.replace("final_heads_sha256=sha(R/'models/final_heads.pkl')","final_heads_sha256=sha(R/'models/final_heads.pkl'),recommendation_supported={h:bool(s['horizons'][h]['development_supported'] and guard(scores[h]['selected'],scores[h]['prior'])) for h in ['1','7']}")
ast.parse(v)
if (R/'docs/protocol.json').exists():
 assert (R/'code/bounded_level_experiment.py').read_text(encoding='utf-8')==v
 assert (R/'code/source_metrics.py').read_text(encoding='utf-8')==(S/'source_metrics.py').read_text(encoding='utf-8')
else:
 (R/'code/bounded_level_experiment.py').write_text(v,encoding='utf-8');(R/'code/source_metrics.py').write_text((S/'source_metrics.py').read_text(encoding='utf-8'),encoding='utf-8')
g=(S/'run_cache_bounded_guarded.py').read_text(encoding='utf-8').replace("R=T/'experiment/round21'","R=T/'experiment/round28'").replace('code/localized_peak_experiment.py','code/bounded_level_experiment.py')
(R/'code/run_bounded_level_guarded.py').write_text(g,encoding='utf-8');print('Bounded level correction trial ready')
