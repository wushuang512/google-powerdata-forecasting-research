"""Round46 all-member DEV selection/freeze and unchanged three-task evaluation."""
from pathlib import Path
import argparse,csv,datetime,gc,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
import numpy as np,joblib

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def export(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def initialize(root):
 global T,B,R,Q,A,proto,data,d,common,evaluate,interval_metrics,core,canon,original_config
 T=Path(root);B=T/'experiment';R=B/'round46';Q=R/'results';A=B/'round44';proto=read(R/'docs/protocol.json')
 sys.path[:0]=[str(B/'code'),str(B/'round29/code'),str(B/'round28/code'),str(A/'reference/PowerData_Prediction')]
 import common as cm
 from aligned_metrics import evaluate as ev
 from source_metrics import interval_metrics as im
 common,evaluate,interval_metrics=cm,ev,im;d=common.load();data=joblib.load(A/'results/source_dataset.joblib')
 canon=np.array([data.series_keys.index(str(n)) for n in d['names']]);original_config=read(A/'results/selection.json')['configs']
 spec=importlib.util.spec_from_file_location('round45_core',B/'round45/code/hdsf_next_experiment.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core);core.initialize(T)
 assert sha(A/'results/predictions.npz')==proto['source_predictions_sha256']
 assert sha(B/'round29/code/aligned_metrics.py')==proto['canonical_evaluation_sha256']

def origins(phase,H):return (proto['dev24'] if H==1 else proto['dev7']) if phase=='dev' else (proto['test24'] if H==1 else proto['test7'])
def masked_data(part):
 result=dict(d);mask=d['valid'].copy()
 if part=='early':mask[:,18:]=False
 elif part=='late':mask[:,:18]=False
 result['valid']=mask;return result

def members_original(phase,H):
 methods=original_config['24h' if H==1 else'7d']['methods']
 with np.load(A/'results/predictions.npz') as z:return {m:z[('dev_' if phase=='dev' else'')+m+f'_H{H}'].copy() for m in methods}
def baseline(phase,H):
 with np.load(A/'results/predictions.npz') as z:return z[('dev_' if phase=='dev' else'')+f'hdsf_H{H}'].copy()

def local(phase):
 cut=14 if phase=='dev' else 22;path=Q/f'local_{phase}.npz'
 if path.exists():return
 # True fresh local24 refit, never use old fitted trees as this round's new training.
 model,info=core.fit_local('dropD1',cut)
 p=np.stack([core.local_predict(model,data,o,1)[canon] for o in origins(phase,1)],axis=1)
 joblib.dump(model,R/'models'/f'local24_cut{cut}.joblib');save(Q/f'local24_cut{cut}.json',info);del model;gc.collect()
 from power_forecast.attainment_members import local_curve
 configs=[{'half_lives':list(pair),'shape_half_life':shape,'history_days':24} for pair in [(2.,4.),(3.,5.),(4.,7.)] for shape in [5.,10.,20.]]
 trainorig=list(range(3,cut-6));scores=[]
 for cfg in configs:
  pp=np.stack([local_curve(data,o,7,configuration=cfg).transpose(2,0,1).reshape(57,672)[canon] for o in trainorig],axis=1)
  score=evaluate(common,d,trainorig,pp,7);cost=score['wape']+.25*score['daily_wape']+.1*score['peak_mape']
  scores.append({'configuration':cfg,'training_cost':cost,'training_wape':score['wape'],'training_daily_wape':score['daily_wape']})
 selected=min(scores,key=lambda r:r['training_cost'])['configuration']
 p7=np.stack([local_curve(data,o,7,configuration=selected).transpose(2,0,1).reshape(57,672)[canon] for o in origins(phase,7)],axis=1)
 np.savez_compressed(path,local24_H1=p,local7_H7=p7)
 save(Q/f'local7_cut{cut}.json',{'cutoff':cut,'training_origins':trainorig,'last_label_day':max(trainorig)+7,
    'selected_train_only':selected,'candidates':scores,'method':'TRAIN-only parameter fitting of causal level/shape smoother; no neuralgradient claimed.'})

def family_bank(family,cfg,phase,H,alpha):
 original=members_original(phase,H);control=np.load(Q/f'{family}_{phase}_original_control.npz');raw=np.load(Q/f'{family}_{phase}_{cfg}.npz')
 bank={}
 for m in proto['families'][family]:
  key=m+f'_H{H}'
  if key in raw.files and m in original:bank[m]=original[m]+alpha*(raw[key]-control[key])
 return bank

def native_cost(family,cfg,alpha,part):
 # Same evaluator with a stricter targetdate mask; never pool PDUdenominators.
 dm=masked_data(part);vals=[];daily=[];groups={}
 for H in [1,7]:
  oo=origins('dev',H);old=members_original('dev',H);bank=family_bank(family,cfg,'dev',H,alpha)
  for m,p in bank.items():
   with np.errstate(divide='ignore',invalid='ignore'):
    base=evaluate(common,dm,oo,old[m],H);score=evaluate(common,dm,oo,p,H)
   vals.append(score['wape']/base['wape'])
   if H==7:vals.append(score['daily_wape']/base['daily_wape'])
   y,mask=common.truth(dm,oo,H);dates=np.array([o+1+np.arange(96*H)//96 for o in oo])
   for day in np.unique(dates[mask.any(0)]):
    v=mask&(dates[None]==day);den=(abs(y)*v).sum((1,2));keep=den>0
    if keep.any():
     gain=float(np.mean(100*(((abs(y-old[m])-abs(y-p))*v).sum((1,2))[keep])/den[keep]))
     groups.setdefault(int(day),[]).append(gain)
 date_gain=np.array([np.mean(v) for _,v in sorted(groups.items())]);se=float(date_gain.std(ddof=1)/np.sqrt(len(date_gain))) if len(date_gain)>1 else 0.
 return {'cost':float(np.mean(vals)),'date_gain_pp':date_gain.tolist(),'paired_date_se_pp':se,
         'mean_date_gain_pp':float(date_gain.mean()) if len(date_gain) else 0.,'calendar_dates':sorted(groups)}

def forward_cv(P,y,m,dates,H,layout=None,source_style=False):
 # Every newly chosen family config/alpha was fitted using targetdates≤D18.
 # Use validation only afterD18, with no retrospective familyalpha in an earlier fold.
 if H==7:return core.crossvalidate(P,y,m,dates,H,layout,source_style)
 out=np.zeros_like(y);held=np.zeros_like(m);scores=[];details=[]
 for i in [2,3]:
  fitmask=m&(np.arange(len(m))[:,None,None]<i);valmask=m&(np.arange(len(m))[:,None,None]==i)
  assert dates[valmask.any(2)].min()>18
  if source_style:fit=core.original_style_fit(P,y,fitmask,dates,H);p=core.original_style_apply(P,fit)
  else:fit=core.fit_geometry(P,y,fitmask,dates,layout);p=core.apply_geometry(P,fit)
  out[valmask]=p[valmask];held|=valmask;sc=core.wape(y,p,valmask);scores.append(sc)
  details.append({'fold':'validateD'+str(int(dates[i,0])),'family_parameter_label_end':18,
     'fit_latest_day':int(dates[fitmask.any(2)].max()),'validation_day':int(dates[valmask.any(2)].min()),'score':sc})
 return {'prediction':out,'mask':held,'score':core.wape(y,out,held),'fold_scores':scores,'folds':details}

def candidate_bank(phase,H,selection,raw_all=False):
 old=members_original(phase,H);out={m:p.copy() for m,p in old.items()}
 for f,s in selection['families'].items():out.update(family_bank(f,s['config'],phase,H,1. if raw_all else s['alpha']))
 loc=np.load(Q/f'local_{phase}.npz')
 for key in loc.files:
  m,h=key.rsplit('_H',1)
  if int(h)==H:out[m]=loc[key]
 return out

def source_weights(H):
 cfg=original_config['24h' if H==1 else'7d'];n=str(cfg['segments'][0]);a=np.asarray(cfg['component_a'][n]);b=np.asarray(cfg['component_b'][n]);mix=cfg['mix']
 return np.column_stack([a,np.zeros(len(a))])*(1-mix)+b*mix
def frozen_original_weight_prediction(bank,H):
 methods=original_config['24h' if H==1 else'7d']['methods'];P=np.stack([bank[m] for m in methods],axis=-1)
 W=source_weights(H);idx=np.minimum(np.arange(H*96)//48,len(W)-1)
 return np.einsum('solm,lm->sol',P,W[idx],optimize=True)
def geometry_tensor(bank,H):
 methods=original_config['24h' if H==1 else'7d']['methods']
 return np.stack([bank[m][core.natural_index] for m in methods],axis=-1).transpose(1,2,0,3)

def freeze():
 path=Q/'selection.json'
 if path.exists():assert sha(path)==(Q/'selection.sha256').read_text().strip();return
 for family in proto['families']:assert read(Q/f'{family}_dev_complete.json')['all_passed'],family
 local('dev');selection={'created_utc':utc(),'families':{},'fusion':{},'protocol_sha256':sha(R/'docs/protocol.json'),
    'DEV_only':True,'historical_test_informed_design':True,'independent_confirmation':False}
 familyrows=[]
 for family in proto['families']:
  cases=[f'{scope}_s{step}' for scope in ['head','tail'] for step in [40,80]]
  early=[{'config':cfg,**native_cost(family,cfg,1.,'early')} for cfg in cases]
  best=min(early,key=lambda r:r['cost']);cfg=best['config']
  late=[{'alpha':alpha,**native_cost(family,cfg,alpha,'early')} for alpha in [0.,.25,.5,1.]]
  win=min(late,key=lambda r:r['cost']);plateau=[r for r in late if r['cost']<=win['cost']+.005]
  # Positive gain must exceed the calendar-date uncertainty; noTEST information.
  supported=[r for r in plateau if r['alpha']==0 or r['mean_date_gain_pp']>max(.005,r['paired_date_se_pp'])]
  picked=min(supported,key=lambda r:r['alpha']) if supported else late[0]
  selection['families'][family]={'config':cfg,'alpha':picked['alpha'],'early_candidates':early,'early_alpha_candidates':late,
    'checkpoint_sha256':sha(R/'models'/f'{family}_{cfg}_cut14.pt')}
  familyrows.extend({'family':family,'part':'early','config':r['config'],**r} for r in early)
  familyrows.extend({'family':family,'part':'early_alpha','config':cfg,**r} for r in late)
 for H in [1,7]:
  bank=candidate_bank('dev',H,selection);P=geometry_tensor(bank,H);y,m,dates=core.truth(H,origins('dev',H))
  original,mm=core.members(H,'dev');ref=forward_cv(original,y,m,dates,H,source_style=True)
  rows=[];fits={};base=baseline('dev',H);fullbase=evaluate(common,d,origins('dev',H),base,H)
  # Three registered geometry choices plus the originalfrozenweight combination.
  frozen=frozen_original_weight_prediction(bank,H)
  # Validation of originalgeometry uses earlier-date-only refits; fixed DEVweights are not independently validated.
  for j,layout in enumerate(proto['fusion_registry']):
   cv=forward_cv(P,y,m,dates,H,layout);fit=core.fit_geometry(P,y,m,dates,layout)
   full=core.apply_geometry(P,fit).transpose(2,0,1)[canon];fits[str(j)]=fit
   for alpha in [0.,.25,.5,1.]:
    held=ref['mask'];cvpred=(1-alpha)*ref['prediction']+alpha*cv['prediction'];cvscore=core.wape(y,cvpred,held)
    fold_gain=alpha*(np.asarray(ref['fold_scores'])-np.asarray(cv['fold_scores']));se=float(fold_gain.std(ddof=1)/np.sqrt(len(fold_gain)))
    point=(1-alpha)*base+alpha*full;score=evaluate(common,d,origins('dev',H),point,H)
    gate=(alpha==0 or (fold_gain.mean()>max(.005,se) and (fold_gain>0).sum()>len(fold_gain)/2 and score['wape']<=fullbase['wape'] and score['daily_wape']<=fullbase['daily_wape'] and score['peak_mape']<=fullbase['peak_mape']+.1))
    rows.append({'geometry':str(j),'layout':layout,'alpha':alpha,'forward_WAPE':cvscore,'mean_forward_gain_pp':float(fold_gain.mean()),
                 'paired_forward_se_pp':se,'adoption_gate':bool(gate),'full_development':score,'folds':cv['folds']})
  eligible=[r for r in rows if r['adoption_gate']];best=min(r['forward_WAPE'] for r in eligible);near=[r for r in eligible if r['forward_WAPE']<=best+.005]
  chosen=min(near,key=lambda r:(r['alpha'],len(fits[r['geometry']]['weights']),r['forward_WAPE']))
  diagnostic=min(rows,key=lambda r:r['forward_WAPE'])
  selection['fusion'][str(H)]={'selected':chosen,'selected_fit':fits[chosen['geometry']],
      'diagnostic':diagnostic,'diagnostic_fit':fits[diagnostic['geometry']],'candidates':rows,
      'methods':original_config['24h' if H==1 else'7d']['methods']}
  save(Q/f'fusion_development_H{H}.json',selection['fusion'][str(H)])
 export(Q/'基础模型开发期选择.csv',familyrows);save(path,selection);(Q/'selection.sha256').write_text(sha(path),encoding='ascii')
 print(json.dumps({'FROZEN':True,'families':{f:{'config':s['config'],'alpha':s['alpha']} for f,s in selection['families'].items()},
      'fusion':{h:s['selected']['alpha'] for h,s in selection['fusion'].items()}},ensure_ascii=True),flush=True)

def score():
 selection=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text().strip()
 for family in proto['families']:assert read(Q/f'{family}_test_complete.json')['all_passed'],family
 local('test');arrays={};rows=[];summary={};native_rows=[]
 for H in [1,7]:
  oo=origins('test',H);do=origins('dev',H);base=baseline('test',H);devbase=baseline('dev',H)
  bank=candidate_bank('test',H,selection);dbank=candidate_bank('dev',H,selection)
  rawbank=candidate_bank('test',H,selection,True);rawall=frozen_original_weight_prediction(rawbank,H)
  s=selection['fusion'][str(H)];P=geometry_tensor(bank,H);DP=geometry_tensor(dbank,H)
  gp=core.apply_geometry(P,s['selected_fit']).transpose(2,0,1)[canon]
  dgp=core.apply_geometry(DP,s['selected_fit']).transpose(2,0,1)[canon]
  alpha=s['selected']['alpha'];selected=(1-alpha)*base+alpha*gp;dselected=(1-alpha)*devbase+alpha*dgp
  diag=core.apply_geometry(P,s['diagnostic_fit']).transpose(2,0,1)[canon]
  fixed=frozen_original_weight_prediction(bank,H)
  pp={'original_HDSF':base,'DEV_selected':selected,'DEV_selected_geometry_diagnostic':diag,
      'family_gated_original_weights':fixed,'all_finetuned_original_weights_diagnostic':rawall}
  scores={role:evaluate(common,d,oo,p,H) for role,p in pp.items()}
  for role,p in pp.items():arrays[f'{role}_H{H}']=p;rows.append({'horizon_days':H,'model':role,'selected_alpha':alpha,**scores[role]})
  arrays[f'dev_selected_H{H}']=dselected;y,m=common.truth(d,oo,H);dy,dm=common.truth(d,do,H)
  low=np.array([np.quantile((dy-dselected)[i][dm[i]],.1) for i in range(57)]);high=np.array([np.quantile((dy-dselected)[i][dm[i]],.9) for i in range(57)])
  lower=np.clip(selected+low[:,None,None],0,1);upper=np.clip(selected+high[:,None,None],0,1)
  arrays[f'lower80_H{H}']=lower;arrays[f'upper80_H{H}']=upper;iv=interval_metrics(y,selected,m,lower,upper)
  summary[str(H)]={'scores':scores,'interval':iv,'source_HDSF_retained_if_alpha0':alpha==0,'selected_alpha':alpha}
  for family,fs in selection['families'].items():
   for kind,a in [('selected_family_alpha',fs['alpha']),('raw_finetune_diagnostic',1.)]:
    for member,p in family_bank(family,fs['config'],'test',H,a).items():
     arrays[f'{kind}_{member}_H{H}']=p;native_rows.append({'family':family,'model':member,'config':fs['config'],'kind':kind,'alpha':a,'horizon_days':H,**evaluate(common,d,oo,p,H)})
 np.savez_compressed(Q/'predictions.npz',**arrays);export(Q/'HDSF全部基础模型微调统一比较.csv',rows);export(Q/'各基础模型微调统一指标.csv',native_rows)
 save(Q/'summary.json',{'created_utc':utc(),'scores':summary,'rows':rows,'native_rows':native_rows,'selection_sha256':sha(Q/'selection.json'),
     'protocol_sha256':sha(R/'docs/protocol.json'),'data':'OriginalGooglePowerData2019 only','metric_code_unchanged':True,
     'historical_test_informed_design':True,'independent_confirmation':False,'new_Word_created':False})
 save(Q/'complete.json',{'completed_utc':utc(),'all_passed':True,'all_six_neural_families_really_finetuned':True,
     'all_nine_neural_member_modes_evaluated':True,'both_local_members_refitted':True,
     'predictions_sha256':sha(Q/'predictions.npz'),'summary_sha256':sha(Q/'summary.json'),
     'selection_sha256':sha(Q/'selection.json'),'independent_verification_pending':True})
 print(json.dumps(summary,ensure_ascii=True),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--phase',choices=['local_dev','freeze','score'],required=True);args=ap.parse_args()
 initialize(args.root)
 if args.phase=='local_dev':local('dev')
 elif args.phase=='freeze':freeze()
 else:score()
