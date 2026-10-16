from pathlib import Path
import argparse,hashlib,pickle,joblib,sys,warnings
import numpy as np
from core import *
from models import *
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('phase',choices=['validation','test']);a=ap.parse_args();R=Path(a.root);F,cfg,d,q=load(R);reg=read(F/'docs/historical_controls.json')
assert sha(F/'results/selection.json')==reg['original_target_selection_sha256']
for rel,dd in reg['bindings'].items():assert sha(R/rel)==dd,rel
sys.path[:0]=[str(R/'experiment/generalization_validation_20261012/code'),str(R/'experiment/round44/reference/PowerData_Prediction')]
from local_models import fields,prediction,smooth
source=joblib.load(R/'experiment/round44/results/local_cut22.joblib');frozen=dict(kind='local24',models=source.models,coefficients=source.coefficients,lags=source.lags,feature_names=source.feature_names)
with Path(reg['ridge_file']).open('rb') as f:ridge=pickle.load(f)
feature_names=read(R/'experiment/round5/docs/calibration_features.json')['features'];c44=read(R/'experiment/round44/results/selection.json')['configs']
cut=q['split']['train'][1] if a.phase=='validation' else q['split']['validation'][1];known=d['y'][:cut];scale=float(np.quantile(known[np.isfinite(known)],.95)/.8)
def base(name,H):return read_forecasts(F,q,a.phase,H,name)/scale
def weights(H):
    c=c44['24h' if H==1 else'7d'];key=str(c['segments'][0]);aa=np.array(c['component_a'][key]);bb=np.array(c['component_b'][key]);W=(1-c['mix'])*np.column_stack([aa,np.zeros(len(aa))])+c['mix']*bb
    assert W.shape[1]==len(c['methods']) and np.allclose(W.sum(1),1);return c['methods'],W
def blend(P,W):return np.einsum('olk,lk->ol',P,W[np.arange(P.shape[1])//48])
# Thirty source Ridge features, with actual source missing-proxy semantics and
# unknown source cell encoded all-zero. Observed forecast residuals are past only.
def calibration(o,daybase,banks):
    y=d['y']/scale;items={};slot=np.arange(96)
    def add(name,v):items[name]=np.full(96,float(v)) if np.ndim(v)==0 else np.asarray(v)
    add('base',daybase);add('predicted_shape',daybase-daybase.mean());add('last_minus_base',y[o-1]-daybase);add('week_minus_base',y[o-7]-daybase);add('mean3_minus_base',y[o-3:o].mean(0)-daybase)
    for k in [1,3,7]:add(f'level{k}_minus_base',y[o-k:o].mean()-daybase.mean());add(f'power_std{k}',y[o-k:o].std())
    for name in ['past_proxy_mean','proxy_availability7','proxy_slot_available','last_proxy']:add(name,0.)
    available=[k for k in banks if max(7,o-7)<=k<o and np.isfinite(y[k]).all()]
    if available:
        available.sort();errs=np.stack([y[k]-banks[k] for k in available]);add('observed_bias7',errs.mean());add('observed_bias1',errs[-1].mean());add('observed_error_shape7',errs.mean(0))
    else:
        for name in ['observed_bias7','observed_bias1','observed_error_shape7']:add(name,0.)
    add('hour_sin',np.sin(2*np.pi*slot/96));add('hour_cos',np.cos(2*np.pi*slot/96))
    for name in feature_names:
        if name.startswith('cell_'):add(name,0.)
    assert set(items)==set(feature_names);X=np.column_stack([items[n] for n in feature_names]);assert np.isfinite(X).all();return ridge.predict(X)
b1=.25*base('source_chronos2_original',1)+.25*base('source_tirex2_original',1)+.5*base('timesfm3',1);priorbank={o:b1[i] for i,o in enumerate(origins(q,a.phase,1))}
for H in [1,7]:
    oo=origins(q,a.phase,H);ch=base('source_chronos2_original',H);ti=base('source_tirex2_original',H);tf=base('timesfm3',H);to=base('toto2',H);patch=base('source_patchtst_original',H)
    three=.25*ch+.25*ti+.5*tf;cal=[]
    for i,o in enumerate(oo):
        curve=[]
        for day in range(H):
            sl=slice(day*96,(day+1)*96);curve.append(three[i,sl]+reg['source_r5_residual_alpha']*calibration(o,three[i,sl],priorbank))
        cal.append(np.concatenate(curve))
    cal=np.maximum(0,np.stack(cal));r8=np.maximum(0,cal+reg['source_r8_component_alpha']*.5*(base('source_timesfm3_r8b',H)-tf))
    mm,W=weights(H);bank={}
    for n in mm:
        if n=='chronos2_long':bank[n]=ch
        elif n=='tirex2':bank[n]=ti
        elif n=='timesfm25':bank[n]=base('source_timesfm25_original',H)
        elif n in ['toto2_joint','toto2_single']:bank[n]=to
        elif n in ['timesfm30','timesfm30_long','timesfm30_uni']:bank[n]=tf
        elif n=='patchtstfm_long':bank[n]=patch
        elif n=='local24':
            with warnings.catch_warnings():warnings.simplefilter('ignore');bank[n]=np.stack([prediction(frozen,d['y'][o-14:o]/scale,d['dates'][o]) for o in oo])
        elif n=='local7':bank[n]=np.stack([smooth(d['y'][o-14:o]/scale,7) for o in oo])
        else:raise ValueError(n)
    P=np.stack([bank[n] for n in mm],-1);hdsf=np.maximum(0,blend(P,W))
    if H==1:delta=(base('source_patchtst_r47curve',1)-patch).mean(1,keepdims=True)
    else:
        delta=np.zeros_like(hdsf)
        for name,f in [('timesfm30_long','timesfm3'),('timesfm30_uni','timesfm3'),('toto2_joint','toto2'),('patchtstfm_long','patchtst')]:
            delta+=(base('source_'+f+'_r46',H)-bank[name])*W[np.arange(672)//48,mm.index(name)][None]
        shaped=delta.reshape(len(oo),7,96);delta=(shaped-shaped.mean(-1,keepdims=True)).reshape(delta.shape)
    out=dict(Google_Chronos_TiRex_equal=.5*ch+.5*ti,Google_three_fixed=three,Google_R5_Ridge=cal,Google_R8b_finetuned=r8,Google_HDSF_original=hdsf,Google_R48_robust=np.maximum(0,hdsf+delta))
    for name,p in out.items():save_forecasts(F,cfg,d,q,a.phase,H,name,np.maximum(0,p)*scale,dict(historical_source_control=True,source_weights_frozen=True,input_history_adapted_to14days=True,unknown_Google_fields_not_replaced_by_NLR_IT=True,source_bound_sha256=sha(F/'docs/historical_controls.json'),seven_day_extension=H==7 and name in ['Google_Chronos_TiRex_equal','Google_three_fixed','Google_R5_Ridge','Google_R8b_finetuned'],original_target_selection_sha256=reg['original_target_selection_sha256']))
save(F/'results'/f'historical_{a.phase}_assembled.json',dict(all_passed=True,utc=utc(),input_adaptation=True,source_config_sha256=sha(F/'docs/historical_controls.json'),original_target_selection_unchanged=sha(F/'results/selection.json')==reg['original_target_selection_sha256']));print('Historical controls actual forecasts assembled',a.phase,flush=True)
