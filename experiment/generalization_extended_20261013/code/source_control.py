"""Exact fixed original HDSF/R48 source controls on the new NLR origins."""
from pathlib import Path
import argparse, os, sys, hashlib
sys.dont_write_bytecode=True;os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('case');ap.add_argument('family',choices=['timesfm3','toto2','patchtst','local','assemble']);ap.add_argument('--phase',default='dev');a=ap.parse_args()
ROOT=Path(a.root);F=ROOT/'experiment/generalization_extended_20261013';OLD=ROOT/'experiment/generalization_validation_20261012'
assert a.case.startswith('NLR')
sys.path[:0]=[str(OLD/'code'),str(ROOT/'experiment/round46/code'),str(ROOT/'experiment/round44/reference/PowerData_Prediction')]
from core import read,save,sha,utc,np_save,original_weights,simplex_tensor
proto=read(F/'docs/protocol.json');cfg=proto['cases'][a.case];cut=cfg['cutoff'];oo=cfg['origins'][a.phase]
z=np.load(F/'data'/f'{a.case}.npz');sc=z['scales'];x=z['x']/sc[:,None,None];dates=z['dates']
def base(f,H):
    return np.stack([np.load(F/'cache'/f'{a.case}_{f}_cut{cut}_s0_{a.phase}_H{H}_o{o}.npz')['p'] for o in oo],1)
path=F/'cache'/f'{a.case}_source_{a.family}_{a.phase}.npz'
if path.exists():
    with np.load(path) as q:assert str(q['protocol_sha256'])==sha(F/'docs/protocol.json')
    raise SystemExit(0)
if a.family=='local':
    import joblib
    from local_models import setup,prediction,smooth
    setup(ROOT);model=joblib.load(ROOT/'experiment/round44/results/local_cut22.joblib')
    frozen=dict(kind='local24',models=model.models,coefficients=model.coefficients,lags=model.lags,feature_names=model.feature_names)
    arrays=dict(local24=np.stack([np.stack([prediction(frozen,x[j,o-28:o],dates[o]) for o in oo]) for j in range(2)]),
        local7=np.stack([np.stack([smooth(x[j,o-24:o],7) for o in oo]) for j in range(2)]))
elif a.family=='assemble':
    extra={f:np.load(F/'cache'/f'{a.case}_source_{f}_{a.phase}.npz') for f in ['timesfm3','toto2','patchtst','local']}
    arrays={}
    for H in [1,7]:
        members,W=original_weights(ROOT,H);bank={}
        for name in members:
            if name=='chronos2_long':bank[name]=base('chronos2',H)
            elif name=='tirex2':bank[name]=base('tirex2',H)
            elif name=='timesfm25':bank[name]=base('timesfm25',H)
            elif name in ['toto2_joint','toto2_single']:bank[name]=base('toto2',H)
            elif name=='timesfm30':bank[name]=extra['timesfm3']['short_H1']
            elif name in ['timesfm30_long','timesfm30_uni']:bank[name]=base('timesfm3',H)
            elif name=='patchtstfm_long':bank[name]=base('patchtst',H)
            elif name=='local24':bank[name]=extra['local']['local24']
            elif name=='local7':bank[name]=extra['local']['local7']
            else:raise ValueError(name)
        tensor=np.stack([bank[k] for k in members],-1);plain=simplex_tensor(tensor,W)
        if H==1:
            before=base('patchtst',1);after=extra['patchtst']['updated_H1'];delta=(after-before).mean(-1,keepdims=True).repeat(96,-1)
        else:
            delta=np.zeros_like(plain)
            for name,f in [('timesfm30_long','timesfm3'),('timesfm30_uni','timesfm3'),('toto2_joint','toto2'),('patchtstfm_long','patchtst')]:
                before=bank[name];after=extra[f]['updated_H7'];w=W[np.arange(672)//48,members.index(name)];delta+=(after-before)*w[None,None]
            shaped=delta.reshape(2,len(oo),7,96);delta=(shaped-shaped.mean(-1,keepdims=True)).reshape(plain.shape)
        arrays[f'HDSF_original_source_H{H}']=np.maximum(0,plain)*sc[:,None,None]
        arrays[f'R48_fixed_source_H{H}']=np.maximum(0,plain+delta)*sc[:,None,None]
else:
    import torch
    from hdsf_foundation_adapters import Adapter
    sys.path.append(str(ROOT/'experiment/round44/.venv/Lib/site-packages'))
    adapter=Adapter(ROOT,a.family,'cuda');adapter.set_scope('tail');initial=adapter.trainable_state();arrays={}
    if a.family=='timesfm3':
        arrays['short_H1']=np.stack([adapter.predict(x[:,o-14:o].reshape(2,-1),96,'uni') for o in oo],1)
    cfg46=read(ROOT/'experiment/round46/results/selection.json')['families']
    states=[(7,ROOT/'experiment/round46/models'/f'{a.family}_{cfg46[a.family]["config"]}_cut22.pt')]
    if a.family=='patchtst':states.insert(0,(1,ROOT/'experiment/round47/models/patchtst_curve_cut22.pt'))
    source_hashes={}
    for H,p in states:
        source_hashes[str(p)]=sha(p);adapter.load_trainable_state(initial);adapter.load_trainable_state(torch.load(p,map_location='cpu',weights_only=True));days=14 if a.family=='toto2' else 21
        rows=[]
        for o in oo:
            point=adapter.predict(x[:,o-days:o].reshape(2,-1),H*96,'uni');assert point.shape==(2,H*96) and np.isfinite(point).all();rows.append(point)
        arrays[f'updated_H{H}']=np.stack(rows,1)
    save(F/'results'/f'{a.case}_source_{a.family}_{a.phase}_bindings.json',dict(source_weights_sha256=source_hashes,source_adapter_sha256=sha(ROOT/'experiment/round46/code/hdsf_foundation_adapters.py'),target_weights_updated=False))
np_save(path,**arrays,protocol_sha256=sha(F/'docs/protocol.json'),origins=np.array(oo),script_sha256=sha(Path(__file__)))
print('Fixed source control',a.case,a.family,a.phase,list(arrays),flush=True)
