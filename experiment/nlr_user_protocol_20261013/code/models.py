from pathlib import Path
import hashlib,sys
import numpy as np
from core import *
def load(root):
    F=Path(root)/'experiment/nlr_user_protocol_20261013';p=read(F/'docs/protocol.json');assert sha(F/'docs/protocol.json')==(F/'docs/protocol.sha256').read_text().strip()
    with np.load(F/'data/processed.npz') as z:d={k:z[k].copy() for k in z.files}
    q=read(F/'data/origins.json');return F,p,d,q
def origins(q,phase,H):return q['origins'][phase+str(H)]
def target(d,oo,H):return np.stack([d['y'][o:o+H].reshape(-1) for o in oo])
def baseline(d,oo,H,kind):
    parts=[]
    for o in oo:
        hist=d['y'][o-14:o]
        p=np.tile(hist[-1],H) if kind=='last' else hist[-7:-7+H].reshape(-1) if H<7 and kind=='week' else hist[-7:].reshape(-1) if kind=='week' else np.tile(hist[-3:].mean(0),H)
        parts.append(p)
    return np.stack(parts)
def features(d,o,H):
    y=d['y'][o-14:o];c=d['covariates'][o-14:o];assert y.shape==(14,96) and np.isfinite(y).all() and np.isfinite(c).all()
    n=H*96;lead=np.arange(n);slot=lead%96;day=lead//96
    dates=np.datetime64(d['dates'][o])+day.astype('timedelta64[D]');weekday=(dates.astype('datetime64[D]').astype(int)+3)%7
    year=dates.astype('datetime64[Y]');ydays=(dates-year.astype('datetime64[D]')).astype(int)
    cols=[np.ones(n),lead/96,slot/96,np.sin(2*np.pi*slot/96),np.cos(2*np.pi*slot/96),np.sin(2*np.pi*weekday/7),np.cos(2*np.pi*weekday/7),np.sin(2*np.pi*ydays/365.25),np.cos(2*np.pi*ydays/365.25),
      y[-1,slot],y[-3:].mean(0)[slot],y[-7:].mean(0)[slot],y[-7+day,slot]]
    for days in [1,3,7,14]:
        v=y[-days:];cols.extend([np.full(n, v.mean()),np.full(n,v.std()),np.full(n,v.min()),np.full(n,v.max())])
    for j in range(7):
        cols.extend([np.full(n,c[-1,:,j].mean()),np.full(n,c[-3:,:,j].mean()),np.full(n,c[-7:,:,j].mean()),c[-1,slot,j]])
    return np.ascontiguousarray(np.stack(cols,1),dtype=np.float32)
def cachepath(F,phase,H,name):return F/'cache'/f'{phase}_H{H}_{name}.npz'
def save_forecasts(F,p,d,q,phase,H,name,pred,metadata=None):
    oo=origins(q,phase,H);assert pred.shape==(len(oo),H*96) and np.isfinite(pred).all()
    np_save(cachepath(F,phase,H,name),p=np.maximum(0,pred),origins=np.array(oo),protocol_sha256=sha(F/'docs/protocol.json'),data_sha256=sha(F/'data/processed.npz'))
    save(F/'results'/f'{phase}_H{H}_{name}_metadata.json',dict(phase=phase,horizon_days=H,model=name,origins=len(oo),protocol_sha256=sha(F/'docs/protocol.json'),data_sha256=sha(F/'data/processed.npz'),prediction_sha256=sha(cachepath(F,phase,H,name)),**(metadata or{})))
def read_forecasts(F,q,phase,H,name):
    path=cachepath(F,phase,H,name)
    with np.load(path) as z:
        assert list(z['origins'])==origins(q,phase,H) and str(z['protocol_sha256'])==sha(F/'docs/protocol.json') and str(z['data_sha256'])==sha(F/'data/processed.npz')
        return z['p'].copy()
