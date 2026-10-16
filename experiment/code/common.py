from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEV24=[14,16,18,20]
TEST24=list(range(22,31))
DEV7=[14,15]
TEST7=[22,23,24]

def load():
    d=np.load(ROOT/'results/data.npz')
    return {k:d[k] for k in d.files}

def truth(d,origins,h=1):
    return np.stack([d['y'][:,o:o+h].reshape(57,h*96) for o in origins],axis=1),np.stack([d['valid'][:,o:o+h].reshape(57,h*96) for o in origins],axis=1)

def metrics(y,p,m):
    assert y.shape==p.shape==m.shape and np.isfinite(p).all()
    e=np.abs(y-p)
    per=100*(e*m).sum((1,2))/((np.abs(y)*m).sum((1,2)))
    ape=100*e[m]/np.maximum(np.abs(y[m]),1e-8)
    # Daily aggregates compare the same valid time slots for actual and prediction.
    yd=y.reshape(y.shape[0],-1,96); pd_=p.reshape(y.shape[0],-1,96); md=m.reshape(y.shape[0],-1,96)
    count=md.sum(-1); ya=(yd*md).sum(-1)/np.maximum(count,1); pa=(pd_*md).sum(-1)/np.maximum(count,1)
    dv=count>0
    dw=np.mean(100*(np.abs(ya-pa)*dv).sum(1)/(np.abs(ya)*dv).sum(1))
    ym=np.where(md,yd,-np.inf).max(-1); pm=np.where(md,pd_,-np.inf).max(-1)
    peak=100*np.mean((np.abs(ym-pm)/np.maximum(ym,1e-8))[dv])
    # First maximum; stable ties may make this diagnostic uninformative for flat curves.
    yt=np.argmax(np.where(md,yd,-np.inf),axis=-1); pt=np.argmax(np.where(md,pd_,-np.inf),axis=-1)
    return dict(wape=float(per.mean()),mae=float(e[m].mean()),p95_ape=float(np.percentile(ape,95)),daily_wape=float(dw),peak_mape=float(peak),peak_time_mae_minutes=float(np.mean(np.abs(yt-pt)[dv])*15),valid_points=int(m.sum()))

def baseline(d,origins,h=1,kind='last'):
    y=d['y']; out=[]
    for o in origins:
        if kind=='last': a=np.tile(y[:,o-1:o],(1,h,1))
        elif kind=='week': a=np.stack([y[:,o+j-7] if j<7 else y[:,o+(j%7)-7] for j in range(h)],axis=1)
        elif kind=='mean3': a=np.tile(y[:,o-3:o].mean(1)[:,None,:],(1,h,1))
        elif kind=='mean7': a=np.tile(y[:,o-7:o].mean(1)[:,None,:],(1,h,1))
        elif kind=='level': a=np.full((57,h,96),np.nan); a[:]=y[:,o-1].mean(-1)[:,None,None]
        else: raise ValueError(kind)
        out.append(a.reshape(57,h*96))
    return np.stack(out,axis=1)

def smooth(a,width):
    if width<=1:return a
    # Edge padding avoids mixing the end and start of the forecast day.
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(a,size=width,axis=-1,mode='nearest')
