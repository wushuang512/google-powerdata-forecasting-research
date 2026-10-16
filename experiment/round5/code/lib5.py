import os,sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent;TOP=B.parent
sys.path.insert(0,str(B/'code'));sys.path.insert(0,str(B/'round2/code'));sys.path.insert(0,str(B/'round3/code'))
import numpy as np,pandas as pd,torch
from common import load,truth,metrics
torch.set_num_threads(4)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
TRAIN=list(range(7,14));DEV=list(range(14,22));TEST=list(range(22,31))
def anchor(split):
    path=B/f'round3/results/tf3_{split}.npz'
    f=np.load(path)['tf3_cellprod'];c=np.load(B/f'round3/results/chronos_{split}.npz')['native'];t=np.load(B/f'round3/results/tirex_{split}.npz')['native']
    return np.clip(.25*c+.25*t+.5*f,0,1)
def cell_ids(d):return np.array([sorted(set(d['cells'])).index(c) for c in d['cells']])
def history(d,ends,length=672,series=None):
    if series is None:series=np.tile(np.arange(57),len(ends));ends=np.repeat(ends,57)
    yf=d['y'].reshape(57,-1);pf=d['prod'].reshape(57,-1);vm=d['valid'].reshape(57,-1);pm=d['pvalid'].reshape(57,-1)
    x=np.stack([np.stack([yf[i,e-length:e],pf[i,e-length:e],vm[i,e-length:e],pm[i,e-length:e]],-1) for i,e in zip(series,ends)]).astype('float32')
    # Bad production values are placeholders, accompanied by an explicit mask.
    x[:,:,1]*=x[:,:,3]
    slots=np.stack([np.arange(e,e+96) for e in ends]);cal=np.stack([np.sin(2*np.pi*slots/96),np.cos(2*np.pi*slots/96),np.sin(2*np.pi*(slots/96+2)/7),np.cos(2*np.pi*(slots/96+2)/7)],-1).astype('float32')
    return x,cal,cell_ids(d)[series]
def tensor(a):return torch.as_tensor(a,device=DEVICE)
def save_json(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')
