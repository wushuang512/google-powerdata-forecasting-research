import os,sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent;R2=B/'round2'
sys.path.insert(0,str(R2/'code'));sys.path.insert(0,str(B/'code'))
import numpy as np
from common import load,truth,metrics,baseline,smooth
import torch
torch.set_num_threads(4)
TRAIN=list(range(7,14));DEV=list(range(14,22));TEST=list(range(22,31))

def coarse_inputs(d,o,factor=1,context_days=14,flip=False):
    y=d['y'][:,max(0,o-context_days):o].reshape(57,-1)
    if flip:y=1-y
    if factor>1:y=y.reshape(57,-1,factor).mean(-1)
    return y.astype('float32')

def expand(p,factor):
    if factor==1:return p
    # Coarse predictions are block averages. Linear interpolation uses centres.
    x=(np.arange(p.shape[-1])+.5)*factor-.5
    return np.stack([np.interp(np.arange(p.shape[-1]*factor),x,row) for row in p])

def cached_native(split,kind):
    if split=='dev':file=R2/f'results/{"development" if kind=="chronos" else "tirex_dev"}.npz'
    elif split=='test':file=R2/f'results/{"final_chronos24" if kind=="chronos" else "tirex_test"}.npz'
    else:return {}
    z=dict(np.load(file));return {'native':z['zero_uni' if kind=='chronos' else 'tirex_uni']}

