import json,hashlib,datetime
from round2_lib import *
from scipy.optimize import minimize
from selection import peak_adjust
import pandas as pd

def weightfit(a,y,m,prior=None,reg=.03):
    k=a.shape[0];prior=np.ones(k)/k if prior is None else prior
    norm=100*m/np.maximum((np.abs(y)*m).sum((1,2),keepdims=True),1e-10)/len(y)
    def fun(w):return (np.abs(y-np.einsum('k,ksoh->soh',w,a))*norm).sum()+reg*np.square(w-prior).sum()
    def jac(w):return -np.einsum('soh,ksoh->k',np.sign(y-np.einsum('k,ksoh->soh',w,a))*norm,a)+2*reg*(w-prior)
    r=minimize(fun,prior,jac=jac,bounds=[(0,1)]*k,constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],method='SLSQP',options={'ftol':1e-10,'maxiter':250})
    w=np.maximum(r.x,0);return w/w.sum()

def fitted_weights(a,y,m,cells,kind='global',rho=0):
    w=weightfit(a,y,m);wm=np.repeat(w[:,None],len(y),axis=1)
    if kind!='global':
        labels=cells if kind=='cell' else np.arange(len(y))
        for group in np.unique(labels):
            ix=labels==group;v=weightfit(a[:,ix],y[ix],m[ix],w,reg=.10)
            wm[:,ix]=(1-rho)*w[:,None]+rho*v[:,None]
    return wm

def apply_weights(p,selection,d,origins):
    a=np.stack([p[k] for k in selection['pool']]);w=np.array(selection['weights'])
    raw=np.einsum('ks,ksoh->soh',w,a)
    curve=np.clip(smooth(raw,selection.get('smooth_width',1)),0,1)
    return curve,peak_adjust(d,curve,origins,selection['peak_beta'],'last')

def history(d,origins,h=96,width=13):
    return np.clip(smooth(baseline(d,origins,h=h//96),width),0,1)

def main():
    d=load();y,m=truth(d,DEV);allp={}
    for file in ['development.npz','extras.npz','tirex_dev.npz']:
        allp.update(dict(np.load(R/'results'/file)))
    for w in [5,13,25]:allp[f'history{w}']=history(d,DEV,width=w)
    allp['last']=baseline(d,DEV);allp['old_equal_mix']=(allp['zero_uni']+allp['tirex_uni'])/2
    scores={k:metrics(y,z,m) for k,z in allp.items()}
    # One member per family limits redundant fit degrees of freedom.
    families={
        'chronos_uni':[k for k in allp if ('uni' in k and 'tirex' not in k)],
        'chronos_cov':[k for k in allp if 'cov' in k and 'tirex' not in k and 'calendar' not in k],
        'chronos_peer':[k for k in allp if 'peer' in k or k in ['zero_cell','zero_cellprod','zero_all']],
        'chronos_parts':[k for k in allp if 'parts' in k],
        'tirex':[k for k in allp if k.startswith('tirex')],
        'history':[k for k in allp if k.startswith('history')]
    }
    chosen={f:min(names,key=lambda k:scores[k]['wape']) for f,names in families.items() if names}
    pool=list(dict.fromkeys(list(chosen.values())+['old_equal_mix']))
    a=np.stack([allp[k] for k in pool]);grid=[]
    for kind,rhos in [('global',[0]),('cell',[.25,.5,.75]),('domain',[.25,.5,.75])]:
        for rho in rhos:
            w=fitted_weights(a[:,:,:4],y[:,:4],m[:,:4],d['cells'],kind,rho)
            z=np.einsum('ks,ksoh->soh',w,a[:,:,4:]);v=metrics(y[:,4:],z,m[:,4:])
            grid.append(dict(kind=kind,rho=rho,forward_validation_wape=v['wape']))
    best=min(grid,key=lambda g:g['forward_validation_wape']);w=fitted_weights(a,y,m,d['cells'],best['kind'],best['rho'])
    raw=np.einsum('ks,ksoh->soh',w,a)
    smooth_grid=[dict(width=width,**metrics(y,np.clip(smooth(raw,width),0,1),m)) for width in [1,3,5,9]]
    smooth_width=min(smooth_grid,key=lambda row:row['wape'])['width']
    allp['new_curve']=np.clip(smooth(raw,smooth_width),0,1)
    scores['new_curve']=metrics(y,allp['new_curve'],m)
    peakrows=[]
    for beta in [0,.25,.5,.75,1]:
        z=peak_adjust(d,allp['new_curve'],DEV,beta,'last');peakrows.append(dict(beta=beta,**metrics(y,z,m)))
    peak=min([g for g in peakrows if g['wape']<=scores['new_curve']['wape']+.02],key=lambda g:g['peak_mape'])
    allp['new_peak']=peak_adjust(d,allp['new_curve'],DEV,peak['beta'],'last');scores['new_peak']=metrics(y,allp['new_peak'],m)
    needed=list(dict.fromkeys([k for k in pool if k!='old_equal_mix']+['zero_uni','tirex_uni']))
    sel=dict(pool=pool,weights=w.tolist(),mean_weights=dict(zip(pool,w.mean(1).tolist())),chosen=chosen,needed=needed,primary='new_curve',peak_beta=peak['beta'],weight_rule=best,weight_grid=grid,peak_grid=peakrows,development_scores={k:scores[k] for k in pool+['new_curve','new_peak']},test_used_for_selection=False,adaptive_same_month=True)
    sel['frozen_at']=datetime.datetime.now().astimezone().isoformat()
    sel['smooth_width']=smooth_width;sel['smooth_grid']=smooth_grid
    content=json.dumps(sel,indent=2);(R/'results/selection.json').write_text(content);(R/'results/selection.sha256').write_text(hashlib.sha256(content.encode()).hexdigest())
    np.savez_compressed(R/'results/development_selected.npz',**allp)
    pd.DataFrame([dict(model=k,**metrics(y,z,m)) for k,z in allp.items()]).sort_values('wape').to_csv(R/'results/development_all.csv',index=False)
    print(json.dumps({k:sel[k] for k in ['pool','mean_weights','weight_rule','peak_beta']},indent=2),flush=True)
    print('DEV PRIMARY',scores['new_curve'],flush=True)

if __name__=='__main__':main()
