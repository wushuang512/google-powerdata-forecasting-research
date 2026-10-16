import hashlib,datetime
from lib import *
from scipy.optimize import minimize
import pandas as pd

def solve(a,y,m,minimum=.5,reg=.15):
    k=len(a);prior=np.zeros(k);prior[0]=1
    norm=100*m/np.maximum((np.abs(y)*m).sum((1,2),keepdims=True),1e-12)/len(y)
    def obj(w):return (np.abs(y-np.einsum('k,ksoh->soh',w,a))*norm).sum()+reg*np.square(w-prior).sum()
    def grad(w):return -np.einsum('soh,ksoh->k',np.sign(y-np.einsum('k,ksoh->soh',w,a))*norm,a)+2*reg*(w-prior)
    r=minimize(obj,prior,jac=grad,bounds=[(minimum,1)]+[(0,1)]*(k-1),constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(k)}],method='SLSQP',options={'ftol':1e-10,'maxiter':300})
    w=np.maximum(r.x,0);return w/w.sum()

def main():
    d=load();y,m=truth(d,DEV);p={}
    for kind in ['chronos','tirex']:
        p.update({kind+'_'+k:v for k,v in dict(np.load(R/f'results/{kind}_dev.npz')).items()})
    p.update(dict(np.load(R/'results/residual_dev.npz')));p.update(dict(np.load(R/'results/tf3_dev.npz')))
    for kind in ['chronos','tirex']:p.update(dict(np.load(R/f'results/hierarchy_{kind}_dev.npz')))
    p['old_equal_mix']=(p['chronos_native']+p['tirex_native'])/2
    score={k:metrics(y,z,m) for k,z in p.items()}
    families={'chronos':[k for k in p if k.startswith('chronos')],'tirex':[k for k in p if k.startswith('tirex')],'residual':[k for k in p if k.startswith('residual')],'online':[k for k in p if k.startswith('online')],'timesfm3':[k for k in p if k.startswith('tf3')]}
    families['hierarchy']=[k for k in p if k.startswith('hier_')]
    # Family member selection uses only first four development target days.
    chosen={f:min(ks,key=lambda k:metrics(y[:,:4],p[k][:,:4],m[:,:4])['wape']) for f,ks in families.items() if ks}
    pool=['old_equal_mix']+list(chosen.values());a=np.stack([p[k] for k in pool]);cells=np.unique(d['cells']);grid=[];cv={}
    for minimum,reg in [(1,0)]+[(mm,rr) for mm in [0,.5,.75] for rr in [.03,.15,.5]]:
        z=np.empty_like(y[:,4:]);tag=f'min{minimum}_reg{reg}'
        for cell in cells:
            hold=d['cells']==cell;train=~hold
            w=np.array([1]+[0]*(len(pool)-1)) if minimum==1 else solve(a[:,train,:4],y[train,:4],m[train,:4],minimum,reg)
            z[hold]=np.einsum('k,ksoh->soh',w,a[:,hold,4:])
        cv[tag]=z;grid.append(dict(tag=tag,minimum=minimum,reg=reg,wape=metrics(y[:,4:],z,m[:,4:])['wape']))
    best=min(grid,key=lambda r:r['wape']);bestp=cv[best['tag']];eligible=[]
    for row in grid:
        dif=[]
        for cell in cells:
            ix=d['cells']==cell
            dif.append(metrics(y[ix,4:],cv[row['tag']][ix],m[ix,4:])['wape']-metrics(y[ix,4:],bestp[ix],m[ix,4:])['wape'])
        se=float(np.std(dif,ddof=1)/np.sqrt(len(dif)));row['paired_cell_se_vs_best']=se
        if row['wape']<=best['wape']+se+1e-10:eligible.append(row)
    rule=max(eligible,key=lambda r:(r['minimum'],r['reg']))
    w=np.array([1]+[0]*(len(pool)-1),float) if rule['minimum']==1 else solve(a,y,m,rule['minimum'],rule['reg'])
    pred=np.clip(np.einsum('k,ksoh->soh',w,a),0,1);p['new_curve']=pred
    needed=list(dict.fromkeys(pool[1:]+['chronos_native','tirex_native']))
    sel=dict(frozen_at=datetime.datetime.now().astimezone().isoformat(),pool=pool,weights=w.tolist(),weight_map=dict(zip(pool,w.tolist())),needed=needed,chosen=chosen,rule=rule,cv_grid=grid,development_score=metrics(y,pred,m),old_development_score=score['old_equal_mix'],test_used_for_this_selection=False,already_seen_month=True,family_selection_days='May15-18',weight_cv='fit May15-18 on nine cells; evaluate May19-22 on held-out cell; paired one-standard-error rule',final_weight_fit='May15-22 all cells',online_errors='Only actual outcomes already observed before forecast origin',restriction='No independent generalization claim from this previously examined month')
    content=json.dumps(sel,indent=2);(R/'results/selection.json').write_text(content,encoding='utf-8',newline='\n');(R/'results/selection.sha256').write_text(hashlib.sha256(content.encode()).hexdigest())
    secondary=min(families['timesfm3'],key=lambda k:score[k]['wape'])
    extra=dict(frozen_at=sel['frozen_at'],single_model=secondary,ensemble_rule='equal weights old_equal_mix and this single model',reason='Best TimesFM3 on complete development period; secondary exploratory comparison, not primary nested selection',test_used=False)
    content2=json.dumps(extra,indent=2);(R/'results/secondary.json').write_text(content2,encoding='utf-8',newline='\n');(R/'results/secondary.sha256').write_text(hashlib.sha256(content2.encode()).hexdigest())
    np.savez_compressed(R/'results/dev_selected.npz',**p);np.savez_compressed(R/'results/weight_cv.npz',**cv)
    pd.DataFrame([dict(model=k,**metrics(y,z,m)) for k,z in p.items()]).sort_values('wape').to_csv(R/'results/development_all.csv',index=False)
    print(json.dumps(sel,indent=2),flush=True)

if __name__=='__main__':main()
