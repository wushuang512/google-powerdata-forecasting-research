"""Fit historical statistical and sparse-expert families on each target prefix."""
from pathlib import Path
import argparse, datetime, hashlib, json, sys, time
sys.dont_write_bytecode=True
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('case');ap.add_argument('--cut',type=int);ap.add_argument('--phase',default='dev');a=ap.parse_args()
ROOT=Path(a.root);F=ROOT/'experiment/generalization_extended_20261013';OLD=ROOT/'experiment/generalization_validation_20261012'
sys.path.insert(0,str(OLD/'code'));from core import read,save,sha,utc,np_save
proto=read(F/'docs/protocol.json');assert sha(F/'docs/protocol.json')==(F/'docs/protocol.sha256').read_text().strip();cfg=proto['cases'][a.case]
z=np.load(F/'data'/f'{a.case}.npz');ppd=int(z['ppd']);dates=z['dates'];scale=z['scales'];y=z['y']/scale[:,None,None];x=z['x']/scale[:,None,None];valid=z['valid'];cut=a.cut or cfg['cutoff']
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib,lightgbm as lgb
status=F/'results'/f'{a.case}_stat_cut{cut}_{a.phase}_status.json'
compatible_scripts=[sha(Path(__file__))]
previous_script=Path(__file__).with_name('statistical_stage_v1.py')
if previous_script.exists():
    import ast
    old_ast=ast.parse(previous_script.read_text(encoding='utf-8'));new_ast=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    for function in ['features','bank_training','sparse_fit']:
        before=next(n for n in old_ast.body if getattr(n,'name',None)==function);after=next(n for n in new_ast.body if getattr(n,'name',None)==function)
        assert ast.dump(before)==ast.dump(after),function
    compatible_scripts.append(sha(previous_script))

def origins(H):
    if a.case.startswith('NLR'):return cfg['origins'][a.phase]
    if a.phase=='oof':return [7] if H==7 else list(range(7,14))
    if a.phase=='dev' and a.case=='Google2019':return list(range(14,22)) if H==1 else[14,15]
    return cfg['origins'][a.phase+str(H)]

def features(j,o,H):
    # Every stochastic feature comes from strictly completed observations.
    hist=x[j,max(0,o-14):o];assert len(hist)>0
    n=H*ppd;lead=np.arange(n)//ppd;slot=np.arange(n)%ppd;anchor=hist[-1,slot];cols=[anchor]
    for lag in [2,3,7,14]:cols.append(hist[-min(lag,len(hist)),slot])
    for width in [3,7,14]:
        h=hist[-width:];cols.extend([h.mean(0)[slot],h.std(0)[slot],np.full(n,h.mean()),np.full(n,h.std())])
    cols.extend([np.full(n,hist[-1].mean()),np.full(n,hist[-1].max()),np.full(n,hist[-1].min()),
        np.full(n,hist[-1].mean()-hist[-min(2,len(hist))].mean()),np.full(n,len(hist)),lead/7,
        np.sin(2*np.pi*slot/ppd),np.cos(2*np.pi*slot/ppd),np.sin(4*np.pi*slot/ppd),np.cos(4*np.pi*slot/ppd)])
    dow=datetime.date.fromisoformat(str(dates[o])).weekday();future=(dow+lead)%7
    cols.extend([np.sin(2*np.pi*future/7),np.cos(2*np.pi*future/7),(future>=5).astype(float),np.full(n,j/max(len(x)-1,1))])
    start_date=datetime.date.fromisoformat(str(dates[o]));year_day=np.array([(start_date+datetime.timedelta(days=int(v))).timetuple().tm_yday for v in lead])
    cols.extend([np.sin(2*np.pi*year_day/365.25),np.cos(2*np.pi*year_day/365.25)])
    return np.ascontiguousarray(np.stack(cols,1),dtype=np.float32),anchor

def bank_training(H):
    stride=7 if cut>60 else 1;rows=[];labels=[];weights=[];last_label=0;records=[]
    # Prefix-only partial targets are permitted; unknown future labels are never
    # synthesized. Lead availability is recorded for short data prefixes.
    for o in range(3,cut,stride):
        for j in range(len(x)):
            if valid[j,max(0,o-7):o].mean()<.9:continue
            X,anchor=features(j,o,H);length=min(H,cut-o)*ppd;mask=valid[j,o:min(o+H,cut)].reshape(-1)
            if not mask.any():continue
            yy=y[j,o:min(o+H,cut)].reshape(-1)-anchor[:length]
            rows.append(X[:length][mask]);labels.append(yy[mask]);weights.append(np.ones(int(mask.sum()))/max(length,ppd))
            records.append(dict(series=j,origin=o,label_end_exclusive=min(o+H,cut)));last_label=max(last_label,min(o+H,cut))
    assert rows and last_label<=cut
    X=np.concatenate(rows);Y=np.concatenate(labels);W=np.concatenate(weights)
    if len(X)>180000:
        ix=np.random.default_rng(20261013).choice(len(X),180000,replace=False);ix.sort();X,Y,W=X[ix],Y[ix],W[ix]
    return X,Y,W,records

def sparse_fit(X,Y,W,stored=None):
    import torch
    torch.manual_seed(20261013);torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else'cpu'
    mu=X.mean(0) if stored is None else stored['mean'];sd=X.std(0).clip(1e-4) if stored is None else stored['scale'];xx=(X-mu)/sd
    class SparseExperts(torch.nn.Module):
        def __init__(self,n):
            super().__init__();self.gate=torch.nn.Linear(n,4);self.experts=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(n,32),torch.nn.SiLU(),torch.nn.Linear(32,1)) for _ in range(4)])
        def forward(self,v):
            logits=self.gate(v);keep=logits.topk(2,-1).indices;selected=torch.full_like(logits,-float('inf')).scatter(1,keep,logits.gather(1,keep));gate=selected.softmax(-1)
            values=torch.cat([e(v) for e in self.experts],-1);return (gate*values).sum(-1)
    m=SparseExperts(X.shape[1]).to(device);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=.01);rng=np.random.default_rng(20261013);losses=[] if stored is None else stored['loss_history']
    if stored is not None:m.load_state_dict(stored['state'],strict=True)
    for step in range(200 if stored is None else 0):
        ix=rng.choice(len(X),min(256,len(X)),replace=False);v=torch.tensor(xx[ix],dtype=torch.float32,device=device);t=torch.tensor(Y[ix],dtype=torch.float32,device=device);w=torch.tensor(W[ix],dtype=torch.float32,device=device)
        loss=(torch.nn.functional.smooth_l1_loss(m(v),t,reduction='none',beta=.02)*w).sum()/w.sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(float(loss.detach()))
    state={k:v.detach().cpu() for k,v in m.state_dict().items()};m.to('cpu').eval();torch.cuda.empty_cache()
    def infer(A):
        out=[]
        with torch.inference_mode():
            for start in range(0,len(A),2048):out.append(m(torch.tensor((A[start:start+2048]-mu)/sd,dtype=torch.float32)).numpy())
        return np.concatenate(out)
    return dict(state=state,mean=mu,scale=sd,loss_history=losses,architecture='four32-hidden experts; top2 sparse softmax gating'),infer

def factor_fit(H):
    # Historical joint low-rank Fourier direct model. Basis and scalers are fit
    # only on the labelled training prefix; outputs are reconstructed in power.
    existing=F/'models'/f'{a.case}_JointFactorFourier_cut{cut}_H{H}.joblib'
    if existing.exists():
        saved=joblib.load(existing);pipe=saved['pipe'];basis=saved['basis'];mu=saved['mean'];sd=saved['scale'];B=saved['B'];rank=basis.shape[1]
        def feature(o):
            h=(x[:,max(0,o-7):o]-mu[:,None,None])/sd[:,None,None];latent=np.einsum('nk,ndt->kdt',basis,h)
            return np.concatenate([latent[:,-1].mean(-1),latent.mean((1,2)),latent.std((1,2)),latent[:,-min(2,len(h[0]))].mean(-1),latent[:,-1].reshape(-1)])[None]
        def infer(o):
            c=pipe.predict(feature(o)).reshape(rank,H,B.shape[1]);lat=np.einsum('kdc,tc->kdt',c,B);return np.einsum('nk,kdt->ndt',basis,lat)*sd[:,None,None]+mu[:,None,None]
        return saved,infer,dict(reused_frozen_model=True,training_end_exclusive=cut)
    count=valid[:,:cut].sum((1,2));mu=(y[:,:cut]*valid[:,:cut]).sum((1,2))/count
    sd=np.sqrt(((y[:,:cut]-mu[:,None,None])**2*valid[:,:cut]).sum((1,2))/count).clip(.01)
    A=np.where(valid[:,:cut],(y[:,:cut]-mu[:,None,None])/sd[:,None,None],0).reshape(len(x),-1)
    u,s,_=np.linalg.svd(A,full_matrices=False);rank=min(2,len(x));basis=u[:,:rank];basis*=np.where(basis[np.abs(basis).argmax(0),np.arange(rank)]>=0,1,-1)
    tick=np.arange(ppd)/ppd;B=np.stack([np.ones(ppd)]+[f(2*np.pi*k*tick) for k in [1,2,3] for f in [np.sin,np.cos]],1) if ppd>1 else np.ones((1,1))
    def feature(o):
        h=(x[:,max(0,o-7):o]-mu[:,None,None])/sd[:,None,None];latent=np.einsum('nk,ndt->kdt',basis,h)
        return np.concatenate([latent[:,-1].mean(-1),latent.mean((1,2)),latent.std((1,2)),latent[:,-min(2,len(h[0]))].mean(-1),latent[:,-1].reshape(-1)])[None]
    rows=[];target=[]
    for o in range(3,cut-H+1,7 if cut>60 else 1):
        if valid[:,o:o+H].mean()<.95:continue
        labels=(y[:,o:o+H]-mu[:,None,None])/sd[:,None,None];lat=np.einsum('nk,ndt->kdt',basis,labels)
        coef=np.stack([np.linalg.lstsq(B,lat[:,d].T,rcond=None)[0].T for d in range(H)],1)
        rows.append(feature(o)[0]);target.append(coef.reshape(-1))
    if len(rows)<2:return None,None,dict(status='insufficient_complete_prefix_windows',windows=len(rows))
    pipe=make_pipeline(StandardScaler(),Ridge(alpha=10));pipe.fit(np.array(rows),np.array(target))
    def infer(o):
        c=pipe.predict(feature(o)).reshape(rank,H,B.shape[1]);lat=np.einsum('kdc,tc->kdt',c,B);return np.einsum('nk,kdt->ndt',basis,lat)*sd[:,None,None]+mu[:,None,None]
    return dict(pipe=pipe,basis=basis,mean=mu,scale=sd,B=B),infer,dict(windows=len(rows),rank=rank,training_end_exclusive=cut)

save(status,dict(state='running',pid=__import__('os').getpid(),utc=utc()))
for H in [1,7]:
    X,Y,W,records=bank_training(H);models={};sparse_meta=None
    estimators={
        'Ridge_direct':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=10)),
        'LightGBM_direct':lgb.LGBMRegressor(objective='l1',num_leaves=15,n_estimators=300,learning_rate=.04,min_child_samples=40,reg_lambda=1,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True,random_state=20261013),
        'HGB_residual':HistGradientBoostingRegressor(loss='absolute_error',max_leaf_nodes=7,max_iter=150,l2_regularization=1,random_state=20261013)}
    for name,model in estimators.items():
        path=F/'models'/f'{a.case}_{name}_cut{cut}_H{H}.joblib'
        if path.exists():
            info=read(path.with_suffix('.json'));assert info['script_sha256'] in compatible_scripts and info['model_sha256']==sha(path);model=joblib.load(path)
        else:
            model.fit(X,Y);joblib.dump(model,path)
            save(path.with_suffix('.json'),dict(real_fit=True,training_rows=len(X),training_label_end_exclusive=max(r['label_end_exclusive'] for r in records),cutoff=cut,script_sha256=sha(Path(__file__)),model_sha256=sha(path),training_origins=records))
        models[name]=model.predict;print('FIT',a.case,name,cut,H,len(X),flush=True)
    if ppd==96:
        # An explicit neural small-model alternative to the large backbones.
        path=F/'models'/f'{a.case}_SparseMoE_cut{cut}_H{H}.joblib';meta_path=path.with_suffix('.json')
        stored=None
        if path.exists():
            info=read(meta_path);assert info['model_sha256']==sha(path) and info['script_sha256'] in compatible_scripts;stored=joblib.load(path)
        sm,infer=sparse_fit(X,Y,W,stored)
        if stored is None:
            joblib.dump(sm,path);save(meta_path,dict(real_optimizer_steps=200,training_rows=len(X),training_label_end_exclusive=max(r['label_end_exclusive'] for r in records),cutoff=cut,script_sha256=sha(Path(__file__)),model_sha256=sha(path),loss_history=sm['loss_history']))
        models['SparseMoE']=infer
    arrays={};oo=origins(H)
    for name,infer in models.items():
        out=[]
        for o in oo:
            fields=[];anchors=[]
            for j in range(len(x)):
                A,b=features(j,o,H);fields.append(A);anchors.append(b)
            A=np.concatenate(fields);p=infer(A).reshape(len(x),-1)+np.array(anchors)
            p=np.maximum(0,p)
            if a.case=='Google2019':p=np.clip(p,0,1)
            out.append(p)
        arrays[name]=np.stack(out,1)
    fp,fi,meta=factor_fit(H)
    if fi is not None:
        factor_path=F/'models'/f'{a.case}_JointFactorFourier_cut{cut}_H{H}.joblib'
        if not factor_path.exists():joblib.dump(fp,factor_path)
        p=np.stack([fi(o).reshape(len(x),-1) for o in oo],1);arrays['JointFactorFourier']=np.clip(p,0,1) if a.case=='Google2019' else np.maximum(0,p)
    else:save(F/'results'/f'{a.case}_Factor_cut{cut}_H{H}_unavailable.json',meta)
    for key,p in arrays.items():assert p.shape==(len(x),len(oo),H*ppd) and np.isfinite(p).all(),key
    np_save(F/'cache'/f'{a.case}_stat_cut{cut}_{a.phase}_H{H}.npz',**arrays,origins=np.array(oo),script_sha256=sha(Path(__file__)),protocol_sha256=sha(F/'docs/protocol.json'))
save(status,dict(state='complete',utc=utc(),actual_target_fits=True,cut=cut,case=a.case,phase=a.phase))
