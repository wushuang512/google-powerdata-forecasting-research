import argparse,pickle,time,warnings
from lib import *
from classical import features
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pandas as pd

CONFIGS={**{f'ridge{a}':dict(kind='ridge',alpha=a) for a in [10,100,1000,10000,100000]},
 **{f'lgb_l{l}_n{n}_m{c}':dict(kind='lgb',leaves=l,trees=n,child=c) for l,n,c in [(7,100,500),(7,300,500),(15,100,300),(15,300,300),(31,200,500),(7,500,1000)]}}

def get_bank(splits):
    bank={}
    for split in splits:
        origins={'train':TRAIN,'dev':DEV,'test':TEST}[split]
        c=np.load(R/f'results/chronos_{split}.npz')['native'];t=np.load(R/f'results/tirex_{split}.npz')['native']
        for j,o in enumerate(origins):bank[o]=(c[:,j],t[:,j])
    return bank

def xfeatures(d,o,bank):
    pc,pt=bank[o];base=(pc+pt)/2
    x,names=features(d,o,True);n=57;out=[]
    def add(a,name):
        if a.ndim==1:a=np.repeat(a[:,None],96,axis=1)
        out.append(a.reshape(-1,1));names.append(name)
    add(base,'base_forecast');add(pc-pt,'foundation_disagreement')
    add(smooth(base,5),'base_smooth5');add(smooth(base,13),'base_smooth13')
    add(base-d['y'][:,o-1],'base_minus_last')
    for label,fun in [('mean',np.mean),('max',np.max),('min',np.min),('std',np.std)]:add(fun(base,axis=1),'base_'+label)
    add(base-base.mean(1)[:,None],'base_shape')
    add(d['y'][:,o-1,-16:].mean(1)-base.mean(1),'recent_minus_predicted_level')
    avail=[k for k in range(max(7,o-7),o) if k in bank]
    errors=np.stack([d['y'][:,k]-(bank[k][0]+bank[k][1])/2 for k in avail],axis=1) if avail else np.zeros((57,1,96))
    add(errors.mean((1,2)),'past_bias7');add(np.median(errors.mean(2),axis=1),'past_bias_median7');add(errors[:,-1].mean(1),'past_bias1')
    add(smooth(errors.mean(1),13),'past_error_shape')
    add(np.full(57,len(avail),dtype=float),'past_error_days')
    add(np.arange(57,dtype=float),'domain_index')
    cells={k:j for j,k in enumerate(np.unique(d['cells']))};add(np.array([cells[c] for c in d['cells']],float),'cell_index')
    return np.concatenate([x,*out],axis=1),names,base

def fit(d,bank,origins,cfg):
    xx=[];yy=[];ww=[]
    for o in origins:
        x,names,base=xfeatures(d,o,bank);m=d['valid'][:,o].ravel()
        xx.append(x[m]);yy.append((d['y'][:,o]-base).ravel()[m]);ww.append(np.repeat(1/np.maximum(d['y'][:,:o].mean((1,2)),.05),96)[m])
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww)
    if cfg['kind']=='ridge':
        model=make_pipeline(StandardScaler(),Ridge(alpha=cfg['alpha']));model.fit(x,y,ridge__sample_weight=w)
    else:
        model=LGBMRegressor(objective='regression_l1',n_estimators=cfg['trees'],num_leaves=cfg['leaves'],min_child_samples=cfg['child'],learning_rate=.025,reg_lambda=100,reg_alpha=.1,verbosity=-1,n_jobs=4,random_state=20260930,deterministic=True,force_col_wise=True)
        model.fit(x,y,sample_weight=w,feature_name=names,categorical_feature=['domain_index','cell_index'])
    return model

def predict_model(model,d,bank,origins):
    return np.stack([base+model.predict(x).reshape(57,96) for x,_,base in [xfeatures(d,o,bank) for o in origins]],axis=1)

def online(d,bank,origins,window,alpha,shape=False):
    out=[]
    for o in origins:
        base=(bank[o][0]+bank[o][1])/2;avail=[k for k in range(max(7,o-window),o) if k in bank]
        errors=np.stack([d['y'][:,k]-(bank[k][0]+bank[k][1])/2 for k in avail],axis=1)
        if shape:correction=smooth(np.median(errors,axis=1),13)
        else:correction=np.median(errors.mean(2),axis=1)[:,None]
        out.append(np.clip(base+alpha*correction,0,1))
    return np.stack(out,axis=1)

def main():
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    ap=argparse.ArgumentParser();ap.add_argument('--test',action='store_true');args=ap.parse_args()
    d=load();bank=get_bank(['train','dev']+(['test'] if args.test else []));origins=TEST if args.test else DEV
    y,m=truth(d,origins);path=R/f'results/residual_{"test" if args.test else "dev"}.npz';pred={};rows=[]
    cfgs=CONFIGS
    if args.test:
        sel=json.loads((R/'results/selection.json').read_text());cfgs={k:v for k,v in cfgs.items() if 'residual_'+k in sel['needed']}
    for name,cfg in cfgs.items():
        start=time.time();model=fit(d,bank,TRAIN+DEV if args.test else TRAIN,cfg);p=predict_model(model,d,bank,origins)
        pred['residual_'+name]=p;row=dict(model='residual_'+name,**metrics(y,np.clip(p,0,1),m));rows.append(row);print('RESULT',json.dumps(row),'seconds',time.time()-start,flush=True)
        with (R/'models'/f'{name}_{"final" if args.test else "dev"}.pkl').open('wb') as f:pickle.dump(model,f)
        np.savez_compressed(path,**pred)
    for window in [1,3,7]:
        for alpha in [.25,.5,1]:
            for shape in [False,True]:
                name=f'online_w{window}_a{alpha}_shape{int(shape)}';p=online(d,bank,origins,window,alpha,shape)
                pred[name]=p;rows.append(dict(model=name,**metrics(y,p,m)))
    np.savez_compressed(path,**pred);pd.DataFrame(rows).sort_values('wape').to_csv(R/f'results/residual_{"test" if args.test else "dev"}.csv',index=False)
    (R/'docs/residual_configs.json').write_text(json.dumps(CONFIGS,indent=2))
    print(pd.DataFrame(rows).sort_values('wape')[['model','wape','peak_mape']].head(10).to_string(index=False),flush=True)

if __name__=='__main__':main()
