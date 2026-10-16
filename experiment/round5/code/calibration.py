import argparse,pickle
from lib5 import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
CONFIGS={f'residual_ridge{a}':a for a in [1000,10000,100000]}
def bank(splits):
    result={}
    for split in splits:
        for j,o in enumerate({'train':TRAIN,'dev':DEV,'test':TEST}[split]):result[o]=anchor(split)[:,j]
    return result
def features(d,o,b):
    base=b[o];y=d['y'];prod=d['prod'];pv=d['pvalid'];items=[];names=[]
    def add(v,name):
        if v.ndim==1:v=np.repeat(v[:,None],96,axis=1)
        items.append(v.reshape(-1,1));names.append(name)
    add(base,'base');add(base-base.mean(1)[:,None],'predicted_shape');add(y[:,o-1]-base,'last_minus_base');add(y[:,o-7]-base,'week_minus_base');add(y[:,o-3:o].mean(1)-base,'mean3_minus_base')
    for k in [1,3,7]:
        add(y[:,o-k:o].mean((1,2))-base.mean(1),f'level{k}_minus_base');add(y[:,o-k:o].std((1,2)),f'power_std{k}')
    count=pv[:,o-7:o].sum((1,2));pmean=(prod[:,o-7:o]*pv[:,o-7:o]).sum((1,2))/np.maximum(count,1)
    add(pmean,'past_proxy_mean');add(count/(7*96),'proxy_availability7');add(pv[:,o-1].astype(float),'proxy_slot_available');add(np.where(pv[:,o-1],prod[:,o-1],pmean[:,None]),'last_proxy')
    avail=[k for k in range(max(7,o-7),o) if k in b]
    if avail:
        errs=np.stack([y[:,k]-b[k] for k in avail],1);mask=np.stack([d['valid'][:,k] for k in avail],1);daily=(errs*mask).sum(2)/mask.sum(2).clip(1)
        bias=(errs*mask).sum((1,2))/mask.sum((1,2)).clip(1);shape=(errs*mask).sum(1)/mask.sum(1).clip(1);last=daily[:,-1]
    else:bias=np.zeros(57);shape=np.zeros((57,96));last=np.zeros(57)
    add(bias,'observed_bias7');add(last,'observed_bias1');add(shape,'observed_error_shape7')
    slot=np.arange(96);add(np.broadcast_to(np.sin(2*np.pi*slot/96),(57,96)),'hour_sin');add(np.broadcast_to(np.cos(2*np.pi*slot/96),(57,96)),'hour_cos')
    for c in np.unique(d['cells']):add((d['cells']==c).astype(float),f'cell_{c}')
    return np.concatenate(items,1),base,names
def fit(d,b,origins,alpha):
    x=[];y=[];weight=[]
    for o in origins:
        xx,base,n=features(d,o,b);m=d['valid'][:,o].ravel();x.append(xx[m]);y.append((d['y'][:,o]-base).ravel()[m]);weight.append(np.repeat(1/np.maximum(d['y'][:,:o].mean((1,2)),.05),96)[m])
    model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(np.concatenate(x),np.concatenate(y),ridge__sample_weight=np.concatenate(weight));return model,n
def predict(model,d,b,origins):
    return np.clip(np.stack([base+model.predict(x).reshape(57,96) for x,base,_ in [features(d,o,b) for o in origins]],1),0,1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='dev');a=ap.parse_args();d=load();origins=DEV if a.split=='dev' else TEST;trainorig=TRAIN if a.split=='dev' else TRAIN+DEV;b=bank(['train','dev']+(['test'] if a.split=='test' else []));configs=CONFIGS;p={};rows=[];y,m=truth(d,origins)
    if a.split=='test':s=json.loads((R/'results/selection.json').read_text());name=s['chosen']['calibration'];configs={name:CONFIGS[name]}
    for name,alpha in configs.items():
        model,names=fit(d,b,trainorig,alpha);p[name]=predict(model,d,b,origins)
        with (R/'models'/f'{name}_{a.split}.pkl').open('wb') as f:pickle.dump(model,f)
        rows.append(dict(model=name,**metrics(y,p[name],m)));np.savez_compressed(R/f'results/calibration_{a.split}.npz',**p);print('RESULT',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(R/f'results/calibration_{a.split}.csv',index=False);save_json(R/'docs/calibration_features.json',dict(features=names,local_training_origins=trainorig,actual_error_history='forecast origin k < current origin o only, masked valid labels'))
if __name__=='__main__':main()
