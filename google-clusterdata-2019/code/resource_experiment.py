import sys,json,pickle,hashlib,datetime,copy
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];B=ROOT.parent/'experiment';R=B/'round6'
sys.path.insert(0,str(B/'round5/code'))
from lib5 import load,truth,metrics,TRAIN,DEV,TEST,anchor,save_json
from calibration import bank,features as basefeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from assess import comparisons
CONFIGS={f'ridge_{group}_{alpha}':dict(kind='ridge',resource=group=='resource',alpha=alpha) for group in ['control','resource'] for alpha in [10000,100000,1000000]}
CONFIGS.update({f'trees_resource_{leaf}':dict(kind='trees',resource=True,leaf=leaf) for leaf in [7,15]})
def loadr():return np.load(ROOT/'aggregated/resources_15min.npz')['resource']
def features(d,o,b,r,resource):
    x,bb,names=basefeatures(d,o,b)
    if not resource:return x,bb,names
    add=[];nn=[]
    for k in [1,3,7]:
        h=r[:,o-k:o];last=h[:,-1].mean(1);mean=h.mean((1,2));sd=h.std((1,2))
        for j,n in enumerate(['observed','active','cpu_capacity','memory_capacity','cpu_known','memory_known','missing_reason','mapping']):
            for a,s in [(mean,'mean'),(sd,'std')]:add.append(np.repeat(a[:,j,None],96,1).reshape(-1,1));nn.append(f'resource_{n}_{s}{k}')
        for j in [1,2,3]:add.append(np.repeat((last[:,j]-mean[:,j])[:,None],96,1).reshape(-1,1));nn.append(f'resource_change_{j}_{k}')
    # Slot profiles from the last observed day; no future resource state is supplied.
    for j in [0,1,2,3,4,5,6]:add.append(r[:,o-1,:,j].reshape(-1,1));nn.append(f'resource_last_slot_{j}')
    return np.concatenate([x,*add],1),bb,names+nn
def fit(d,b,r,origins,cfg):
    xs=[];ys=[];weights=[]
    for o in origins:
        x,bb,nn=features(d,o,b,r,cfg['resource']);m=d['valid'][:,o].ravel();xs.append(x[m]);ys.append((d['y'][:,o]-bb).ravel()[m]);weights.append(np.repeat(1/np.maximum(d['y'][:,:o].mean((1,2)),.05),96)[m])
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(weights)
    if cfg['kind']=='ridge':
        model=make_pipeline(StandardScaler(),Ridge(alpha=cfg['alpha']));model.fit(x,y,ridge__sample_weight=w)
    else:
        model=HistGradientBoostingRegressor(max_leaf_nodes=cfg['leaf'],max_iter=100,learning_rate=.03,l2_regularization=10,min_samples_leaf=300,early_stopping=False,random_state=20261001);model.fit(x,y,sample_weight=w)
    return model,nn
def predict(model,d,b,r,origins,cfg):return np.clip(np.stack([bb+model.predict(x).reshape(57,96) for x,bb,_ in [features(d,o,b,r,cfg['resource']) for o in origins]],1),0,1)
def prior(split):
    if split=='dev':
        p=np.load(B/'round5/results/calibration_dev.npz')['residual_ridge100000'];return .5*anchor('dev')+.5*p
    return np.load(B/'round5/results/test_candidates.npz')['selected']
def main():
    for sub in ['code','docs','models','results']: (R/sub).mkdir(parents=True,exist_ok=True)
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),registry=CONFIGS,training_origins=TRAIN,development_origins=DEV,final_training_origins=TRAIN+DEV,test_origins=TEST,selection='First 4 development days choose family/config; last 4 choose alpha by paired-cell one-standard-error rule, favor alpha 0; test labels are not loaded before hash is frozen',resource='Complete official machine_events, causal state only; no actual task usage or resource future',comparison='Resource models are controlled against identical regularization without resources; partial usage sample not a modeling input',independent_test=False)
    if not (R/'docs/protocol.json').exists():save_json(R/'docs/protocol.json',protocol)
    d=load();r=loadr();b=bank(['train','dev']);p={};rows=[];y,m=truth(d,DEV);base=prior('dev')
    for name,cfg in CONFIGS.items():
        model,nn=fit(d,b,r,TRAIN,cfg);p[name]=predict(model,d,b,r,DEV,cfg)
        with (R/'models'/f'{name}_dev.pkl').open('wb') as f:pickle.dump(model,f)
        row=dict(model=name,**metrics(y,p[name],m));rows.append(row);print('DEV',row,flush=True)
    np.savez_compressed(R/'results/dev_predictions.npz',**p,prior=base);pd.DataFrame(rows).to_csv(R/'results/dev_metrics.csv',index=False)
    best=min(p,key=lambda k:metrics(y[:,:4],p[k][:,:4],m[:,:4])['wape']);candidates=[]
    for a in [0,.1,.25,.5,1]:
        z=np.clip((1-a)*base[:,4:]+a*p[best][:,4:],0,1);per=100*(np.abs(y[:,4:]-z)*m[:,4:]).sum((1,2))/(y[:,4:]*m[:,4:]).sum((1,2));ref=100*(np.abs(y[:,4:]-base[:,4:])*m[:,4:]).sum((1,2))/(y[:,4:]*m[:,4:]).sum((1,2));v=pd.DataFrame(dict(cell=d['cells'],difference=ref-per)).groupby('cell').difference.mean().to_numpy();se=v.std(ddof=1)/np.sqrt(len(v));candidates.append(dict(alpha=a,wape=float(per.mean()),paired_cell_se=float(se)))
    win=min(candidates,key=lambda x:x['wape']);eligible=[x for x in candidates if x['wape']<=win['wape']+win['paired_cell_se']+1e-12];chosen=min(eligible,key=lambda x:x['alpha']);selection=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),chosen_config=best,config=CONFIGS[best],chosen=chosen,validation=candidates,protocol_sha256=hashlib.sha256((R/'docs/protocol.json').read_bytes()).hexdigest())
    if (R/'results/selection.json').exists():
        original=json.loads((R/'results/selection.json').read_text());assert hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest()==(R/'results/selection.sha256').read_text();save_json(R/'results/corrected_dev_selection_audit.json',dict(original=original,recomputed_development_only=selection,correction='Exclude machine IDs 102 and 124 with ambiguous PDU memberships. Preserve original frozen config and alpha; do not reselect after seeing test results.'));selection=original;best=selection['chosen_config'];chosen=selection['chosen']
    else:
        save_json(R/'results/selection.json',selection);(R/'results/selection.sha256').write_text(hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest())
    print('FROZEN',selection,flush=True)
    # Test starts only here. Retain matched controls for attribution; no ranking drives selection.
    b=bank(['train','dev','test']);ty,tm=truth(d,TEST);test={};configs={best:CONFIGS[best]}
    for alpha in [10000,100000,1000000]:
        for group in ['control','resource']:key=f'ridge_{group}_{alpha}';configs[key]=CONFIGS[key]
    for name,cfg in configs.items():
        model,nn=fit(d,b,r,TRAIN+DEV,cfg);test[name]=predict(model,d,b,r,TEST,cfg)
        with (R/'models'/f'{name}_final.pkl').open('wb') as f:pickle.dump(model,f)
        print('TEST',name,metrics(ty,test[name],tm),flush=True)
    test['prior']=prior('test');a=chosen['alpha'];test['selected']=np.clip((1-a)*test['prior']+a*test[best],0,1);scores={k:metrics(ty,z,tm) for k,z in test.items()};comparison=comparisons(ty,tm,test['selected'],test['prior'],d['cells']);np.savez_compressed(R/'results/test_predictions.npz',**test);pd.DataFrame([dict(model=k,**v) for k,v in scores.items()]).to_csv(R/'results/test_metrics.csv',index=False)
    causal={}
    for o in [14,22,26,30]:
        dd=copy.deepcopy(d);rr=r.copy();rr[:,o:]=999
        for k in ['y','prod']:dd[k][:,o:]=999
        for k in ['valid','pvalid']:dd[k][:,o:]=~dd[k][:,o:]
        causal[f'features_causal_origin{o}']=bool(np.array_equal(features(d,o,b,r,True)[0],features(dd,o,b,rr,True)[0]))
    selection_frozen=hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest()==(R/'results/selection.sha256').read_text();causal['selection_hash']=selection_frozen
    summary=dict(scores=scores,selected=selection,comparison_vs_round5=comparison,resource_gain_controlled={str(alpha):scores[f'ridge_control_{alpha}']['wape']-scores[f'ridge_resource_{alpha}']['wape'] for alpha in [10000,100000,1000000]},causality_checks=causal,all_passed=all(causal.values()),actual_task_usage_used=False,full_download=False,large_improvement=False,independent_confirmation=False)
    save_json(R/'results/summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
