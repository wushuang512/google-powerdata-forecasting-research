import argparse,json,pickle
import numpy as np,pandas as pd
import resource_experiment as ex
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sample',action='store_true');ap.add_argument('--candidate-only',action='store_true');a=ap.parse_args();r=ex.B/('round7' if a.sample else 'round6');s=json.loads((r/'results/selection.json').read_text());d=ex.load();b=ex.bank(['train','dev','test']);b[31]=np.load(ex.B/'round4/results/deployment_origin31_1day.npz')['best_mix'][:,0]
    if a.sample:
        import sample_experiment as sample
        rr=sample.loadr();features=sample.features
    else:rr=ex.loadr();features=ex.features
    name=s['chosen_config'];cfg=s['config']
    with (r/'models'/f'{name}_final.pkl').open('rb') as f:model=pickle.load(f)
    x,base,_=features(d,31,b,rr,cfg['resource']);cand=np.clip(base+model.predict(x).reshape(57,96),0,1)
    primary=pd.read_csv(ex.B/'round5/results/selected_deployment_origin31_1day.csv');prior=primary.predicted_util.to_numpy().reshape(57,96);alpha=s['chosen']['alpha'];chosen=cand if a.candidate_only else np.clip((1-alpha)*prior+alpha*cand,0,1);out=primary[['series','cell','local_trace_time_PT']].copy();out['predicted_util']=chosen.ravel();out['candidate_util']=cand.ravel();out['prior_selected_util']=prior.ravel();path=r/'results'/('candidate_deployment_origin31_1day.csv' if a.candidate_only else 'selected_deployment_origin31_1day.csv');out.to_csv(path,index=False);print('Saved',path,flush=True)
if __name__=='__main__':main()
