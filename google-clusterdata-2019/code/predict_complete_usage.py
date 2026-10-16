"""Actual local inference for frozen round9 and its matched full-CPU neural model."""
import argparse,json,pickle,hashlib
import complete_usage_models as ex
import numpy as np,pandas as pd,torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fusion-candidate',action='store_true');a=ap.parse_args();d,v=ex.inputs();summary=json.loads((ex.R/'results/summary.json').read_text());s=summary['selection'];assert hashlib.sha256((ex.R/'results/selection.json').read_bytes()).hexdigest()==(ex.R/'results/selection.sha256').read_text();priorcsv=pd.read_csv(ex.B/'round8b/results/selected_deployment_origin31_1day.csv');base=priorcsv.predicted_util.to_numpy().reshape(57,96);key=f'fusion_cpu_e{summary["final_matched_neural_epoch"]}' if a.fusion_candidate else s['config'];cpu=np.full((57,96),np.nan)
    if key.startswith('fusion'):
        qq=[];ww=[]
        for seed in [20261001,20261002]:
            saved=torch.load(ex.R/'models'/f'{key}_final_seed{seed}.pt',weights_only=True,map_location=ex.DEVICE);net=ex.CapacityFusionMoE(**saved['config']).to(ex.DEVICE);net.load_state_dict(saved['state']);q,w,g=ex.neural_predict(net,d,v,[31],{31:base});qq.append(q);ww.append(w);del net;torch.cuda.empty_cache()
        candidate=np.mean(qq,0)[:,0,:,1];cpu[d['cells']=='a']=np.mean(ww,0)[d['cells']=='a',0,:,1]
    else:
        from complete_usage_experiment import REGISTRY
        cfg=REGISTRY[key];wf=dict(np.load(ex.R/'results/workload_forecasts.npz'));b=ex.bank(['train','dev','test']);b[31]=np.load(ex.B/'round4/results/deployment_origin31_1day.npz')['best_mix'][:,0]
        with (ex.R/'models'/f'{key}_final.pkl').open('rb') as f:model=pickle.load(f)
        if cfg['kind']=='ridge':
            x,bb,_=ex.residual_features(d,v,31,b,cfg['group'],wf);raw=np.clip(bb+model.predict(x).reshape(57,96),0,1);candidate=base.copy();candidate[d['cells']=='a']=raw[d['cells']=='a']
        else:candidate=ex.predict_candidate(model,d,v,[31],cfg,wf,base[:,None])[:,0]
        cpu[d['cells']=='a']=wf['31'][d['cells']=='a',:,0]
    alpha=1 if a.fusion_candidate else s['chosen']['alpha'];p=np.clip((1-alpha)*base+alpha*candidate,0,1);half=summary['uncertainty']['relative_halfwidth']*d['y'][:,24:31].mean((1,2)).clip(.05);out=priorcsv[['series','cell','local_trace_time_PT']].copy();out['predicted_util']=p.ravel();out['candidate_util']=candidate.ravel();out['prior_finetuned_util']=base.ravel();out['lower80_empirical']=np.maximum(p-half[:,None],0).ravel();out['upper80_empirical']=np.minimum(p+half[:,None],1).ravel();out['candidate_cpu_ncu']=cpu.ravel();out['has_complete_cpu_history']=np.repeat(d['cells']=='a',96);prefix='fusion_candidate' if a.fusion_candidate else 'selected';path=ex.R/f'results/{prefix}_deployment_origin31_1day.csv';out.to_csv(path,index=False)
    checks=dict(rows5472=len(out)==5472,power_finite_bounds=bool(np.isfinite(p).all() and p.min()>=0 and p.max()<=1),cpu_missing_outside_a=bool(np.isnan(cpu[d['cells']!='a']).all()),cpu_finite_nonnegative_in_a=bool(np.isfinite(cpu[d['cells']=='a']).all() and (cpu[d['cells']=='a']>=0).all()),interval_order=bool(((out.lower80_empirical<=out.predicted_util)&(out.predicted_util<=out.upper80_empirical)).all()),input_end='2019-05-31',forecast_start='2019-06-01',selection_hash=True,selected_new_weight=alpha,not_live_forecast=True,interval_scope='Width calibrated for selected point predictor; fusion candidate width reuse is an illustrative research interval, not separately calibrated')
    ex.save_json(ex.R/f'results/{prefix}_deployment_verification.json',checks);print(path,checks,flush=True)
if __name__=='__main__':main()
