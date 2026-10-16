import argparse,pickle
from lib5 import *
from fusion_model import FusionMoE
from train_fusion import predict as fusion_predict
from calibration import bank,features
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fusion-only',action='store_true');a=ap.parse_args();d=load();s=json.loads((R/'results/selection.json').read_text());basefile=B/'round4/results/deployment_origin31_1day.npz';base=np.load(basefile)['best_mix'];key=s['chosen']['fusion'];allq=[];allw=[]
    for seed in s['repeat_seeds']:
        z=torch.load(R/'models'/f'{key}_final_seed{seed}.pt',map_location=DEVICE,weights_only=True);net=FusionMoE(**z['config']).to(DEVICE);net.load_state_dict(z['state']);q,w,g=fusion_predict(net,d,[31]);allq.append(q);allw.append(w)
    q=np.mean(allq,0);w=np.mean(allw,0);pick=s['selected'];family=pick['family'];alpha=pick['alpha'];candidate=base
    if family=='fusion':candidate=q[:,:,:,1]
    elif family=='tf3':
        from finetune_tf3 import load_delta,forecast,BASECFG
        f=load_delta(R/'models'/f'{s["chosen"]["tf3"]}_final.pt');fp=forecast(f,d,[31],BASECFG['tf3_cellprod']);candidate=np.clip(.5*np.load(basefile)['old_equal_mix']+.5*fp,0,1)
    elif family=='calibration':
        b=bank(['train','dev','test']);b[31]=base[:,0]
        with (R/'models'/f'{s["chosen"]["calibration"]}_final.pkl').open('rb') as f:model=pickle.load(f)
        x,bb,names=features(d,31,b);candidate=np.clip(bb+model.predict(x).reshape(57,96),0,1)[:,None]
    p=q[:,:,:,1] if a.fusion_only else np.clip((1-alpha)*base+alpha*candidate,0,1);level=d['y'][:,24:31].mean((1,2))[:,None,None].clip(.05)
    if a.fusion_only:
        c=s['fusion_interval']['widening_relative_to_past_mean'];lo=(q[:,:,:,0]-c*level).clip(0,1);hi=(q[:,:,:,2]+c*level).clip(0,1)
    else:
        c=s['residual_interval']['calibrated_halfwidth_relative_to_past_mean'];lo=(p-c*level).clip(0,1);hi=(p+c*level).clip(0,1)
    prefix='fusion' if a.fusion_only else 'selected';rows=[]
    for i,n in enumerate(d['names']):
        for t in range(96):rows.append((n,d['cells'][i],str(pd.Timestamp('2019-06-01')+pd.Timedelta(minutes=15*t)),p[i,0,t],lo[i,0,t],hi[i,0,t],w[i,0,t,1]))
    path=R/f'results/{prefix}_deployment_origin31_1day.csv';pd.DataFrame(rows,columns=['series','cell','local_trace_time_PT','predicted_util','lower80_empirical','upper80_empirical','production_proxy_predicted_util']).to_csv(path,index=False);print('Saved',path,flush=True)
if __name__=='__main__':main()
