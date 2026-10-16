import gc,pickle
from lib5 import *
from train_fusion import train as train_fusion,predict as predict_fusion,CONFIGS as FCONFIG
import finetune_tf3 as ft
from calibration import fit,bank,predict as predict_calibration,CONFIGS as CCONFIG
def main():
    d=load();s=json.loads((R/'results/selection.json').read_text());assert hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest()==(R/'results/selection.sha256').read_text();p={};q={};w={};gates={};rows=[];y,m=truth(d,TEST)
    def record(k,z):p[k]=z;rows.append(dict(model=k,**metrics(y,z,m)));np.savez_compressed(R/'results/test_candidates.npz',**p);pd.DataFrame(rows).to_csv(R/'results/test_candidates.csv',index=False);print('RESULT',rows[-1],flush=True)
    key=s['chosen']['fusion'];name,ep=key.rsplit('_e',1);allq=[];allw=[]
    for seed in s['repeat_seeds']:
        net,hist=train_fusion(d,22,FCONFIG[name],seed,int(ep));pq,wq,gate=predict_fusion(net,d,TEST);torch.save(dict(state=net.state_dict(),config=FCONFIG[name],cutoff=22,epoch=int(ep),seed=seed),R/'models'/f'{key}_final_seed{seed}.pt');allq.append(pq);allw.append(wq);record(f'fusion_seed{seed}',pq[:,:,:,1]);save_json(R/'results'/f'fusion_final_seed{seed}_history.json',hist);del net;gc.collect();torch.cuda.empty_cache()
    fq=np.mean(allq,0);fw=np.mean(allw,0);q['fusion']=fq;w['fusion']=fw;record('fusion',fq[:,:,:,1])
    # Fixed-width, fixed-epoch ablations; no further model choice from their TEST results.
    for ab in ['fusion32','no_cross','no_frequency','dense_moe']:
        net,hist=train_fusion(d,22,FCONFIG[ab],20261001,s['ablation_test_epoch']);pq,wq,gate=predict_fusion(net,d,TEST);torch.save(dict(state=net.state_dict(),config=FCONFIG[ab],cutoff=22,epoch=s['ablation_test_epoch'],seed=20261001),R/'models'/f'ablation_{ab}_final.pt');record('ablation_'+ab,pq[:,:,:,1]);gates[ab]=dict(mean_probability=gate.mean(0).tolist(),top2_route_counts=np.bincount(np.argsort(gate,1)[:,-2:].ravel(),minlength=3).tolist());del net;gc.collect();torch.cuda.empty_cache()
    key=s['chosen']['tf3'];name,step=key.rsplit('_s',1);cfg=dict(scope='head',lr=1e-4,objective='ensemble') if name=='tf3_ensemble_head_lr1e4' else ft.CONFIGS[name]
    f,hist=ft.train(d,22,cfg,int(step));ft.save_delta(f,R/'models'/f'{key}_final.pt',cfg,22,int(step));tfp=np.clip(ft.forecast(f,d,TEST,ft.BASECFG['tf3_cellprod']),0,1);record('tf3_finetuned_single',tfp);record('tf3',np.clip(anchor('test')+.5*(tfp-np.load(B/'round3/results/tf3_test.npz')['tf3_cellprod']),0,1));save_json(R/'results/tf3_final_training.json',hist);del f;gc.collect();torch.cuda.empty_cache()
    b=bank(['train','dev','test']);key=s['chosen']['calibration'];model,names=fit(d,b,TRAIN+DEV,CCONFIG[key]);record('calibration',predict_calibration(model,d,b,TEST))
    with (R/'models'/f'{key}_final.pkl').open('wb') as f:pickle.dump(model,f)
    record('anchor',anchor('test'));pick=s['selected'];record('selected',p['anchor'] if pick['alpha']==0 else np.clip((1-pick['alpha'])*p['anchor']+pick['alpha']*p[pick['family']],0,1))
    np.savez_compressed(R/'results/test_quantiles.npz',**q);np.savez_compressed(R/'results/test_proxy_quantiles.npz',**w);save_json(R/'results/ablation_gates.json',gates)
if __name__=='__main__':main()
