import datetime
from lib5 import *
def halfwidth(d,origins,p,y,m):
    levels=np.stack([d['y'][:,o-7:o].mean((1,2)) for o in origins],1)[:,:,None].clip(.05)
    scores=(np.abs(y-p)/levels)[m]
    return float(np.quantile(scores,.8,method='higher'))
def main():
    d=load();y,m=truth(d,DEV);base=anchor('dev');repeat=json.loads((R/'results/repeat_plan.json').read_text());fusion=np.load(R/'results/fusion_repeat_dev.npz')['equal_seed_mean']
    fts=dict(np.load(R/'results/tf3_dev.npz'));fts.update(dict(np.load(R/'results/tf3_ensemble_dev.npz')));originaltf=np.load(B/'round3/results/tf3_dev.npz')['tf3_cellprod'];ftcomposite={k:np.clip(base+.5*(v-originaltf),0,1) for k,v in fts.items()}
    cal=dict(np.load(R/'results/calibration_dev.npz'));chosen={'fusion':repeat['chosen'],'tf3':min(ftcomposite,key=lambda k:metrics(y[:,:4],ftcomposite[k][:,:4],m[:,:4])['wape']),'calibration':min(cal,key=lambda k:metrics(y[:,:4],cal[k][:,:4],m[:,:4])['wape'])}
    families={'fusion':fusion,'tf3':ftcomposite[chosen['tf3']],'calibration':cal[chosen['calibration']]};rows=[];pred={'anchor':base}
    def per(p):return 100*(np.abs(y[:,4:]-p[:,4:])*m[:,4:]).sum((1,2))/(np.abs(y[:,4:])*m[:,4:]).sum((1,2))
    base_per=per(base);rows.append(dict(key='anchor',family='anchor',alpha=0,wape=float(base_per.mean()),paired_cell_se=0,improvement_pp=0))
    for family,pp in families.items():
        for alpha in [.1,.25,.5,1]:
            p=np.clip((1-alpha)*base+alpha*pp,0,1);key=f'{family}_a{alpha}';pred[key]=p;diff=base_per-per(p);bycell=np.array([diff[d['cells']==c].mean() for c in np.unique(d['cells'])]);rows.append(dict(key=key,family=family,alpha=alpha,wape=float(per(p).mean()),paired_cell_se=float(bycell.std(ddof=1)/np.sqrt(len(bycell))),improvement_pp=float(diff.mean())))
    best=min(rows,key=lambda r:r['wape']);eligible=[r for r in rows if r['wape']<=best['wape']+best['paired_cell_se']+1e-12];pick=min(eligible,key=lambda r:(r['alpha'],r['wape']))
    # Calibration diagnostics use earlier four DEV days to calibrate and later four to evaluate.
    levels=np.stack([d['y'][:,o-7:o].mean((1,2)) for o in DEV],1)[:,:,None].clip(.05);p=pred[pick['key']];early_c=halfwidth(d,DEV[:4],p[:,:4],y[:,:4],m[:,:4]);late_cover=((np.abs(y[:,4:]-p[:,4:])<=early_c*levels[:,4:])&m[:,4:]).sum()/m[:,4:].sum()
    sel=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),chosen=chosen,candidates=rows,selected=pick,rule='First four DEV target days choose family member; later four choose mixture via one-standard-error rule favoring alpha0/smaller alpha. No TEST arrays loaded.',repeat_seeds=[20261001,20261002],fusion_average='fixed equal two seed average',tf3_mixture='Replace only original 50% TimesFM3 component of fixed best ensemble before additional alpha shrinkage',residual_interval=dict(nominal_coverage=.8,calibrated_halfwidth_relative_to_past_mean=halfwidth(d,DEV,p,y,m),early4_to_late4_coverage=float(late_cover),early_halfwidth=early_c,guaranteed_coverage=False,dependency='same month repeated selection; serial and cell correlation'),independent_confirmation=False,test_seen_in_prior_rounds=True,ablation_test_epoch=10)
    fq=np.load(R/'results/fusion_repeat_quantiles_dev.npz')['equal_seed_mean'];fscores=(np.maximum(fq[:,:,:,0]-y,y-fq[:,:,:,2])/levels)[m];sel['fusion_interval']=dict(widening_relative_to_past_mean=max(0,float(np.quantile(fscores,.8,method='higher'))),nominal=.8,guaranteed_coverage=False)
    save_json(R/'results/selection.json',sel);(R/'results/selection.sha256').write_text(hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest());np.savez_compressed(R/'results/dev_selected.npz',**pred);pd.DataFrame(rows).to_csv(R/'results/blend_validation.csv',index=False);print(json.dumps(sel,indent=2),flush=True)
if __name__=='__main__':main()
