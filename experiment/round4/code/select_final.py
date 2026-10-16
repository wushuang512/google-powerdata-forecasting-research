import sys,json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(R/'code'))
from aligned import *
def main():
    d=load();y,m=truth(d,DEV);r3=B/'round3/results'
    old=(np.load(r3/'chronos_dev.npz')['native']+np.load(r3/'tirex_dev.npz')['native'])/2
    anchor=(old+np.load(r3/'tf3_dev.npz')['tf3_cellprod'])/2
    families={'aligned':dict(np.load(R/'results/development.npz')),'tf25':dict(np.load(R/'results/tf25_dev.npz'))}
    chosen={f:min(ps,key=lambda k:metrics(y[:,:4],ps[k][:,:4],m[:,:4])['wape']) for f,ps in families.items()}
    rows=[];preds={'anchor':anchor};cells=np.unique(d['cells'])
    def domain_score(p):return 100*(np.abs(y[:,4:]-p[:,4:])*m[:,4:]).sum((1,2))/(np.abs(y[:,4:])*m[:,4:]).sum((1,2))
    base=domain_score(anchor)
    for family in ['anchor',*families]:
        for alpha in ([0] if family=='anchor' else [.1,.25,.5,1]):
            name='anchor' if family=='anchor' else chosen[family];p=anchor if alpha==0 else (1-alpha)*anchor+alpha*families[family][name];key=f'{name}_a{alpha}';preds[key]=p
            loss=domain_score(p);delta=base-loss;bycell=np.array([delta[d['cells']==c].mean() for c in cells]);se=bycell.std(ddof=1)/np.sqrt(len(cells))
            rows.append(dict(key=key,family=family,model=name,alpha=alpha,wape=float(loss.mean()),improvement_pp=float(delta.mean()),paired_cell_se=float(se)))
    best=min(rows,key=lambda x:x['wape']);eligible=[x for x in rows if x['wape']<=best['wape']+best['paired_cell_se']];pick=min(eligible,key=lambda x:(x['alpha'],x['wape']))
    selection=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),chosen_on_first4=chosen,candidates=rows,selected=pick,needed=[chosen['tf25']],anchor='R3 secondary fixed .25 Chronos2 uni + .25 TiRex2 uni + .50 TF3 cellprod',selection_rule='First four DEV days select family member; last four DEV days choose alpha via paired-cell one-standard-error rule favoring zero/smaller alpha. No TEST labels loaded.',confirmation=False,prior_test_already_seen=True)
    txt=json.dumps(selection,indent=2);(R/'results/selection.json').write_text(txt,encoding='utf-8');(R/'results/selection.sha256').write_text(hashlib.sha256(txt.encode()).hexdigest())
    pd.DataFrame(rows).to_csv(R/'results/blend_validation.csv',index=False);print(json.dumps(selection,indent=2),flush=True)
if __name__=='__main__':main()
