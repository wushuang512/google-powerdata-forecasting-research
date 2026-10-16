import sys,json,argparse,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'round3/code'))
from timesfm3_run import forecast,CONFIGS,TimesFM3Forecaster
from lib import load,truth,metrics,np
import pandas as pd
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--days',type=int,choices=[1,7],default=1);ap.add_argument('--origin',type=int,default=31);ap.add_argument('--backtest7',action='store_true');a=ap.parse_args();d=load()
    if a.backtest7:
        origins=[22,23,24];days=7;z=np.load(B/'round2/results/test7_predictions.npz');old=z['old_equal_mix'];tirex=z['tirex_uni'];prefix='test7'
    else:
        origins=[a.origin];days=a.days
        subprocess.run([sys.executable,'-u','-X','utf8',str(B/'round2/code/predict_legacy.py'),'--origin',str(a.origin),'--days',str(days)],check=True)
        old=pd.read_csv(B/f'round2/results/legacy_deployment_origin{a.origin}_{days}day.csv').predicted_util.to_numpy().reshape(57,1,days*96);prefix=f'deployment_origin{a.origin}_{days}day'
    model=TimesFM3Forecaster.from_pretrained(str(B/'models/timesfm3-base'),device='cuda',per_core_batch_size=4,local_files_only=True)
    f=forecast(model,d,origins,CONFIGS['tf3_cellprod'],h=days*96);p=np.clip(.5*old+.5*f,0,1)
    np.savez_compressed(R/f'results/{prefix}.npz',best_mix=p,old_equal_mix=old,tf3_cellprod=f)
    if a.backtest7:
        y,m=truth(d,origins,7);scores={k:metrics(y,z,m) for k,z in [('best_mix',p),('old_equal_mix',old),('tf3_cellprod',f),('tirex_uni',tirex)]}
        (R/'results/seven_day_summary.json').write_text(json.dumps(dict(scores=scores,weights_transferred_unchanged=True,overlapping_windows=True,independent_confirmation=False),indent=2));print(json.dumps(scores,indent=2),flush=True)
    rows=[]
    for i,n in enumerate(d['names']):
        for j,o in enumerate(origins):
            for t in range(days*96):
                row=dict(series=n,cell=d['cells'][i],origin_end_day=o,local_trace_time_PT=str(pd.Timestamp('2019-05-01')+pd.Timedelta(days=o,minutes=t*15)),predicted_util=float(p[i,j,t]))
                if a.backtest7:row.update(actual_util=float(y[i,j,t]),valid_target=bool(m[i,j,t]))
                rows.append(row)
    path=R/f'results/{prefix}.csv';pd.DataFrame(rows).to_csv(path,index=False);print('Saved',path,flush=True)
if __name__=='__main__':main()
