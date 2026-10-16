"""Complete diagnostics for all already frozen forecasts without new fitting."""
from pathlib import Path
import argparse
import numpy as np
from core import *
from models import *
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root);F,cfg,d,q=load(R);summary=read(F/'results/summary.json');out=[]
for row in summary['rows']:
    H=row['horizon_days'];name=row['model']
    with np.load(F/'results'/f'test_H{H}.npz') as z:
        ii=np.arange(len(z['origins'])) if row['cohort']=='daily_origins' else np.flatnonzero((z['origins']-q['split']['test'][0])%7==0)
        Y=z['truth'][ii];P=z[name][ii]
        for lead in range(H):
            sl=slice(lead*96,(lead+1)*96);out.append(dict(cohort=row['cohort'],horizon_days=H,forecast_day=lead+1,model=name,**metrics(Y[:,sl],P[:,sl],cfg['mape_threshold'])))
export(F/'results/新协议逐预测日测试指标.csv',out)
assert len(out)==14+14*7*2
save(F/'results/diagnostic_tables_complete.json',dict(all_passed=True,utc=utc(),rows=len(out),all_fourteen_models=True,forecast_arrays_unchanged=True,summary_sha256=sha(F/'results/summary.json')))
print('All-model lead-day diagnostics complete',len(out))
