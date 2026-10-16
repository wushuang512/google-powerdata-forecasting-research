"""Numerical future poisoning of the actual inherited Ridge feature function."""
from pathlib import Path
import argparse,ast,pickle
import numpy as np
from core import *
from models import *
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root);F,cfg,d,q=load(R);reg=read(F/'docs/historical_controls.json')
end=q['split']['validation'][1];known=d['y'][:end];scale=float(np.quantile(known[np.isfinite(known)],.95)/.8)
with Path(reg['ridge_file']).open('rb') as f:ridge=pickle.load(f)
fields=read(R/'experiment/round5/docs/calibration_features.json')['features']
tree=ast.parse((F/'code/historical_assemble.py').read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calibration')
scope=dict(np=np,d=d,scale=scale,feature_names=fields,ridge=ridge);exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_calibration_function','exec'),scope)
oo=origins(q,'test',1);p=(.25*read_forecasts(F,q,'test',1,'source_chronos2_original')+.25*read_forecasts(F,q,'test',1,'source_tirex2_original')+.5*read_forecasts(F,q,'test',1,'timesfm3'))/scale
banks={o:p[i] for i,o in enumerate(oo)};checks=[]
for i in [0,len(oo)//2,len(oo)-1]:
    o=oo[i];real=scope['calibration'](o,p[i],banks)
    poison={k:v.copy() for k,v in d.items()};poison['y'][o:]=1e10;poison['covariates'][o:]=-1e10;scope['d']=poison
    mutated=scope['calibration'](o,p[i],banks);scope['d']=d
    equal=bool(np.array_equal(real,mutated));assert equal
    checks.append(dict(origin_day=o,date=str(d['dates'][o]),future_actual_targets_weather_unchanged_output=equal,max_absolute_difference=float(abs(real-mutated).max())))
save(F/'results/transfer_future_poison_verification.json',dict(all_passed=True,utc=utc(),checks=checks,actual_function_sha256=sha(F/'code/historical_assemble.py'),ridge_sha256=sha(Path(reg['ridge_file'])),source_histories_are_prior14days=True,scope='Actual calibration function and original Ridge model; source neural histories bounded by separate input/implementation checks; this is an implementation check, not proof against all overfitting'))
print('Actual inherited calibration future-poison tests passed',len(checks))
