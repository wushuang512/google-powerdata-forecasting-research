from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
T=Path(sys.argv[1]);H=int(sys.argv[2]);v=json.loads((T/'experiment/round36/results/current_entry_verification.json').read_text(encoding='utf-8'));assert v['all_passed'];r=next(r for r in v['entries'] if r['horizon_days']==H)
if len(sys.argv)>3 and sys.argv[3]=='research':
 assert H==1;r=dict(method='DEV-frozen Moirai2 uni7 50%+prior50%, historicalpeak25%; TESTcurveworse, notdeployed',curve=str(T/'experiment/round36/results/exports/diagnostic_1day_curve.csv'),daily=str(T/'experiment/round36/results/exports/diagnostic_1day_daily_mean.csv'),horizon_days=1,historical_replay=True,current_live_forecast=False)
else:
 for k in ['curve','daily']:assert hashlib.sha256(Path(r[k]).read_bytes()).hexdigest()==r[k+'_sha256']
print(json.dumps(r,ensure_ascii=False))
