from pathlib import Path
import argparse,json,hashlib
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--round',type=int,choices=[27,28],required=True);ap.add_argument('--days',type=int,choices=[1,7],default=7);a=ap.parse_args();T=Path(a.root).resolve();Q=T/f'experiment/round{a.round}/results';ex=json.loads((Q/'exports/verification.json').read_text(encoding='utf-8'));s=json.loads((Q/'selection.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert ex['all_passed'] and ex['selection_sha256']==sha(Q/'selection.json') and sha(ex[f'file{a.days}'])==ex[f'sha{a.days}'] and sha(ex[f'daily_file{a.days}'])==ex[f'daily_sha{a.days}']
print(json.dumps(dict(round=a.round,curve_file=ex[f'file{a.days}'],daily_file=ex[f'daily_file{a.days}'],frozen_alpha=s['horizons'][str(a.days)]['chosen_alpha'],research_alternative=True,current_comprehensive_recommendation=False,historical_replay=True,no_june_truth=True),ensure_ascii=False))
