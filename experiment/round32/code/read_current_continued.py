from pathlib import Path
import sys,json,hashlib
T=Path(sys.argv[1]);H=int(sys.argv[2]);v=json.loads((T/'experiment/round32/results/current_entry_verification.json').read_text(encoding='utf-8'));assert v['all_passed'];r=next(r for r in v['entries'] if r['horizon_days']==H)
for field in ['curve','daily']:assert hashlib.sha256(Path(r[field]).read_bytes()).hexdigest()==r[field+'_sha256']
print(json.dumps(r,ensure_ascii=False))
