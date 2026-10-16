from pathlib import Path
import sys,json,csv,hashlib
T=Path(sys.argv[1]);H=int(sys.argv[2]);assert H in [1,7];R=T/'experiment/round30';v=json.loads((R/'results/new_entry_verification.json').read_text(encoding='utf-8'));assert v['all_passed'];rows=[r for r in v['files'] if r['days']==H];assert len(rows)==2
for r in rows:
 p=Path(r['curve']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['curve_sha256'];assert hashlib.sha256(Path(r['daily']).read_bytes()).hexdigest()==r['daily_sha256']
print(json.dumps(dict(research_only=True,current_recommendation=False,historical_replay=True,no_june_truth=True,methods=rows,scope='IBM TinyTimeMixer R2 512 history; decoder/head or all-parameter multitask training, no TEST choice'),ensure_ascii=False))
