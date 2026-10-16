from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
T=Path(sys.argv[1]);nr=int(sys.argv[2]);H=int(sys.argv[3]);v=json.loads((T/'experiment/round40/results/research_entry_verification.json').read_text(encoding='utf-8'));assert v['all_passed'];r=next(r for r in v['entries'] if r['round']==nr and r['horizon_days']==H)
for k in ['curve','daily']:assert hashlib.sha256(Path(r[k]).read_bytes()).hexdigest()==r[k+'_sha256']
print(json.dumps(r,ensure_ascii=False));print('研究候选的2019年历史复现，未升级当前推荐，不是今天实时预测。')
