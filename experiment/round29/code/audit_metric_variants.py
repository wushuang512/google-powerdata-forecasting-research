from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
from docx import Document
T=Path(sys.argv[1]);B=T/'experiment';O=Path(sys.argv[2]);O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(B/'code'));import common
ref=T/'参考材料/9.28算力中心需求预测实验.docx'
doc=Document(ref);blocks=[''.join(e.itertext()) for e in []]
from docx.oxml.ns import qn
blocks=[{'index':i,'text':''.join(n.text or '' for n in e.iter(qn('w:t')))} for i,e in enumerate(doc.element.body)]
(O/'reference_extract.json').write_text(json.dumps(dict(path=str(ref),sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),blocks=blocks),ensure_ascii=False,indent=2),encoding='utf-8')
d=common.load();old=json.loads((B/'model_comparison_20261005/results/task_revision/summary.json').read_text(encoding='utf-8'))
orig=[22,23,24];y,m=common.truth(d,orig,7);yd=y.reshape(57,-1,96);md=m.reshape(57,-1,96);cnt=md.sum(-1);dv=cnt>0
ps={'mean3':common.baseline(d,orig,7,'mean3')}
for r in old['seven_day']:
 if any(k in r['model'] for k in ['tirex','equal','chronos','tf25','timesfm']):
  with np.load(B/r['source']) as a:ps[r['model']]=a[r['array']]
def dw(a,b,ok):return float(np.mean(100*(abs(a-b)*ok).sum(1)/(abs(a)*ok).sum(1)))
res=[]
for name,p in ps.items():
 pd=p.reshape(57,-1,96);ya=(yd*md).sum(-1)/cnt.clip(1);pa=(pd*md).sum(-1)/cnt.clip(1);yf=yd.mean(-1);pf=pd.mean(-1)
 a=dict(model=name,curve=common.metrics(y,p,m)['wape'],valid_slots=dw(ya,pa,dv),full96_both=dw(yf,pf,dv),valid_truth_full_prediction=dw(ya,pf,dv),full_truth_valid_prediction=dw(yf,pa,dv),complete_days_only=dw(ya,pa,cnt==96),daily_mape=float(np.mean(100*abs(ya-pa)[dv]/abs(ya[dv]))),pooled_daily=float(100*abs(ya-pa)[dv].sum()/abs(ya[dv]).sum()))
 for threshold in [93,94,95,96]:
  ok=cnt>=threshold;a[f'daycount{threshold}']=dw(ya,pa,ok)
 ok=cnt==96
 a['complete_day_origin_macro']=float(np.mean([dw(ya.reshape(57,3,7)[:,j],pa.reshape(57,3,7)[:,j],ok.reshape(57,3,7)[:,j]) for j in range(3)]))
 a['complete_day_pooled']=float(100*abs(ya-pa)[ok].sum()/abs(ya[ok]).sum())
 a['complete_day_mape']=float(np.mean(100*abs(ya-pa)[ok]/abs(ya[ok])))
 res.append(a)
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),valid_7_points=int(m.sum()),valid_24_points=int(common.truth(d,list(range(22,31)),1)[1].sum()),variants=res)
(O/'daily_variants.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
