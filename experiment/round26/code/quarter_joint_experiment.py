"""Predeclared conservative quarter-strength joint correction, with binary DEV admission."""
from pathlib import Path
import sys,ast,argparse,json
import numpy as np
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round26';Q=R/'results'
src=B/'round25/code/level_shape_experiment.py';e={'__file__':str(src),'__name__':'pure_previous_utilities'};tree=ast.parse(src.read_text(encoding='utf-8'));cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ap' for t in n.targets));exec(compile(ast.Module(body=tree.body[:cut],type_ignores=[]),str(src),'exec'),e)
read=e['read'];save=e['save'];sha=e['sha'];utc=e['utc'];z=e['z'];score=e['score'];guard=e['guard'];common=e['common'];interval_metrics=e['interval_metrics']
TOLERANCES=dict(wape_pp=.005,daily_wape_pp=.005,peak_mape_pp=.005,p95_ape_pp=.01,mae_relative=.002,rmse_relative=.002,hit2_percent_pp=.05,hit5_percent_pp=.05,group_wape_pp=.005,absolute_bias_pp=.005,peak_time_minutes=5,daily_mae_relative=.005,daily_rmse_relative=.005)
def admission_guard(s,b):
 return s['wape']<=b['wape']+.005 and s['daily_wape']<=b['daily_wape']+.005 and s['peak_mape']<=b['peak_mape']+.005 and s['p95_ape']<=b['p95_ape']+.01 and all(s[k]<=b[k]*1.002 for k in ['mae','rmse']) and all(s[k]>=b[k]-.05 for k in ['hit2_percent','hit5_percent']) and all(s[k]<=b[k]+.005 for k in ['cella5_wape','other52_wape','absolute_bias_percent']) and s['peak_time_mae_minutes']<=b['peak_time_mae_minutes']+5 and all(s[k]<=b[k]*1.005 for k in ['daily_mae','daily_rmse'])
def improvements(s,b):
 categories=[]
 if s['wape']<b['wape']-.0001:categories.append('curve')
 if s['daily_wape']<b['daily_wape']-.0001:categories.append('daily_mean')
 if s['peak_mape']<b['peak_mape']-.0001:categories.append('peak_amplitude')
 if s['p95_ape']<b['p95_ape']-.0001 or any(s[k]>b[k]+.005 for k in ['hit2_percent','hit5_percent']):categories.append('point_error_distribution')
 return categories
def prepare():
 for k in ['code','docs','models','results']:(R/k).mkdir(parents=True,exist_ok=True)
 if (R/'docs/protocol.json').exists():assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];return
 files=['results/data.npz','round25/code/level_shape_experiment.py','round25/results/dev_predictions.npz','round25/results/test_predictions.npz','round25/results/selection.json','round24/results/dev_predictions.npz','round24/results/test_predictions.npz','round19/results/test_predictions.npz','round9/results/selected_intervals_test.npz']
 save(R/'docs/protocol.json',dict(created_utc=utc(),script_sha256=sha(__file__),metrics_script_sha256=sha(R/'code/source_metrics.py'),method='R19 recommended curve plus fixed25% of the DEV-frozen R25 joint correction. No TEST sweep of interpolation weights.',fixed_strength=.25,choice_reason='Conservative trust-region hypothesis to limit cumulative point/peak/level adjustments after several adaptive stages with only two overlapping DEV origins. Quarter strength is a prespecified engineering choice, not a proven optimal weight. No strength grid is scored on TEST.',user_preference='2026-10-06 human explicitly accepts multiple improvements with very small remaining gaps as recommended best; not strict first in every metric. This prospective tolerance policy does not rewrite prior frozen trials.',tolerances=TOLERANCES,selection='Binary DEV admission only: both origins14/15 and aggregate must pass registered small-tradeoff limits versus R19; aggregate at least two of four distinct categories curve/daily/peak/point-distribution improve over.0001pp or hit+.005pp. If passed use fixed.25, else0. Diagnostic always fixed.25 frozen before TEST. No candidate or strength reordering after TEST.',independent_confirmation=False,test_seen_before=True,source_hashes={f:sha(B/f) for f in files},resource_policy=dict(admission_physical_gib=2.125,admission_commit_gib=6,job_commit_hard_cap_bytes=512*2**20,runtime_physical_floor_gib=1.5,runtime_commit_floor_gib=2)))
 (R/'docs/protocol.sha256').write_text(sha(R/'docs/protocol.json'),encoding='ascii')
def develop(d):
 assert not (Q/'selection.json').exists();base=z('round24/results/dev_predictions.npz','prior7');proposal=z('round25/results/dev_predictions.npz','selected7');cp=.75*base+.25*proposal;rows=[];ok=True
 for origins,sl,label in [([14],slice(0,1),'early'),([15],slice(1,2),'late'),([14,15],slice(None),'aggregate')]:
  b=score(d,origins,base[:,sl],7);c=score(d,origins,cp[:,sl],7);passed=admission_guard(c,b);rows.append(dict(split=label,prior=b,candidate=c,guard_passed=passed,improved_categories=improvements(c,b)));ok &= passed
 ok &= len(rows[-1]['improved_categories'])>=2;a=.25 if ok else 0.;sel=(1-a)*base+a*proposal;prior1=z('round25/results/dev_predictions.npz','selected1');np.savez_compressed(Q/'dev_predictions.npz',prior7=base,proposal7=proposal,candidate7=cp,selected7=sel,selected1=prior1);save(Q/'dev_summary.json',dict(splits=rows,testing_not_loaded=True,fixed_strength=.25,development_supported=bool(ok)))
 y,m=common.truth(d,[14,15],7);rad=np.array([np.quantile(abs(sel[i]-y[i])[m[i]],.8) for i in range(57)]);np.savez_compressed(R/'models/interval7_radius.npz',radius=rad)
 s=dict(frozen_utc=utc(),horizons={'1':dict(chosen_alpha=0.,development_supported=False),'7':dict(candidate={'id':'fixed_quarter_joint'},chosen_alpha=a,diagnostic_alpha=.25,development_supported=bool(ok))},protocol_sha256=sha(R/'docs/protocol.json'),dev_summary_sha256=sha(Q/'dev_summary.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),interval_radius_sha256=sha(R/'models/interval7_radius.npz'),independent_confirmation=False);save(Q/'selection.json',s);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');print('SELECTION FROZEN',json.dumps(s),flush=True)
def test(d):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();base=z('round19/results/test_predictions.npz','selected7');proposal=z('round25/results/test_predictions.npz','selected7');diag=.75*base+.25*proposal;a=s['horizons']['7']['chosen_alpha'];p=(1-a)*base+a*proposal;arrays=dict(prior7=base,proposal7=proposal,candidate7=diag,diagnostic7=diag,selected7=p)
 for k in ['prior1','candidate1','selected1','diagnostic1','lower1','upper1']:arrays[k]=z('round25/results/test_predictions.npz',k)
 rad=z('round26/models/interval7_radius.npz','radius')[:,None,None];lo=np.clip(p-rad,0,1);hi=np.clip(p+rad,0,1);arrays['lower7']=lo;arrays['upper7']=hi;scores={'1':read(B/'round25/results/summary.json')['scores']['1'],'7':{k:score(d,[22,23,24],v,7) for k,v in dict(prior=base,candidate=diag,selected=p).items()}};y,m=common.truth(d,[22,23,24],7);scores['7']['interval_selected']=interval_metrics(y,p,m,lo,hi);np.savez_compressed(Q/'test_predictions.npz',**arrays);save(Q/'summary.json',dict(selection=s,scores=scores,no_post_test_selection=True,independent_confirmation=False));print('ACTUAL FIXED QUARTER TEST',json.dumps(scores['7']),flush=True)
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='prepare');a=ap.parse_args()
if a.stage=='prepare':prepare()
else:
 assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];d=common.load();{'develop':develop,'test':test}[a.stage](d)

