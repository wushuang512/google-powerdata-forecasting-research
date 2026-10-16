from pathlib import Path
import sys,json,datetime,hashlib
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round39'
for name in ['code','docs','models','results']:(R/name).mkdir(parents=True,exist_ok=True)
old=json.loads((B/'round37/docs/protocol.json').read_text(encoding='utf-8'));families=[dict(id='global57_3hour',rate=12,group='global',context_days=14,variant='base'),dict(id='corr8_native15min',rate=1,group='corr',context_days=14,variant='base'),dict(id='corr8_3hour',rate=12,group='corr',context_days=14,variant='base')]
proto=dict(old,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model='Moirai2 training-onlydynamiccorrelationgroups andglobal57coarsejointforecast',families=families,adapters=[],rates=[1,12],grouping='Fixedalgorithm: hourmeans beforecut;averagelevel/hourdifference correlations. Greedyhighestpositive-degree seed plus7mostsimilarremaining domains, group<=8. Cut14DEV,cut22TEST refitunsupervisedtrain-only. Global57uses3hbins toavoid large15min autoregression workspace.',reason='R37/38 DEV didnot showstable benefitfromlowrankheadlearning. Trynewpeerinformation scope usingonly historicalpower; noTEST rank used tochoose groupalgorithm.',train_dev_label_end_exclusive=1344,train_final_label_end_exclusive=2112,trained_weights='none,officialbackbone frozen,unsupervisedgroups beforeforecastcut',independent_confirmation=False)
assert not (R/'docs/protocol.json').exists();(R/'docs/protocol.json').write_text(json.dumps(proto,ensure_ascii=False,indent=2),encoding='utf-8');s=(B/'round37/code/multires_moirai.py').read_text(encoding='utf-8').replace("R=B/'round37'","R=B/'round39'").replace('code/multires_moirai.py','code/correlation_moirai.py')
insertion="""def correlation_groups(cut):
 power=flat[:,:cut*96].reshape(57,cut*24,4).mean(-1);difference=np.diff(power,axis=1)
 def corr(x):
  x=x-x.mean(-1,keepdims=True);x=x/x.std(-1,keepdims=True).clip(1e-8);return np.clip(x@x.T/x.shape[1],-1,1)
 c=.5*corr(power)+.5*corr(difference);left=list(range(57));groups=[]
 while left:
  degree=np.maximum(c[np.ix_(left,left)],0).sum(1);seed=left[int(np.argmax(degree))];others=sorted([i for i in left if i!=seed],key=lambda i:(-c[seed,i],i));ix=np.array([seed]+others[:7]);groups.append(ix);left=[i for i in left if i not in ix]
 assert sorted(np.concatenate(groups).tolist())==list(range(57)) and max(map(len,groups))<=8
 save(f'correlation_groups_cut{cut}.json',dict(data_end_exclusive=cut*96,source_prefix_sha256=hashlib.sha256(np.ascontiguousarray(flat[:,:cut*96]).tobytes()).hexdigest(),groups=[x.tolist() for x in groups],algorithm='Halfhourlylevels plushalf theirfirstdifferences Pearson; greedy degree seed/top7; nofuture/testlabels',source='All57PowerData domains, notall8ClusterData cells'));return groups
"""
s=s.replace('def infer_one(family,H,o,cut):',insertion+'def infer_one(family,H,o,cut):')
line=next(v for v in s.splitlines() if v.startswith(" rate=family['rate'];days="))
newline=line[:line.index(';groups=')]+";groups=[np.arange(57)] if family['group']=='global' else correlation_groups(cut);m=None;last=None";s=s.replace(line,newline)
s=s.replace("dim=1 if family['group']=='uni' else len(ix)","dim=len(ix)")
compile(s,'correlation_moirai.py','exec');(R/'code/correlation_moirai.py').write_text(s,encoding='utf-8')
g=(B/'round37/code/run_multires_guarded.py').read_text(encoding='utf-8').replace("R=T/'experiment/round37'","R=T/'experiment/round39'").replace("['smoke','train_dev','native_dev','develop','train_final','native_test','test','native_deploy','deploy']","['smoke','native_dev','develop','native_test','test','native_deploy','deploy']").replace('code/multires_moirai.py','code/correlation_moirai.py');(R/'code/run_correlation_guarded.py').write_text(g,encoding='utf-8');print('R39 registered global57coarse andbalancedcorrelationpeer forecasters, no download orgradienttraining')
