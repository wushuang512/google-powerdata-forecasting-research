from pathlib import Path
import sys,json,datetime,hashlib
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round40';DATA=T/'google-clusterdata-2019';U=DATA/'aggregated/complete_usage_a_15min.npz'
for name in ['code','docs','models','results']:(R/name).mkdir(parents=True,exist_ok=True)
old=json.loads((B/'round39/docs/protocol.json').read_text(encoding='utf-8'));families=[dict(id='power57_control_3hour',rate=12,group='global',context_days=14,variant='base',usage=False),dict(id='power57_cpu_mem_a_3hour',rate=12,group='global',context_days=14,variant='base',usage=True)]
proto=dict(old,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model='Moirai2 global57 powerforecast plusobserved cell-a CPU/memory auxiliaries, matchednocovariatecontrol',families=families,adapters=[],rates=[12],source_usage_sha256=hashlib.sha256(U.read_bytes()).hexdigest(),source_projection_audit_sha256=hashlib.sha256((DATA/'results/complete_projection_audit.json').read_bytes()).hexdigest(),source_scope='Officialcella12085files necessarycolumns; auxiliaryCPU/memoryonlyfivea domains. Deduplicateidenticalchannels usingonlytrainingprefix, preservingdistinctCPU/memoryunits. NeverfakeCPU forother52.',future='Jointfutureauxiliarychannels aremodelgenerated, notrealfutureCPU. Nativecoarse7d56steps<=4patches so directsinglepass, history14days. Onlypoweroutputscored.',grouping='Global57powers+historicalaux, all originalweights frozen',reason='Explicitmatchedtest ofactualadditionalhistoricalinformation; no extra datareadordownload, all sourcealignedandaudited.')
assert not (R/'docs/protocol.json').exists();(R/'docs/protocol.json').write_text(json.dumps(proto,ensure_ascii=False,indent=2),encoding='utf-8');s=(B/'round39/code/correlation_moirai.py').read_text(encoding='utf-8').replace("R=B/'round39'","R=B/'round40'").replace('code/correlation_moirai.py','code/usage_moirai.py')
start=s.index('def infer_one(');end=s.index('\ndef infer(phase):',start)
fn="""def usage_channels(cut):
 p=T/'google-clusterdata-2019/aggregated/complete_usage_a_15min.npz';audit=read(T/'google-clusterdata-2019/results/complete_projection_audit.json');assert audit['all_passed']
 with np.load(p) as z:
  assert bool(z['complete']) and int(z['source_files'])==int(z['expected_source_files'])==12085;u=z['aggregate'].reshape(57,-1,6);names=z['names'];fields=z['features'].tolist();assert np.array_equal(names,d['names'])
 ix=np.where(d['cells']=='a')[0];assert len(ix)==5;present=u[...,3]>0;assert not present[d['cells']!='a'].any();channels=[];masks=[];labels=[];seen=set()
 for i in ix:
  for j,unit in [(0,'CPU_NCU'),(1,'memory_normalized')]:
   v=u[i,:,j];m=present[i]&np.isfinite(v)&(v>=0);v=np.where(m,v,0);key=hashlib.sha256(unit.encode()+np.ascontiguousarray(v[:cut*96]).tobytes()+np.ascontiguousarray(m[:cut*96]).tobytes()).hexdigest()
   if key in seen:continue
   assert m[:cut*96].any();seen.add(key);channels.append(v);masks.append(m);labels.append(dict(source_domain=str(names[i]),source_feature=fields[j],unit=unit))
 save(f'usage_source_cut{cut}.json',dict(source_files=12085,complete=True,source_sha256=sha(p),training_dedup_end_exclusive=cut*96,used_channels=labels,only_a5_realCPU=True,no_future_input=True));return np.stack(channels),np.stack(masks)
def infer_one(family,H,o,cut):
 rate=12;days=14;hx,hm=hist(o*96,rate,days)
 if family['usage']:
  u,mask=usage_channels(cut);a=u[:,(o-days)*96:o*96].reshape(len(u),-1,rate).mean(-1);v=mask[:,(o-days)*96:o*96].reshape(len(u),-1,rate).all(-1);hx=np.concatenate([hx,a]);hm=np.concatenate([hm,v])
 dim=len(hx);m=model(days*96//rate,H*96//rate,dim,rate,'base',cut);x=torch.tensor(hx.T[None],dtype=torch.float32,device=DEV);obs=torch.tensor(hm.T[None],device=DEV)
 with torch.inference_mode():p=m(x,obs,torch.zeros(x.shape[:2],dtype=torch.bool,device=DEV)).cpu().numpy()
 assert p.shape==(1,9,H*96//rate,dim) and np.isfinite(p).all();out=p[0].transpose(2,0,1)[:57];return np.clip(out,0,1)
"""
s=s[:start]+fn+s[end:]
# Bind completeusage evenin controlcaches, so sourcechanges cannotbesilent.
s=s.replace("sig=hashlib.sha256((sha(R/'docs/protocol.json')", "sig=hashlib.sha256((sha(T/'google-clusterdata-2019/aggregated/complete_usage_a_15min.npz')+sha(R/'docs/protocol.json')")
needle="def develop():\n bank={};scores={};sel={}"
replace="""def develop():
 differences=[]
 with np.load(Q/'native_dev.npz') as new,np.load(B/'round39/results/native_dev.npz') as old:
  for H in [1,7]:
   for o in origins('dev',H):
    dif=float(np.max(abs(new[f'power57_control_3hour_H{H}_o{o}']-old[f'global57_3hour_H{H}_o{o}'])));assert dif<1e-6;differences.append(dict(H=H,origin=o,max_difference=dif))
 save('matched_control_regression.json',dict(all_passed=True,checks=differences))
 bank={};scores={};sel={}"""
assert needle in s;s=s.replace(needle,replace);compile(s,'usage_moirai.py','exec');(R/'code/usage_moirai.py').write_text(s,encoding='utf-8')
g=(B/'round39/code/run_correlation_guarded.py').read_text(encoding='utf-8').replace("R=T/'experiment/round39'","R=T/'experiment/round40'").replace('code/correlation_moirai.py','code/usage_moirai.py');(R/'code/run_usage_guarded.py').write_text(g,encoding='utf-8');print('R40 registered realhistoryCPU/memory matchedcontrol; onlycella auxiliaries, nofuture truthandno download')
