from pathlib import Path
import sys,json,datetime,hashlib
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round38'
for name in ['code','docs','models','results']:(R/name).mkdir(parents=True,exist_ok=True)
old=json.loads((B/'round37/docs/protocol.json').read_text(encoding='utf-8'));adapters=[dict(id=f'{mode}_{v["id"]}',mode=mode,rank=v['rank'],teacher_lambda=v['teacher_lambda'],lr=v['lr']) for mode in ['level','seasonal'] for v in old['adapters']];families=[dict(id=f'{mode}_{group}_{v}',rate=1,mode=mode,group=group,context_days=7 if group=='uni' else 14,variant=v) for mode in ['level','seasonal'] for group in ['uni','cell'] for v in ['base']+[a['id'] for a in adapters if a['mode']==mode]]
proto=dict(old,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model='Native15min Moirai2 boundedlowrankquantile adaptationandorigin-conditionedseasonaldeviationforecast',rates=[1],families=families,adapters=adapters,representation='level orseasonal: subtract lastthreefullypast96point daysmeanprofile fromallhistoricalinputandtraininglabels; forecastdeviation andadd same knownoriginprofile, nofuturetruth. Full15min information retained.',reason='R37 early/lateDEV didnot supportcoarse modelreplacement. This continuation registered before readingR37 newTESTscores. Coarse results retained, noTEST-basedfamilyselection.',train_norm='Pastactualpowermean regardlessofrepresentation, notnearzero residualmean; scalercontextpast-only',round37_test_scores_read_before_protocol=False,source_weights_sha256=hashlib.sha256((B/'round35/models/moirai2/model.safetensors').read_bytes()).hexdigest())
assert not (R/'docs/protocol.json').exists();(R/'docs/protocol.json').write_text(json.dumps(proto,ensure_ascii=False,indent=2),encoding='utf-8');s=(B/'round37/code/multires_moirai.py').read_text(encoding='utf-8').replace("R=B/'round37'","R=B/'round38'")
oldhist="""def hist(t,rate,days=7):
 x=flat[:,t-days*96:t].reshape(57,-1,rate).mean(-1);m=valid[:,t-days*96:t].reshape(57,-1,rate).all(-1);assert x.shape[-1]==days*96//rate;return x,m
"""
newhist="""def profile(t):return flat[:,t-3*96:t].reshape(57,3,96).mean(1)
def hist(t,rate,days=7,mode='level'):
 x=flat[:,t-days*96:t].reshape(57,-1,rate).mean(-1);m=valid[:,t-days*96:t].reshape(57,-1,rate).all(-1)
 if mode=='seasonal':x=x-np.tile(profile(t).reshape(57,-1,rate).mean(-1),(1,days))
 assert x.shape[-1]==days*96//rate;return x,m
"""
assert oldhist in s;s=s.replace(oldhist,newhist).replace('def features(cut,rate):','def features(cut,rate,mode):').replace("path=Q/f'features_rate{rate}_cut{cut}.npz'","path=Q/f'features_{mode}_rate{rate}_cut{cut}.npz'")
s=s.replace("+sha(R/'docs/protocol.json')).encode()", "+sha(R/'docs/protocol.json')+json.dumps({'rate':rate,'mode':mode,'cut':cut,'torch':torch.__version__})).encode()")
s=s.replace("'truth','mask','pdu','time']","'truth','mask','pdu','time','norm']").replace('hx,hm=hist(t,rate);remaining=',"hx,hm=hist(t,rate,mode=mode);norm=(flat[:,t-7*96:t]*valid[:,t-7*96:t]).sum(-1)/valid[:,t-7*96:t].sum(-1).clip(1);remaining=")
needle="   for ix in groups:\n";replacement="   if mode=='seasonal':truth[:,:remaining]-=np.tile(profile(t),(1,4))[:,:remaining]\n"+needle;assert needle in s;s=s.replace(needle,replacement,1)
s=s.replace("bank['x'].append(cap['x'].reshape(-1,384));", "bank['norm'].append(norm[ix].astype(np.float32));bank['x'].append(cap['x'].reshape(-1,384));")
s=s.replace("f'feature_audit_rate{rate}_cut{cut}.json'", "f'feature_audit_{mode}_rate{rate}_cut{cut}.json'").replace('def train(cut,rate,variants):','def train(cut,rate,mode,variants):').replace('data=features(cut,rate);','data=features(cut,rate,mode);').replace('norm=loc.abs().clamp_min(.05)','norm=torch.tensor(data["norm"],device=DEV).abs().clamp_min(.05)').replace("Q/f'features_rate{rate}_cut{cut}.npz'", "Q/f'features_{mode}_rate{rate}_cut{cut}.npz'")
s=s.replace("hx,hm=hist(o*96,rate,days);", "hx,hm=hist(o*96,rate,days,mode=family['mode']);")
s=s.replace("assert np.isfinite(out).all();return np.clip(out,0,1)", "if family['mode']=='seasonal':out+=np.tile(profile(o*96),(1,H))[:,None,:]\n assert np.isfinite(out).all();return np.clip(out,0,1)")
s=s.replace(" for rate in [4,12]:train(14,rate,[v['id'] for v in proto['adapters']])", " for mode in ['level','seasonal']:train(14,1,mode,[v['id'] for v in proto['adapters'] if v['mode']==mode])")
s=s.replace("needed.setdefault(f['rate'],set()).add(f['variant'])", "needed.setdefault((f['rate'],f['mode']),set()).add(f['variant'])").replace(" for rate,names in needed.items():train(22,rate,sorted(names))", " for (rate,mode),names in needed.items():train(22,rate,mode,sorted(names))")
# Everyuntouched level/base forecast must reproduce original native controlbefore DEV selection.
needle="def develop():\n bank={};scores={};sel={}"
replace="""def develop():
 checks=[]
 with np.load(Q/'native_dev.npz') as new,np.load(SRC/'results/native_dev.npz') as old:
  for H in [1,7]:
   for group,family in [('uni','uni7'),('cell','cell14')]:
    for o in origins('dev',H):
     a=new[f'level_{group}_base_H{H}_o{o}'];b=old[f'{family}_H{H}_o{o}'];difference=float(np.max(abs(a-b)));assert difference<1e-6,'Nativecontrol regression';checks.append(dict(H=H,origin=o,group=group,maxdifference=difference))
 save('native_control_regression.json',dict(all_passed=True,checks=checks,scope='same officialweights andhistorical inputs, noTEST used'))
 bank={};scores={};sel={}"""
assert needle in s;s=s.replace(needle,replace).replace('code/multires_moirai.py','code/native_residual_moirai.py');compile(s,'native_residual_moirai.py','exec');(R/'code/native_residual_moirai.py').write_text(s,encoding='utf-8')
g=(B/'round37/code/run_multires_guarded.py').read_text(encoding='utf-8').replace("R=T/'experiment/round37'","R=T/'experiment/round38'").replace("code/multires_moirai.py","code/native_residual_moirai.py");(R/'code/run_native_residual_guarded.py').write_text(g,encoding='utf-8');print('R38 preregistered native15min level/seasonal variants withfew-thousandtrainableparameters, sourcecontrols regression required')
