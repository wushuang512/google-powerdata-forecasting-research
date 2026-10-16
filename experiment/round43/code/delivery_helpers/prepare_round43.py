from pathlib import Path
import json,sys,hashlib,datetime
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round43';assert not R.exists()
for p in ['code','docs','models','results/native_cache']:(R/p).mkdir(parents=True,exist_ok=True)
s=(B/'round41/code/joint_factor_moirai.py').read_text(encoding='utf-8')
start=s.index('def history(');end=s.index('def set_scope(',start)
s=s[:start]+'''def history(t,days,rate,k,basis):
 y=flat[:,t-days*96:t];m=valid[:,t-days*96:t];x=np.where(m,(y-basis['mean'][:,None])/basis['scale'][:,None],0);f=basis['vectors'][:k]@x
 if CURRENT_MODE=='daily_increment':
  yp=flat[:,t-(days+1)*96:t-96];mp=valid[:,t-(days+1)*96:t-96];xp=np.where(mp,(yp-basis['mean'][:,None])/basis['scale'][:,None],0);f=f-basis['vectors'][:k]@xp;m=m&mp
 obs=np.broadcast_to((m.mean(0)>=.95)[None],f.shape).copy();return f.reshape(k,-1,rate).mean(-1),obs.reshape(k,-1,rate).all(-1)
CURRENT_MODE='level'
'''+s[end:]
s=s.replace("def infer_one(f,H,o,cut):\n basis=", "def infer_one(f,H,o,cut):\n global CURRENT_MODE\n CURRENT_MODE=f['mode']\n basis=")
old="out=p[0][None] if k==1 else p[0].transpose(2,0,1);assert out.shape==(k,9,H*96//rate) and np.isfinite(out).all();return out"
new="""out=p[0][None] if k==1 else p[0].transpose(2,0,1)
 if f['mode']=='daily_increment':
  y=flat[:,o*96-96:o*96];m=valid[:,o*96-96:o*96];x=np.where(m,(y-basis['mean'][:,None])/basis['scale'][:,None],0);last=(basis['vectors'][:k]@x).reshape(k,-1,rate).mean(-1);slots=96//rate
  for j in range(H):out[:,:,j*slots:(j+1)*slots]+=last[:,None] if j==0 else out[:,:,(j-1)*slots:j*slots]
 assert out.shape==(k,9,H*96//rate) and np.isfinite(out).all();return out"""
assert old in s;s=s.replace(old,new);compile(s,'joint_factor_moirai.py','exec');(R/'code/joint_factor_moirai.py').write_text(s,encoding='utf-8')
proto=json.loads((B/'round41/docs/protocol.json').read_text(encoding='utf-8'));families=[dict(id=f'factor{k}_r{rate}_{mode}',rank=k,rate=rate,context_days=7,variant='base',scope='frozen',mode=mode) for k in [1,3,6] for rate in [1,4,12] for mode in ['level','daily_increment']]
proto.update(round=43,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model='Moirai2 joint factor dailyincrement forecasting with exact matched 7daylevel controls and one shared correction configuration',families=families,train_steps=0,script_sha256=hashlib.sha256(s.encode()).hexdigest(),difference='Powerfactor difference fromprevious24h; past7delta days use8rawhistoricaldays. Forecast futuredelta andcausallyintegrate startingknownlastday profile, thenprojectcorrection into priorPowercurve. FuturetruePower never input. Signedfactor increments mayaccumulateerror; DEVselection must supportboth24h and7day. MatchsameK/rate/history length levelcontrols. This is customresearch frontend, notofficialfinetuning recipe.',forecast_context_days=7,registered_after='R41 andR42 protocols andfrozen result already exist. MayTEST alreadyviewed; this is exploratory, not independent confirmation.')
(R/'docs/protocol.json').write_text(json.dumps(proto,ensure_ascii=False,indent=2),encoding='utf-8')
g=(B/'round41/code/run_factor_guarded.py').read_text(encoding='utf-8').replace("R=T/'experiment/round41'","R=T/'experiment/round43'");(R/'code/run_factor_guarded.py').write_text(g,encoding='utf-8');print('R43 registered nine paired delta/level controls, one shared architecture forboth horizons')
