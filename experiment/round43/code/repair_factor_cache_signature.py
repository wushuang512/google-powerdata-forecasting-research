from pathlib import Path
import sys,json,hashlib,datetime,numpy as np
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round43';Q=R/'results';src=R/'code/joint_factor_moirai.py';old_bytes=src.read_bytes();old_file_sha=hashlib.sha256(old_bytes).hexdigest();old=src.read_text(encoding='utf-8');proto=json.loads((R/'docs/protocol.json').read_text(encoding='utf-8'));assert hashlib.sha256(old.encode()).hexdigest()==proto['script_sha256']
new=old.replace("(o-f['context_days'])*96:o*96", "(o-f['context_days']-(1 if f.get('mode')=='daily_increment' else 0))*96:o*96");assert new!=old and old.count("(o-f['context_days'])*96:o*96")==2;compile(new,'joint_factor_moirai.py','exec')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(B/'results/data.npz') as z:flat=z['y'].reshape(57,-1);valid=z['valid'].reshape(57,-1)
selection=Q/'selection.json';freeze_before=sha(selection);native_before={p:sha(Q/f'native_{p}.npz') for p in ['dev','test','deploy']};sel=json.loads(selection.read_text(encoding='utf-8'));records=[];updates=[]
for phase in ['dev','test','deploy']:
 cut=14 if phase=='dev' else 22
 with np.load(R/f'models/factor_basis_cut{cut}.npz') as z:assert str(z['signature'])==hashlib.sha256(sha(R/'docs/protocol.json').encode()+np.ascontiguousarray(flat[:,:cut*96]).tobytes()+np.ascontiguousarray(valid[:,:cut*96]).tobytes()).hexdigest()
 for f in proto['families'] if phase=='dev' else[sel['family']]:
  for H in [1,7]:
   oo=([14,16,18,20] if H==1 else[14,15]) if phase=='dev' else([31] if phase=='deploy' else(list(range(22,31)) if H==1 else[22,23,24]))
   for o in oo:
    key=f'{f["id"]}_H{H}_o{o}';p=Q/'native_cache'/f'{phase}_{key}.npz';start=o-f['context_days'];extra=1 if f['mode']=='daily_increment' else 0
    assert not extra or start<=cut,'Previously omitted extra day must already be independently bound by training PCA prefix; no uncertified cache migration'
    suffix=sha(B/'round35/code/moirai_native_adapter.py')+sha(B/'round35/models/moirai2/model.safetensors')+sha(R/f'models/factor_basis_cut{cut}.npz')+'base'
    def fingerprint(script_sha,startday):return hashlib.sha256((sha(R/'docs/protocol.json')+script_sha+suffix).encode()+np.ascontiguousarray(flat[:,startday*96:o*96]).tobytes()+np.ascontiguousarray(valid[:,startday*96:o*96]).tobytes()).hexdigest()
    with np.load(p) as z:quantiles=z['quantiles'].copy();old_sig=str(z['signature']);assert old_sig==fingerprint(old_file_sha,start)
    new_sig=fingerprint(hashlib.sha256(new.encode()).hexdigest(),start-extra);records.append(dict(path=str(p.relative_to(R)),original_file_sha256=sha(p),old_signature=old_sig,new_signature=new_sig,quantiles_sha256=hashlib.sha256(quantiles.tobytes()).hexdigest(),extra_history_already_bound_by_original_PCA_prefix=bool(extra),actual_input_days=f['context_days']+extra));updates.append((p,quantiles,new_sig))
(R/'docs/cache_signature_before.py').write_bytes(old_bytes);src.write_bytes(new.encode('utf-8'))
for p,q,sig in updates:np.savez_compressed(p,quantiles=q,signature=sig)
assert sha(selection)==freeze_before and all(sha(Q/f'native_{phase}.npz')==v for phase,v in native_before.items())
(Q/'cache_signature_repair.json').write_text(json.dumps(dict(all_passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_script_sha256=old_file_sha,original_normalized_text_sha256=hashlib.sha256(old.encode()).hexdigest(),current_script_sha256=sha(src),first_attempt='Read-only preflight stopped at signature validation because rawWindows CRLF byteSHA differsfromnormalizedLF textSHA; no mutation. Corrected rawbyte comparison before migration.',changes='Only cache input signature extends by one pastday for dailyincrements. Algorithm, hyperparameters and forecasts unchanged. Existing DEV extra days were already bound by whole training PCA prefix, TEST/DEPLOY selected level uses noextra days.',same_hyperparameters_and_frozen_selection=True,selection_sha256=freeze_before,native_arrays_before_sha256=native_before,records=records),ensure_ascii=False,indent=2),encoding='utf-8');print('Signature coverage extended; all existing cache inputs bound; frozen selection and actual forecasts unchanged')
