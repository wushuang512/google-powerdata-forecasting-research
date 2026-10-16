from pathlib import Path
import zipfile,hashlib,json,sys
R=Path(sys.argv[1])/'experiment/round44';rev='3d92535ca4138648b9d1e2f3f1421b5dec4dc3aa';prefix='granite-tsfm-'+rev+'/'
for p in (R/'pip-cache/http-v2').rglob('*.body'):
 if p.stat().st_size<20*1024*1024:continue
 try:
  with zipfile.ZipFile(p) as z:
   if not any(n.startswith(prefix+'tsfm_public/models/patchtst_fm/') for n in z.namelist()):continue
   dest=R/'reference'/('granite-tsfm-'+rev);dest.mkdir(exist_ok=True);files=[]
   for n in z.namelist():
    assert n.startswith(prefix);tail=n[len(prefix):]
    if not tail or n.endswith('/'):continue
    target=(dest/tail).resolve();assert target.is_relative_to(dest.resolve());raw=z.read(n);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    if tail.startswith('tsfm_public/models/patchtst_fm/'):files.append(dict(path=tail,sha256=hashlib.sha256(raw).hexdigest()))
   audit=dict(source_url='https://github.com/ibm-granite/granite-tsfm',commit=rev,archive_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),directory=str(dest),patch_model_files=files,installation='Editable from unchanged official source; upstream wheel omits model subpackages',build_version_label='0.0+g3d92535',metadata_exception='Upstream toolkit declares Torch>=2.10, while prescribed HDSF requirements fix2.8. Inference uses original private model module under sourceCPU Torch2.8; no dataset shim and no altered model code. Record actual import andforward success, do not claim pipcheck consistency.')
   (R/'results/granite_source.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print(str(dest));break
 except zipfile.BadZipFile:continue
else:raise RuntimeError('Official cached source archive not found')
