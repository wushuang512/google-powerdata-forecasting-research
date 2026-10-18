"""Run with --help. No command copies data or weights into the Git repository."""
import argparse,os,sys
from pathlib import Path
from repro import support as s

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--source-root',type=Path,help='Optional fixed upstream source mirror; defaults to this checkout')
 sub=ap.add_subparsers(dest='command',required=True)
 sub.add_parser('init');sub.add_parser('doctor');p=sub.add_parser('fetch');p.add_argument('kind',choices=['google','nlr','models']);p.add_argument('--accept-timesfm-license',action='store_true')
 p=sub.add_parser('local-assets');p.add_argument('kind',choices=['google','nlr','models']);p.add_argument('original_root',type=Path)
 for name in ['google-prepare','google-baselines','google-score','nlr-prepare']:sub.add_parser(name)
 p=sub.add_parser('nlr-run');p.add_argument('--cpu-python',type=Path,required=True)
 p=sub.add_parser('legacy-bank');p.add_argument('family',choices=['chronos2','tirex2','timesfm3'])
 sub.add_parser('legacy-calibrator');sub.add_parser('legacy-r8-train')
 p=sub.add_parser('neural-smoke');p.add_argument('family',choices=['timesfm3','toto2','patchtst'])
 p=sub.add_parser('google-all');p.add_argument('--cpu-python',type=Path,required=True);p.add_argument('--gpu-python',type=Path,required=True);p.add_argument('--include-nlr-source-models',action='store_true')
 p=sub.add_parser('google-member');p.add_argument('member');p.add_argument('--phase',choices=['dev','test'],required=True)
 p=sub.add_parser('google-adapt');p.add_argument('family',choices=['timesfm3','toto2','patchtst']);p.add_argument('--recipe',choices=['r46','r47curve'],required=True);p.add_argument('--phase',choices=['dev','test'],required=True);p.add_argument('--trained-state',type=Path)
 args=ap.parse_args();w=s.initialize(args.work);s.attach_sources(w,args.source_root)
 if args.command=='init':print('Created isolated reproduction work directory:',w);return
 if args.command=='fetch':s.fetch(w,args.kind,args.accept_timesfm_license);return
 if args.command=='local-assets':s.linked_local_assets(w,args.original_root,args.kind);return
 if args.command=='doctor':
  import importlib.metadata as im
  required=['numpy','pandas','scipy','scikit-learn','lightgbm','torch','timesfm','transformers','chronos-forecasting','tirex-2','toto-2','pyarrow'];versions={}
  for name in required:
   try:versions[name]=im.version(name)
   except im.PackageNotFoundError:versions[name]=None
  missing=[m['model_id'] for m in s.read(s.REPO/'repro/assets.lock.json')['models'] if not (w/m['location']/m['files'][0]['name']).exists()]
  import json;print(json.dumps({'packages':versions,'missing_model_locations':missing,'upstream_hdsf_present':(w/'experiment/round44/reference/PowerData_Prediction/power_forecast/data.py').is_file(),'granite_source_present':any((w/p).exists() for p in ['third_party/granite-tsfm/tsfm_public/models/patchtst_fm','experiment/round44/reference/granite-tsfm-3d92535ca4138648b9d1e2f3f1421b5dec4dc3aa/tsfm_public/models/patchtst_fm'])},indent=2));return
 if args.command=='google-all':
  from repro import pipeline
  pipeline.google(w,args.cpu_python,args.gpu_python,args.source_root,args.include_nlr_source_models);return
 if args.command=='neural-smoke':
  from repro import neural_smoke
  neural_smoke.run(w,args.family);return
 if args.command.startswith('legacy-'):
  from repro import legacy
  if args.command=='legacy-bank':legacy.bank(w,args.family)
  elif args.command=='legacy-calibrator':legacy.fit_source_calibrator(w)
  else:legacy.train_r8(w)
  return
 if args.command.startswith('google-'):
  from repro import google as g
  if args.command=='google-prepare':print(g.prepare(w))
  elif args.command=='google-baselines':print(g.baselines(w))
  elif args.command=='google-member':g.member(w,args.member,args.phase)
  elif args.command=='google-adapt':
   if args.recipe=='r47curve' and args.family!='patchtst':ap.error('The adopted R47 curve route is PatchTST only')
   g.train_or_infer(w,args.family,args.recipe,args.phase,args.trained_state)
  elif args.command=='google-score':print(g.score(w))
 else:
  from repro import nlr
  nlr.run(w,args.command,getattr(args,'cpu_python',None))
if __name__=='__main__':main()
