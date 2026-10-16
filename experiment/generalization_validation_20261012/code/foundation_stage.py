"""One isolated foundation family; offline weights; caches bound to protocol."""
from pathlib import Path
import argparse,os,sys,time,gc
sys.dont_write_bytecode=True
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg')
import numpy as np
from core import *
def run(root,folder,family,phase):
    protocol=protocol_check(folder)
    if phase!='dev':
        selection=folder/'results/nlr_selection.json'
        assert sha(selection)==(selection.with_suffix('.sha256')).read_text().strip()
    sys.path[:0]=[str(root/'experiment/round46/code'),str(root/'experiment/round44/reference/PowerData_Prediction')]
    # Supplemental research packages only; preserve primary CUDA torch/numpy.
    sys.path.append(str(root/'experiment/round44/.venv/Lib/site-packages'))
    import torch
    from hdsf_foundation_adapters import Adapter
    device='cpu' if family=='tirex2' else'cuda';assert device=='cpu' or torch.cuda.is_available()
    z=np.load(folder/'data/nlr_15min.npz');x=z['x']/z['scales'][:,None,None];dates=z['dates']
    origins=protocol['nlr_origins'][phase]
    # Candidate/member/horizon definitions are identical to original HDSF.
    tasks={'chronos2':[('chronos2_long',1,28)],'tirex2':[('tirex2',1,14)],'timesfm25':[('timesfm25',1,14)],
        'timesfm3':[('timesfm30',1,14),('timesfm30_long',1,21),('timesfm30_uni',1,21),('timesfm30_long',7,21),('timesfm30_uni',7,21)],
        'toto2':[('toto2_joint',1,14),('toto2_single',1,14),('toto2_joint',7,14)],
        'patchtst':[('patchtstfm_long',1,21),('patchtstfm_long',7,21)]}[family]
    configs=read(root/'experiment/round46/results/selection.json')['families']
    modes=[('original',None,tasks)]
    if family in ['timesfm3','toto2','patchtst']:
        modes.append(('google_r46',root/'experiment/round46/models'/f'{family}_{configs[family]["config"]}_cut22.pt',[t for t in tasks if t[1]==7]))
    if family=='patchtst':modes.append(('google_r47_curve',root/'experiment/round47/models/patchtst_curve_cut22.pt',[t for t in tasks if t[1]==1]))
    identity=sha(folder/'docs/protocol.json');status=folder/'results'/f'{family}_{phase}_status.json'
    save(status,dict(state='loading',pid=os.getpid(),utc=utc(),family=family,phase=phase,device=device))
    adapter=Adapter(root,family,device);adapter.set_scope('tail');initial=adapter.trainable_state()
    completed=0;total=sum(len(oo)*len(mts) for oo,mts in [(origins,mode[2]) for mode in modes])
    for label,statepath,definitions in modes:
        if statepath is not None:
            assert protocol['source_bindings'][str(statepath.relative_to(root))]==sha(statepath)
            adapter.load_trainable_state(torch.load(statepath,map_location='cpu',weights_only=True))
        else:adapter.load_trainable_state(initial)
        for o in origins:
            memo={}
            for name,h,days in definitions:
                path=folder/'cache'/f'{phase}_{family}_{label}_{name}_H{h}_o{o}.npz'
                ctx=np.ascontiguousarray(x[:,o-days:o].reshape(2,-1),dtype=np.float32);digest=hashlib.sha256(ctx.tobytes()).hexdigest()
                if path.exists():
                    q=np.load(path);assert str(q['protocol'])==identity and str(q['input_digest'])==digest
                    assert np.isfinite(q['p']).all();completed+=1;continue
                began=time.perf_counter();key=(days,h)
                if key in memo:point=memo[key]
                else:
                    # Each homogeneous target is one independent channel; no
                    # nested facility-total/IT crossattention is introduced.
                    point=adapter.predict(ctx,h*96,'uni');memo[key]=point
                assert point.shape==(2,h*96) and np.isfinite(point).all()
                np_save(path,p=point,input_digest=digest,protocol=identity,origin=o,member=name,label=label,device=device)
                completed+=1
                save(status,dict(state='inference',pid=os.getpid(),utc=utc(),completed=completed,total=total,family=family,phase=phase,label=label,member=name,origin_date=str(dates[o]),seconds=time.perf_counter()-began))
                print(f'{family} {phase} {label} {name} H{h} {dates[o]} {time.perf_counter()-began:.2f}s [{completed}/{total}]',flush=True)
        adapter.load_trainable_state(initial)
    save(folder/'results'/f'{family}_{phase}_complete.json',dict(all_passed=True,utc=utc(),files=completed,protocol_sha256=identity,device=device,
        pretrained_weights_updated_on_target=False,google_finetuned_weights_reused=True))
    save(status,dict(state='complete',utc=utc(),family=family,phase=phase,completed=completed))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root');a.add_argument('family');a.add_argument('phase',choices=['dev','test','confirm']);s=a.parse_args()
    root=Path(s.root);folder=root/'experiment/generalization_validation_20261012'
    try:run(root,folder,s.family,s.phase)
    except BaseException as ex:
        import traceback
        save(folder/'results'/f'{s.family}_{s.phase}_failure.json',dict(utc=utc(),error=repr(ex),traceback=traceback.format_exc()));raise
