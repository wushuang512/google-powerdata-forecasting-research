"""Compare actual learned head/last-block tensors with original checkpoint tensors."""
from pathlib import Path
import argparse, datetime, hashlib, json, sys
sys.dont_write_bytecode=True
import torch

ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('family');args=ap.parse_args()
t=Path(args.root);r=t/'experiment/round46';q=r/'results'
sys.path[:0]=[str(r/'code'),str(t/'experiment/round44/reference/PowerData_Prediction')]
from hdsf_foundation_adapters import Adapter
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
torch.manual_seed(20261012)
family=args.family
adapter=Adapter(t,family,'cpu' if family=='tirex2' else 'cuda')
adapter.set_scope('head');head=set(adapter.train_names)
adapter.set_scope('tail');tail=set(adapter.train_names)-head
assert tail,'Actual encoder block names required'
base={n:v.detach().cpu().clone() for n,v in adapter.model.named_parameters() if n in head|tail}
sel=read(q/'selection.json');results=[]
for cut,cfgs in [(14,[f'{scope}_s{step}' for scope in ['head','tail'] for step in [40,80]]),(22,[sel['families'][family]['config']])]:
    for cfg in cfgs:
        path=r/'models'/f'{family}_{cfg}_cut{cut}.pt'
        state=torch.load(path,map_location='cpu',weights_only=True)
        changedhead=[];changedtail=[];l2head=0.;l2tail=0.
        for name,tensor in state.items():
            assert name in base
            diff=tensor-base[name]
            if torch.count_nonzero(diff).item():
                if name in head:changedhead.append(name);l2head+=float(diff.double().square().sum())
                else:changedtail.append(name);l2tail+=float(diff.double().square().sum())
        assert changedhead,(family,cfg,cut,'no output-head weight changes')
        if cfg.startswith('tail'):assert changedtail,(family,cfg,cut,'no encoder-block weight changes')
        results.append(dict(config=cfg,cutoff=cut,model_file_sha256=sha(path),
                            changed_head_tensors=changedhead,changed_encoder_tensors=changedtail,
                            head_delta_l2=l2head**.5,encoder_delta_l2=l2tail**.5,all_passed=True))
        del state
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),family=family,
         all_passed=True,checkpoint_count=len(results),results=results,
         head_names=sorted(head),actual_last_encoder_block_names=sorted(tail),
         device=str(adapter.device),selection_sha256=sha(q/'selection.json'))
(q/f'{family}_weight_update_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(family=family,all_passed=True,checkpoint_count=len(results),
                     encoder_delta_l2=[x['encoder_delta_l2'] for x in results if x['config'].startswith('tail')])))
