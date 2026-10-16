"""Independent real-model replay after poisoning unobserved power suffixes."""
from pathlib import Path
import argparse, datetime, hashlib, json, sys
sys.dont_write_bytecode = True
import numpy as np
import torch

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('family')
args = ap.parse_args()
top = Path(args.root)
r = top / 'experiment/round46'
sys.path.insert(0, str(r / 'code'))
import hdsf_foundation_stage as stage
from hdsf_foundation_adapters import Adapter
stage.initialize(top)
family = args.family
sel = read(r / 'results/selection.json')
assert sha(r / 'results/selection.json') == (r / 'results/selection.sha256').read_text().strip()
cfg = sel['families'][family]['config']
scope, step = cfg.split('_s')
modelpath = r / 'models' / f'{family}_{cfg}_cut22.pt'
info = read(modelpath.with_suffix('.json'))
assert sha(modelpath) == info['model_file_sha256']
checks = []
def check(name, value, **details):
    checks.append(dict(name=name, passed=bool(value), **details))
    assert value, name

original = stage.data.target.copy()
valid = stage.data.target_valid.copy()
# Verify both training recipes by comparing the actual slices, including labels.
for cut in [14, 22]:
    seq = stage.sample_batches(cut, family, 80)
    poisoned = original.copy()
    poisoned[:, cut:] = 0.999
    pvalid = valid.copy()
    pvalid[:, cut:] = False
    flat, pf = original.reshape(57, -1), poisoned.reshape(57, -1)
    vm, pvm = valid.reshape(57, -1), pvalid.reshape(57, -1)
    for batch in seq:
        ix = batch['series_ids']
        o, ctx, h = batch['origin_tick'], batch['context_ticks'], batch['horizon_ticks']
        assert o + h <= cut * 96
        assert np.array_equal(flat[ix, o-ctx:o+h], pf[ix, o-ctx:o+h])
        assert np.array_equal(vm[ix, o:o+h], pvm[ix, o:o+h])
    check(f'training_prefix_future_poison_cut{cut}', True, batches=len(seq))

torch.manual_seed(20261012)
adapter = Adapter(top, family, 'cpu' if family == 'tirex2' else 'cuda')
adapter.set_scope(scope)
check('original_parameter_hash_before_load', adapter.hash_trainable() == info['initial_trainable_sha256'])
state = torch.load(modelpath, map_location='cpu', weights_only=True)
adapter.load_trainable_state(state)
del state
check('selected_parameter_hash_after_load', adapter.hash_trainable() == info['updated_trainable_sha256'])
before = adapter.hash_trainable()
origin = 22
# The model sees a prefix ending D22; overwrite every unseen target and its mask.
stage.data.target[:, origin:] = 0.987654
stage.data.target_valid[:, origin:] = False
replays = []
with np.load(r / 'results' / f'{family}_test_{cfg}.npz') as cache:
    for member in stage.protocol['families'][family]:
        for h in stage.member_tasks(member):
            key = f'{member}_H{h}'
            unseen = stage.predict_member(adapter, member, origin, h)[stage.canon]
            saved = cache[key][:, 0]
            diff = float(np.max(np.abs(unseen-saved)))
            check('actual_future_poison_replay_' + key,
                  np.allclose(unseen, saved, atol=2e-6, rtol=1e-6),
                  max_absolute_difference=diff, series=57, origin=origin)
            replays.append(dict(member=member, horizon_days=h, max_absolute_difference=diff))
stage.data.target[:] = original
stage.data.target_valid[:] = valid
check('inference_parameters_unchanged', before == adapter.hash_trainable())
out = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           family=family, all_passed=all(c['passed'] for c in checks),
           checks=checks, check_count=len(checks), replays=replays,
           device=str(adapter.device), selected_model_sha256=sha(modelpath),
           selection_sha256=sha(r / 'results/selection.json'),
           overfitting_absence_certified=False, independent_confirmation=False)
path = r / 'results' / f'{family}_runtime_audit.json'
path.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps(dict(family=family, all_passed=out['all_passed'], checks=len(checks), replays=replays)))
