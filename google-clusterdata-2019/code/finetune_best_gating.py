"""Small, bounded convex expert gate around the frozen round8b best forecast.

Training labels, scaling statistics, and replay forecasts obey each time cutoff.
The reused May development/test periods provide exploratory evidence only.
"""
import copy, datetime, gc, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
B = ROOT.parent / 'experiment'
R = B / 'round8c'
sys.path.insert(0, str(B / 'round5/code'))
from lib5 import load, truth, metrics, TRAIN, DEV, TEST, anchor, save_json, DEVICE
from calibration import bank, fit, features
from finetune_tf3 import load_delta, forecast, BASECFG
from assess import comparisons

CONFIGS = {f'{scope}_penalty{pen}': dict(scope=scope, penalty=pen)
           for scope in ['global', 'regime'] for pen in [.1, 1., 10.]}
STEPS = [100, 300, 600]
W0 = np.array([.25, .25, .5], dtype=np.float32)
FEATURE_NAMES = ['last_level_change', 'mean3_level_change', 'recent_volatility',
                 'proxy_availability7', 'hour_sin', 'hour_cos',
                 'chronos_deviation', 'tirex_deviation', 'timesfm_deviation']

def gate_features(d, origins, experts):
    values = []
    for j, o in enumerate(origins):
        past = d['y'][:, o-7:o]
        mean = past.mean((1, 2)).clip(.05)
        slot = np.arange(96)
        scalar = [(past[:, -1].mean(1)-mean)/mean,
                  (past[:, -3:].mean((1, 2))-mean)/mean,
                  past[:, -3:].std((1, 2))/mean,
                  d['pvalid'][:, o-7:o].mean((1, 2))]
        xx = [np.repeat(z[:, None], 96, axis=1) for z in scalar]
        xx += [np.broadcast_to(np.sin(2*np.pi*slot/96), (57, 96)),
               np.broadcast_to(np.cos(2*np.pi*slot/96), (57, 96))]
        center = (experts[:, j]*W0).sum(-1)
        xx += [(experts[:, j, :, k]-center)/mean[:, None] for k in range(3)]
        values.append(np.stack(xx, -1))
    x = np.stack(values, 1).astype('float32')
    assert np.isfinite(x).all()
    return x

def parts(d, origins, cutoff, inference_split=None):
    """Return exact best baseline plus three known forecast experts.

    The TimesFM expert incorporates only the frozen accepted update; its
    effective update includes clipping used by round8b, for exact anchoring.
    """
    selection = json.loads((B/'round8b/results/selection.json').read_text())
    a = selection['chosen']['alpha']
    if inference_split is not None:
        cs = np.load(B/f'round3/results/chronos_{inference_split}.npz')['native']
        ts = np.load(B/f'round3/results/tirex_{inference_split}.npz')['native']
        fs = np.load(B/f'round3/results/tf3_{inference_split}.npz')['tf3_cellprod']
        if inference_split == 'dev':
            old = .5*anchor('dev') + .5*np.load(B/'round5/results/calibration_dev.npz')['residual_ridge100000']
            candidate = np.load(B/'round8b/results/dev_predictions.npz')[selection['config']]
            prior = np.clip((1-a)*old+a*candidate, 0, 1)
        else:
            cache = np.load(B/'round8b/results/test_predictions.npz')
            old, candidate, prior = [cache[k] for k in ['prior', 'candidate', 'selected']]
    else:
        cachepath = R/'results'/f'training_parts_cutoff{cutoff}.npz'
        if cachepath.exists():
            z = np.load(cachepath)
            assert z['origins'].tolist() == list(origins)
            return z['prior'], z['experts']
        splits = ['train'] if cutoff == 14 else ['train', 'dev']
        cs = np.concatenate([np.load(B/f'round3/results/chronos_{s}.npz')['native'] for s in splits], 1)
        ts = np.concatenate([np.load(B/f'round3/results/tirex_{s}.npz')['native'] for s in splits], 1)
        fs = np.concatenate([np.load(B/f'round3/results/tf3_{s}.npz')['tf3_cellprod'] for s in splits], 1)
        bb = bank(splits)
        reg, _ = fit(d, bb, origins, 100000)
        bases = []
        for o in origins:
            xx, base, _ = features(d, o, bb)
            bases.append(.5*base+.5*np.clip(base+reg.predict(xx).reshape(57, 96), 0, 1))
        old = np.stack(bases, 1)
        stage = 'dev' if cutoff == 14 else 'final'
        f = load_delta(B/'round8b/models'/f'{selection["config"]}_{stage}.pt')
        new = np.clip(forecast(f, d, origins, BASECFG['tf3_cellprod']), 0, 1)
        del f
        gc.collect()
        torch.cuda.empty_cache()
        candidate = np.clip(old+.5*(new-fs), 0, 1)
        prior = np.clip((1-a)*old+a*candidate, 0, 1)
    effective_tf = fs+2*a*(candidate-old)
    experts = np.stack([cs, ts, effective_tf], -1)
    if inference_split is None:
        np.savez_compressed(cachepath, prior=prior, experts=experts, origins=origins)
    return prior, experts

def weights(model, x):
    xs = model['scaler'].transform(x.reshape(-1, len(FEATURE_NAMES))).reshape(x.shape)
    if model['config']['scope'] == 'global':
        xs = np.zeros_like(xs)
    logits = .5*np.tanh(xs@model['coef']+model['bias'])+np.log(W0)
    z = np.exp(logits-logits.max(-1, keepdims=True))
    return z/z.sum(-1, keepdims=True)

def predict(model, d, origins, prior, experts):
    x = gate_features(d, origins, experts)
    w = weights(model, x)
    result = np.clip(prior+((w-W0)*experts).sum(-1), 0, 1)
    assert np.isfinite(result).all()
    return result

def train(d, origins, cutoff, config, steps=600, callback=None):
    prior, experts = parts(d, origins, cutoff)
    xx = gate_features(d, origins, experts)
    yy, mm = truth(d, origins)
    scaler = StandardScaler().fit(xx.reshape(-1, len(FEATURE_NAMES))[mm.reshape(-1)])
    xx = scaler.transform(xx.reshape(-1, len(FEATURE_NAMES))).reshape(xx.shape)
    if config['scope'] == 'global':
        xx *= 0
    tt = lambda x: torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    x, p0, ee, y, mask = map(tt, [xx, prior, experts, yy, mm])
    denom = (y*mask).sum((1, 2)).clamp_min(1e-6)
    coef = torch.nn.Parameter(torch.zeros(len(FEATURE_NAMES), 3, device=DEVICE), requires_grad=config['scope']=='regime')
    bias = torch.nn.Parameter(torch.zeros(3, device=DEVICE))
    params = [bias]+([coef] if config['scope']=='regime' else [])
    optimizer = torch.optim.Adam(params, lr=.03)
    w0 = tt(W0)
    history = []
    print('FIT', cutoff, config, 'trainable_parameters', sum(z.numel() for z in params), flush=True)
    for step in range(1, steps+1):
        w = torch.softmax(w0.log()+.5*torch.tanh(x@coef+bias), -1)
        pred = (p0+((w-w0)*ee).sum(-1)).clamp(0, 1)
        # Smooth absolute loss is stable near zero; weights match macro WAPE.
        smooth = ((pred-y).square()+.002**2).sqrt()-.002
        point = ((smooth*mask).sum((1, 2))/denom).mean()
        penalty = config['penalty']*(w-w0).square().mean()+.0001*coef.square().sum()
        loss = point+penalty
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(z.grad is None or torch.isfinite(z.grad).all() for z in params)
        optimizer.step()
        if step % 100 == 0:
            row = dict(step=step, objective=float(loss.detach()), point=float(point.detach()),
                       max_weight_change=float((w-w0).abs().max().detach()))
            history.append(row)
            print('TRAIN', row, flush=True)
        if callback and step in STEPS:
            callback(dict(scaler=scaler, coef=coef.detach().cpu().numpy().copy(),
                          bias=bias.detach().cpu().numpy().copy(), config=config, cutoff=cutoff, step=step), step)
    return dict(scaler=scaler, coef=coef.detach().cpu().numpy().copy(),
                bias=bias.detach().cpu().numpy().copy(), config=config, cutoff=cutoff, step=step), history

def select(d, y, mask, prior, predictions):
    best = min(predictions, key=lambda k: metrics(y[:, :4], predictions[k][:, :4], mask[:, :4])['wape'])
    rows = []
    for a in [0, .1, .25, .5, 1]:
        pp = (1-a)*prior[:, 4:]+a*predictions[best][:, 4:]
        def error(z):
            return 100*(np.abs(y[:, 4:]-z)*mask[:, 4:]).sum((1, 2))/(y[:, 4:]*mask[:, 4:]).sum((1, 2))
        gain = error(prior[:, 4:])-error(pp)
        cg = pd.DataFrame(dict(cell=d['cells'], gain=gain)).groupby('cell').gain.mean().to_numpy()
        daily = []
        for j in range(4):
            def dw(z):
                return 100*((np.abs(y[:, j+4]-z[:, j])*mask[:, j+4]).sum(1)/(y[:, j+4]*mask[:, j+4]).sum(1)).mean()
            daily.append(dw(prior[:, 4:])-dw(pp))
        se_cell = float(cg.std(ddof=1)/np.sqrt(len(cg)))
        se_day = float(np.std(daily, ddof=1)/2)
        rows.append(dict(alpha=a, wape=float(error(pp).mean()), paired_cell_se=se_cell,
                         paired_day_se=se_day, conservative_se=max(se_cell, se_day)))
    win = min(rows, key=lambda x: x['wape'])
    chosen = min([x for x in rows if x['wape'] <= win['wape']+win['conservative_se']+1e-12], key=lambda x:x['alpha'])
    return best, chosen, rows

def main():
    for sub in ['code', 'docs', 'models', 'results']:
        (R/sub).mkdir(parents=True, exist_ok=True)
    if (R/'results/summary.json').exists():
        print('Already complete; preserve original frozen result.', flush=True)
        return
    if (R/'results/selection.json').exists():
        raise RuntimeError('Existing frozen selection requires explicit replay, never reselect from test results.')
    protocol = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), registry=CONFIGS,
                    checkpoints=STEPS, train_origins=TRAIN, dev_origins=DEV, test_origins=TEST,
                    final_train_origins=TRAIN+DEV, feature_names=FEATURE_NAMES,
                    parameters='3 global biases or 30 shared regime parameters; no site-specific learned parameters',
                    bounds='softmax nonnegative weights sum one; logits update bounded by +/-0.5; power [0,1]',
                    initialization='Exactly zero correction to frozen round8b accepted forecast',
                    objective='Macro-normalized smooth absolute power loss plus weight and coefficient shrinkage',
                    selection='First four dev days choose checkpoint; last four choose shrinkage alpha by the larger paired-cell/day one-SE rule, preferring alpha zero. Freeze hashes before new test scoring.',
                    development_reused=True, test_reused_in_previous_rounds=True, independent_confirmation=False,
                    complete_cpu_input=False, external_license='TimesFM3 Non-Commercial v1.0; research use only')
    save_json(R/'docs/protocol.json', protocol)
    d = load()
    prior, ex = parts(d, DEV, 14, 'dev')
    y, m = truth(d, DEV)
    predictions, rows = {}, []
    for name, cfg in CONFIGS.items():
        def checkpoint(model, step):
            key = f'{name}_s{step}'
            predictions[key] = predict(model, d, DEV, prior, ex)
            row = dict(model=key, **metrics(y, predictions[key], m))
            rows.append(row)
            with (R/'models'/f'{key}_dev.pkl').open('wb') as f:
                pickle.dump(model, f)
            np.savez_compressed(R/'results/dev_predictions.npz', prior=prior, **predictions)
            pd.DataFrame(rows).to_csv(R/'results/dev_metrics.csv', index=False)
            print('DEV', row, flush=True)
        _, history = train(d, TRAIN, 14, cfg, callback=checkpoint)
        save_json(R/'results'/f'{name}_history.json', history)
    key, chosen, validation = select(d, y, m, prior, predictions)
    selection = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), config=key,
                     chosen=chosen, validation=validation,
                     protocol_sha256=hashlib.sha256((R/'docs/protocol.json').read_bytes()).hexdigest())
    save_json(R/'results/selection.json', selection)
    (R/'results/selection.sha256').write_text(hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest())
    print('FROZEN', selection, flush=True)
    name, step = key.rsplit('_s', 1)
    model, history = train(d, TRAIN+DEV, 22, CONFIGS[name], int(step))
    with (R/'models'/f'{key}_final.pkl').open('wb') as f:
        pickle.dump(model, f)
    base, experts = parts(d, TEST, 22, 'test')
    candidate = predict(model, d, TEST, base, experts)
    selected = np.clip((1-chosen['alpha'])*base+chosen['alpha']*candidate, 0, 1)
    ty, tm = truth(d, TEST)
    scores = {k:metrics(ty,z,tm) for k,z in dict(prior=base,candidate=candidate,selected=selected).items()}
    checks = {}
    w = weights(model, gate_features(d, TEST, experts))
    checks['weights_finite_convex'] = bool(np.isfinite(w).all() and (w>=0).all() and np.allclose(w.sum(-1),1))
    checks['power_finite_bounded'] = bool(np.isfinite(selected).all() and (selected>=0).all() and (selected<=1).all())
    checks['training_cutoff22'] = model['cutoff']==22
    checks['selection_hash'] = hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest()==(R/'results/selection.sha256').read_text()
    zero = copy.deepcopy(model)
    zero['coef'] *= 0
    zero['bias'] *= 0
    checks['zero_gate_exact_prior'] = bool(np.allclose(predict(zero,d,TEST,base,experts),base,atol=1e-7,rtol=0))
    for o in [14,22,30]:
        e = ex[:,[DEV.index(o)]] if o in DEV else experts[:,[TEST.index(o)]]
        x0 = gate_features(d,[o],e)
        dd = {k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in d.items()}
        for k in ['y','prod','valid','pvalid']:
            dd[k][:,o:] = 0
        checks[f'future_input_invariance_origin{o}'] = bool(np.array_equal(x0,gate_features(dd,[o],e)))
    assert all(checks.values()), checks
    np.savez_compressed(R/'results/test_predictions.npz',prior=base,candidate=candidate,selected=selected)
    save_json(R/'results/final_training.json',history)
    save_json(R/'results/summary.json',dict(scores=scores,selection=selection,
                  comparison_vs_previous=comparisons(ty,tm,selected,base,d['cells']),
                  candidate_comparison=comparisons(ty,tm,candidate,base,d['cells']),
                  independent_confirmation=False,large_improvement=False,checks=checks,all_passed=True))
    print('TEST',scores,'CHECKS',checks,flush=True)

if __name__ == '__main__':
    main()
