"""Run only the six registered history/control Ridge development candidates.

This stage does not load a foundation model, call workload forecasts, or score
test dates. Its verified outputs can be reused by the registered full run.
"""
import datetime
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import complete_usage_models as ex

NAMES = [f'{group}_ridge{alpha}' for group in ['control', 'history']
         for alpha in [10000, 100000, 1000000]]
LEDGER = ex.R / 'results/partial_dev_ledger.json'
PREDICTIONS = ex.R / 'results/partial_dev_predictions.npz'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def atomic_predictions(values):
    tmp = PREDICTIONS.with_name(PREDICTIONS.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    os.replace(tmp, PREDICTIONS)


def input_fingerprints():
    """Bind the cache to the protocol, complete CPU and actual frozen inputs."""
    paths = {
        'protocol': ex.R / 'docs/protocol.json',
        'complete_cpu': ex.ROOT / 'aggregated/complete_usage_a_15min.npz',
        'resources': ex.ROOT / 'aggregated/resources_15min.npz',
        'power_data': ex.B / 'results/data.npz',
        'prior_development_selection': ex.B / 'round8b/results/selection.json',
        'prior_development_predictions': ex.B / 'round8b/results/dev_predictions.npz',
        'round5_development_calibration': ex.B / 'round5/results/calibration_dev.npz',
    }
    # The fitting helper opens these already-made forecasts, but accesses only
    # origin-indexed forecasts and labels strictly before its fitting origins.
    # Test labels are never scored or used for cache selection in this stage.
    for split in ['train', 'dev', 'test']:
        for model in ['tf3', 'chronos', 'tirex']:
            paths[f'{model}_{split}_forecasts'] = (
                ex.B / 'round3/results' / f'{model}_{split}.npz')
    with np.load(paths['complete_cpu']) as z:
        assert bool(z['complete'])
        assert int(z['source_files']) == int(z['expected_source_files']) == 12085
    return {key: {'path': str(path), 'sha256': sha256(path)}
            for key, path in paths.items()}


def array_sha256(values):
    values = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(values.dtype).encode('ascii'))
    h.update(json.dumps(list(values.shape)).encode('ascii'))
    h.update(values.tobytes())
    return h.hexdigest()


def registered_config(name, cfg):
    protocol = json.loads((ex.R / 'docs/protocol.json').read_text(encoding='utf-8'))
    assert name in NAMES, f'Unregistered light-stage candidate: {name}'
    assert protocol['registry'][name] == cfg, 'Candidate differs from frozen registry'
    assert cfg['kind'] == 'ridge' and cfg['group'] in ['control', 'history']
    return protocol


def validate_ledger(ledger, current):
    assert ledger['version'] == 1
    assert ledger['training_origins'] == ex.TRAIN
    assert ledger['development_origins'] == ex.DEV
    assert ledger['test_scored'] is False
    assert ledger['inputs'] == current, 'Light-stage frozen inputs changed'
    assert ledger['protocol_sha256'] == current['protocol']['sha256']
    assert ledger['complete_cpu_aggregate_sha256'] == current['complete_cpu']['sha256']


def cached_development_ridge(name, cfg):
    """Return a verified cached DEV prediction, or None when it is not present.

    A modified input, model, configuration or saved prediction fails loudly;
    it cannot silently substitute a different experiment's cached result.
    """
    if name not in NAMES:
        return None
    registered_config(name, cfg)
    if not LEDGER.exists():
        return None
    ledger = json.loads(LEDGER.read_text(encoding='utf-8'))
    validate_ledger(ledger, input_fingerprints())
    if name not in ledger['models']:
        return None
    entry = ledger['models'][name]
    assert entry['config'] == cfg
    assert entry['training_origins'] == ex.TRAIN
    assert entry['development_origins'] == ex.DEV
    model = ex.R / 'models' / f'{name}_dev.pkl'
    features = ex.R / 'docs' / f'{name}_features.json'
    assert sha256(model) == entry['model_sha256'], 'Cached model changed'
    assert sha256(features) == entry['features_sha256'], 'Cached feature list changed'
    assert PREDICTIONS.exists(), 'Cached predictions missing'
    with np.load(PREDICTIONS) as z:
        assert name in z.files
        prediction = z[name].copy()
    assert prediction.shape == (57, len(ex.DEV), 96)
    assert np.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1
    assert array_sha256(prediction) == entry['prediction_sha256'], 'Cached prediction changed'
    return prediction


def main():
    for sub in ['models', 'docs', 'results']:
        (ex.R / sub).mkdir(parents=True, exist_ok=True)
    inputs = input_fingerprints()
    protocol = json.loads((ex.R / 'docs/protocol.json').read_text(encoding='utf-8'))
    if LEDGER.exists():
        ledger = json.loads(LEDGER.read_text(encoding='utf-8'))
        validate_ledger(ledger, inputs)
    else:
        ledger = dict(
            version=1,
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            stage='registered_control_and_history_ridge_development_only',
            protocol_sha256=inputs['protocol']['sha256'],
            complete_cpu_aggregate_sha256=inputs['complete_cpu']['sha256'],
            inputs=inputs, training_origins=ex.TRAIN, development_origins=ex.DEV,
            test_scored=False, foundation_models_loaded=False, models={},
        )
    predictions = {}
    for name in NAMES:
        cached = cached_development_ridge(name, protocol['registry'][name])
        if cached is not None:
            predictions[name] = cached
    # Only DEV labels below are selected from the common loaded monthly data.
    d, v = ex.inputs()
    prior = ex.previous('dev')
    y, mask = ex.truth(d, ex.DEV)
    for name in NAMES:
        cfg = protocol['registry'][name]
        registered_config(name, cfg)
        if name in predictions:
            print('LIGHT_DEV_CACHE', name, flush=True)
            continue
        model, features = ex.fit_candidate(d, v, ex.TRAIN, cfg, {})
        pred = ex.predict_candidate(model, d, v, ex.DEV, cfg, {}, prior)
        model_path = ex.R / 'models' / f'{name}_dev.pkl'
        tmp = model_path.with_name(model_path.name + '.tmp')
        with tmp.open('wb') as f:
            pickle.dump(model, f)
        os.replace(tmp, model_path)
        feature_path = ex.R / 'docs' / f'{name}_features.json'
        atomic_json(feature_path, features)
        ix = d['cells'] == 'a'
        scores = dict(all57=ex.metrics(y, pred, mask),
                      cell_a_5=ex.metrics(y[ix], pred[ix], mask[ix]))
        assert np.array_equal(pred[~ix], prior[~ix]), 'Other domains must remain frozen'
        predictions[name] = pred
        atomic_predictions(predictions)
        ledger['models'][name] = dict(
            config=cfg, training_origins=ex.TRAIN, development_origins=ex.DEV,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model_path=str(model_path), model_sha256=sha256(model_path),
            features_path=str(feature_path), features_sha256=sha256(feature_path),
            prediction_sha256=array_sha256(pred), score=scores,
        )
        ledger['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        ledger['completed_models'] = len(ledger['models'])
        ledger['all_six_completed'] = len(ledger['models']) == len(NAMES)
        atomic_json(LEDGER, ledger)
        print('LIGHT_DEV', json.dumps(dict(model=name, **scores)), flush=True)
    from export_complete_usage import main as export_source_csv
    if not (ex.ROOT/'aggregated/cella_complete_usage_15min.csv').exists():
        export_source_csv()
    print('LIGHT_STAGE_COMPLETE', len(predictions), 'No test scores calculated', flush=True)


if __name__ == '__main__':
    main()
