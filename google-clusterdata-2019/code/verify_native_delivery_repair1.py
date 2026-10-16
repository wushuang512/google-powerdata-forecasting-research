"""Read-only round10 repair1, original freeze, migration and Word evidence checks.

No model, training code or Word application is imported. JSON is written only
to stdout; the caller may retain that output separately. Missing proof fails.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED = Path(r'${PROJECT_ROOT}')
DESKTOP = Path(r'${PROJECT_ROOT}')
TITLE = 'Google PowerData 2019 原生算电融合与D盘迁移研究汇报.docx'
DEV, TEST = list(range(14, 22)), list(range(22, 31))
CHECKS = {}


def require(key, value):
    CHECKS[key] = bool(value)
    if not CHECKS[key]:
        raise RuntimeError(key)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def date(value):
    result = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise RuntimeError('Evidence time has no timezone')
    return result


def passed(label, proof):
    require(label + '_all_passed', proof.get('all_passed') is True)
    if 'checks' in proof:
        booleans = [v for v in proof['checks'].values() if isinstance(v, bool)]
        require(label + '_boolean_checks', bool(booleans) and all(booleans))


def input_arrays(d, u, resource, present, ix, origin, cfg):
    """Independent copy of the registered past-only query definition."""
    end, start = origin * 96, max(0, origin * 96 - 1344)
    target = d['y'].reshape(57, -1)[ix, start:end].astype('float32')
    proxy = np.where(d['pvalid'].reshape(57, -1)[:, start:end],
                     d['prod'].reshape(57, -1)[:, start:end], np.nan).astype('float32')
    names, covariates = [], []
    for i in ix:
        if np.isfinite(proxy[i]).mean() >= .2:
            names.append(f'production_{i}')
            covariates.append(proxy[i])
    if cfg['usage']:
        flat, mask = u.reshape(57, -1, 6), present.reshape(57, -1)
        for i in ix:
            for j, label in [(0, 'completed_cpu_NCU'), (1, 'completed_memory_normalized')]:
                a = np.where(mask[i, start:end], flat[i, start:end, j], np.nan).astype('float32')
                if not np.isfinite(a).any() or (a[np.isfinite(a)] < 0).any():
                    raise RuntimeError('Historical usage channel invalid')
                names.append(f'{label}_{i}')
                covariates.append(a)
    if cfg['resources']:
        flat = resource.reshape(57, -1, 8)
        for i in ix:
            for j, label in [(1, 'active_fraction'), (2, 'mean_cpu_capacity_NCU')]:
                names.append(f'{label}_{i}')
                covariates.append(flat[i, start:end, j].astype('float32'))
    cov = np.stack(covariates) if covariates else np.empty((0, end - start), dtype='float32')
    return target, cov, names


def input_sha(target, cov, names):
    h = hashlib.sha256(json.dumps(names).encode())
    for a in (target, cov):
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def corrected(prior, native, control, ix, alpha):
    result = prior.copy()
    result[ix] = np.clip(prior[ix] + alpha * (native - control), 0, 1)
    return result


def wape(y, pred, mask):
    return float(np.mean(100 * (np.abs(y - pred) * mask).sum((1, 2)) /
                         (np.abs(y) * mask).sum((1, 2))))


def safe_file(base, value, label, absolute=False):
    base = base.resolve(strict=True)
    path = Path(value)
    if not absolute:
        require(label + '_relative_path_safe', not path.is_absolute() and not path.drive and '..' not in path.parts)
        path = base / path
    path = path.resolve(strict=True)
    require(label + '_inside_expected_root', path.is_relative_to(base) and path.is_file())
    return path


def verify_repair(r, b, cluster, fp, selection, test, deploy, weights):
    """Validate transparent execution repair without changing original DEV/freeze."""
    provenance_path = r / 'docs/repair1_provenance.json'
    provenance = read(provenance_path)
    script = cluster / 'code/native_covariate_experiment_repair1.py'
    policy = dict(timesfm_test='ieee', chronos_test='tf32', timesfm_deploy='ieee', chronos_deploy='ieee')
    expected = dict(repair_id='repair1', status='approved_for_execution',
                    original_script_sha256=fp['script_sha256'], new_script_sha256=sha(script),
                    original_selection_sha256=sha(r / 'results/selection.json'),
                    original_protocol_sha256=fp['protocol_sha256'],
                    original_candidate=selection['candidate'], original_alpha=selection['chosen']['alpha'],
                    scientific_configuration_unchanged=True, source_and_weight_hashes_unchanged=True,
                    test_target_truth_used=False, test_reselection=False,
                    global_fp32_policy=policy, local_fp32_policy='none')
    require('repair_provenance_actual_hashes_and_scientific_invariance',
            all(provenance.get(k) == v for k,v in expected.items()) and
            provenance['frozen_source_fingerprints'] == fp and
            provenance['original_freeze_time'] == selection['frozen_at_utc'])
    require('repair_only_actual_alpha0_freeze', selection['candidate'] == 'chronos_usage_resource' and
            selection['chosen']['alpha'] == 0)
    require('original_strict_control_tolerance_retained', provenance['original_tolerance'] == dict(atol=2e-6, rtol=1e-6))
    require('repair_chronology_after_original_freeze', date(provenance['created_at_utc']) > date(selection['frozen_at_utc']))
    execution = dict(repair_id='repair1', implementation_sha256=sha(script), provenance_sha256=sha(provenance_path),
                     original_selection_sha256=sha(r / 'results/selection.json'), original_protocol_sha256=fp['protocol_sha256'],
                     global_fp32_policy=policy, local_fp32_policy='none')
    require('actual_test_and_deploy_execution_fingerprints', test.get('execution_fingerprint') == execution and
            deploy.get('execution_fingerprint') == execution)
    require('repair_paths_point_to_actual_provenance',
            Path(test['repair_provenance']).resolve(strict=True) == provenance_path.resolve() and
            Path(deploy['repair_provenance']).resolve(strict=True) == provenance_path.resolve())
    require('failure_evidence_retained_in_actual_summary', test.get('original_failure_evidence_preserved') is True)
    repaired_source = read(r / 'docs/repair1_source_audit.json')
    require('repair_actual_source_fingerprints_same_as_original', repaired_source['fingerprints'] == fp and
            repaired_source['source_complete'] is True and repaired_source['source_files'] == 12085 and
            repaired_source['source_data_read_only'] is True)
    for name in ['original_dev_sha256', 'original_failure_sha256']:
        records = provenance.get(name)
        require(name + '_nonempty', isinstance(records, dict) and bool(records))
        for relative, expected_sha in records.items():
            file = safe_file(r, relative, name + '_' + relative)
            require(name + '_actual_sha_' + relative, sha(file) == expected_sha)
    dev_caches = {f'results/cache/dev_{k}_origin{o}.{suffix}'
                  for k in ['timesfm_control','chronos_control','timesfm_usage','timesfm_usage_resource','chronos_usage','chronos_usage_resource']
                  for o in DEV for suffix in ['npz','json']}
    require('all_48_original_dev_caches_bound_by_provenance', dev_caches.issubset(provenance['original_dev_sha256']))
    sources = provenance.get('implementation_sources_sha256')
    require('audited_loader_backend_source_hashes_present', isinstance(sources, dict) and bool(sources))
    for value, expected_sha in sources.items():
        file = safe_file(cluster.parent, value, 'implementation_source_' + value, absolute=True)
        require('implementation_source_actual_sha_' + value, sha(file) == expected_sha)
    require('repair_original_weight_hashes', provenance['original_weights'] == weights)
    evidence = provenance.get('diagnostic_evidence')
    require('repair_actual_diagnostic_evidence_required', isinstance(evidence, list) and bool(evidence))
    diagnosis = None
    for item in evidence:
        path = safe_file(cluster.parent, item['path'], 'diagnostic_evidence', absolute=True)
        require('diagnostic_evidence_actual_sha_' + path.name, sha(path) == item['sha256'])
        value = read(path)
        if 'matching_legacy_conditions' in value:
            require('single_precision_diagnosis', diagnosis is None)
            diagnosis = value
            diagnostic_dir = path.parent
    require('actual_precision_diagnosis_present', diagnosis is not None)
    require('diagnostic_immutable_and_no_truth_scoring', diagnosis['origin'] == 22 and
            diagnosis['immutable_preserved'] is True and diagnosis['future_target_truth_scored'] is False and
            diagnosis['test_reselection'] is False and diagnosis['all_adjacent_repeats_equal'] is True and
            diagnosis['immutable_before'] == diagnosis['immutable_after'])
    for relative, expected_sha in diagnosis['immutable_before'].items():
        file = safe_file(b, relative, 'diagnostic_immutable_' + relative)
        require('diagnostic_immutable_current_sha_' + relative, sha(file) == expected_sha)
    for threads in [2,4]:
        for repeat in [1,2]:
            label = f'inherit_tf32_threads{threads}_repeat{repeat}'
            rows = [row for row in diagnosis['calls'] if row['call'] == label]
            require('required_tf32_repetition_' + label, len(rows) == 1)
            row = rows[0]
            require('tf32_exact_legacy_reproduction_' + label,
                    row['flags']['global_fp32_precision'] == 'tf32' and row['flags']['cpu_threads'] == threads and
                    row['vs_old_test_cache']['points'] == 480 and row['vs_old_test_cache']['maximum_absolute_difference'] == 0 and
                    row['vs_old_test_cache']['bitwise_values_equal'] is True and row['vs_old_test_cache']['registered_tolerance_passed'] is True)
    ieee = [row for row in diagnosis['calls'] if row['flags']['global_fp32_precision'] == 'ieee']
    require('original_ieee_failure_transparently_retained', len(ieee) >= 4 and
            all(row['vs_old_test_cache']['registered_tolerance_passed'] is False for row in ieee) and
            max(row['vs_old_test_cache']['maximum_absolute_difference'] for row in ieee) > 2e-6)
    # Check actual diagnostic arrays against the old prediction archive, never against truth.
    with np.load(b / 'round2/results/final_chronos24.npz', allow_pickle=False) as z:
        power = np.load(b / 'results/data.npz', allow_pickle=False)
        ix = np.where(power['cells'] == 'a')[0]
        old_prediction = z['zero_cellprod'][ix, 0]
        power.close()
    with np.load(diagnostic_dir / 'precision_calls.npz', allow_pickle=False) as z:
        for threads in [2,4]:
            for repeat in [1,2]:
                label = f'inherit_tf32_threads{threads}_repeat{repeat}'
                require('diagnostic_actual_prediction_array_' + label, label in z.files and
                        np.array_equal(z[label], old_prediction))
    for split in ['dev','test']:
        results = r / 'results' if split == 'dev' else r / 'results/repair1'
        regression = read(results / f'{split}_control_regression.json')
        origins = DEV if split == 'dev' else TEST
        for backbone in ['timesfm','chronos']:
            control = regression[backbone]
            require(f'{split}_{backbone}_all_control_origins_actual_regression',
                    set(control['origins']) == {str(o) for o in origins} and control['passed'] is True and
                    control['atol'] == 2e-6 and control['rtol'] == 1e-6 and
                    all(row['passed'] is True and row['atol'] == 2e-6 and row['rtol'] == 1e-6
                        for row in control['origins'].values()))
    return execution, provenance


def verify(top):
    require('absolute_D_root', top.is_absolute() and top.resolve(strict=True) == EXPECTED)
    top = top.resolve(strict=True)
    b, cluster = top / 'experiment', top / 'google-clusterdata-2019'
    r = b / 'round10'
    repaired = r / 'results/repair1'
    protocol = read(r / 'docs/protocol.json')
    registration = read(r / 'docs/registration.json')
    source = read(r / 'docs/source_audit.json')
    selection_path = r / 'results/selection.json'
    selection, dev, test = read(selection_path), read(r / 'results/dev_summary.json'), read(repaired / 'summary.json')
    deploy = read(repaired / 'deployment_verification.json')
    audit = read(cluster / 'results/complete_projection_audit.json')
    for label, proof in [('development', dev), ('test', test), ('source_audit', audit), ('deployment', deploy)]:
        passed(label, proof)
    freeze_hash = sha(selection_path)
    require('selection_sha256', freeze_hash == (r / 'results/selection.sha256').read_text(encoding='utf-8').strip())
    require('unchanged_frozen_selection', dev['selection'] == test['selection'] == selection)
    require('protocol_sha256', sha(r / 'docs/protocol.json') == registration['protocol_sha256'] == selection['protocol_sha256'])
    require('source_fingerprints_frozen', source['fingerprints'] == selection['source_fingerprints'])
    require('freeze_before_scoring_evidence', registration['before_foundation_inference'] is True and
            dev['test_scored'] is False and selection['test_scored_before_freeze'] is False and
            test['test_reselection'] is False and
            date(registration['registered_at_utc']) <= date(selection['frozen_at_utc']) <=
            date(dev['completed_at_utc']) < date(test['completed_at_utc']))
    require('declared_time_splits', protocol['development_origins'] == DEV and
            protocol['test_origins'] == TEST and protocol['deployment_origin'] == 31)
    alpha, candidate = selection['chosen']['alpha'], selection['candidate']
    require('selected_registered_candidate', candidate in protocol['candidates'] and alpha in protocol['alpha_grid'])
    require('exploratory_limits_preserved', test['independent_confirmation'] is False and
            test['large_improvement'] is False and selection['independent_confirmation'] is False)
    fp = source['fingerprints']
    power_path = b / 'results/data.npz'
    usage_path, resource_path = cluster / 'aggregated/complete_usage_a_15min.npz', cluster / 'aggregated/resources_15min.npz'
    for key, path in [('power_sha256', power_path), ('usage_sha256', usage_path), ('resource_sha256', resource_path),
                      ('audit_sha256', cluster / 'results/complete_projection_audit.json'),
                      ('script_sha256', cluster / 'code/native_covariate_experiment.py'),
                      ('protocol_sha256', r / 'docs/protocol.json')]:
        require('current_' + key, sha(path) == fp[key])
    expected_prior_artifacts = {
        'round8b/results/selection.json', 'round8b/results/selection.sha256',
        'round8b/results/dev_predictions.npz', 'round8b/results/test_predictions.npz',
        'round8b/results/selected_deployment_origin31_1day.csv',
        'round5/results/calibration_dev.npz', 'round3/results/chronos_dev.npz',
        'round3/results/tirex_dev.npz', 'round3/results/tf3_dev.npz',
        'round3/results/tf3_test.npz', 'round2/results/development.npz',
        'round2/results/final_chronos24.npz',
    }
    prior_artifacts = fp.get('prior_artifact_sha256')
    require('all_twelve_frozen_prior_artifact_fingerprints', isinstance(prior_artifacts, dict) and
            set(prior_artifacts) == expected_prior_artifacts)
    for relative, expected_sha in prior_artifacts.items():
        rel = Path(relative)
        require('prior_artifact_safe_relative_path_' + relative,
                not rel.is_absolute() and not rel.drive and '..' not in rel.parts)
        path = (b / rel).resolve(strict=True)
        require('prior_artifact_stays_inside_experiment_' + relative,
                path.is_relative_to(b.resolve(strict=True)) and path.is_file())
        require('current_prior_artifact_sha256_' + relative,
                isinstance(expected_sha, str) and re.fullmatch('[0-9a-f]{64}', expected_sha) is not None and
                sha(path) == expected_sha)
    require('complete_a_scope', source['source_complete'] is True and source['source_files'] == 12085 and
            audit['source_files'] == 12085 and audit['aggregated_file_sha256'] == sha(usage_path))
    with np.load(power_path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    with np.load(usage_path, allow_pickle=False) as z:
        require('complete_source_metadata', bool(z['complete']) and int(z['source_files']) == int(z['expected_source_files']) == 12085)
        u, names = z['aggregate'], z['names']
        require('complete_usage_schema', z['features'].tolist() == protocol['complete_usage_schema'])
    with np.load(resource_path, allow_pickle=False) as z:
        resource, rnames = z['resource'], z['names']
    ix = np.where(d['cells'] == 'a')[0]
    present = u[..., 3] > 0
    require('57_and_five_domain_source_order', d['y'].shape == (57, 31, 96) and u.shape == (57, 31, 96, 6) and
            resource.shape == (57, 31, 96, 8) and len(ix) == 5 and
            np.array_equal(names, d['names']) and np.array_equal(rnames, d['names']) and
            source['cpu_domains'] == d['names'][ix].tolist() and not present[d['cells'] != 'a'].any())
    registry = {**protocol['matched_controls'], **protocol['candidates']}
    control = protocol['candidates'][candidate]['backbone'] + '_control'
    weights = {}
    for backbone, folder in [('timesfm', 'timesfm3-base'), ('chronos', 'chronos2-base')]:
        directory = b / 'models' / folder
        weights[backbone] = dict(checkpoint_sha256=sha(directory / 'model.safetensors'),
                                 config_sha256=sha(directory / 'config.json'))
    execution, provenance = verify_repair(r, b, cluster, fp, selection, test, deploy, weights)
    native = {}
    for split, origins in [('dev', DEV), ('test', TEST), ('deploy', [31])]:
        keys = registry if split != 'deploy' else {candidate: registry[candidate], control: registry[control]}
        result_dir = r / 'results' if split == 'dev' else repaired
        cache_dir = r / 'results/cache' if split == 'dev' else r / 'results/cache/repair1'
        data = {}
        causality = read(result_dir / f'{split}_causality_checks.json')
        expected_causal = [f'future_input_invariance_{key}_origin{origin}'
                           for key in keys for origin in origins]
        require(split + '_causality_proof', bool(causality) and all(causality.values()) and
                all(causality.get(key) is True for key in expected_causal))
        for key, cfg in keys.items():
            rows = []
            for origin in origins:
                archive = cache_dir / f'{split}_{key}_origin{origin}.npz'
                meta = read(archive.with_suffix('.json'))
                target, cov, channels = input_arrays(d, u, resource, present, ix, origin, cfg)
                expected = dict(**fp, **weights[cfg['backbone']], origin=origin, split=split,
                                config=cfg, covariate_names=channels, input_sha256=input_sha(target, cov, channels),
                                target_names=d['names'][ix].tolist())
                if split != 'dev':
                    expected['execution_fingerprint'] = execution
                require(f'cache_identity_{split}_{key}_{origin}', meta['identity'] == expected and
                        meta['prediction_archive_sha256'] == sha(archive))
                if split == 'dev':
                    chronology = date(registration['registered_at_utc']) <= date(meta['predicted_at_utc']) <= date(selection['frozen_at_utc'])
                else:
                    chronology = date(meta['predicted_at_utc']) >= date(provenance['created_at_utc']) >= date(selection['frozen_at_utc'])
                require(f'cache_chronology_{split}_{key}_{origin}', chronology)
                with np.load(archive, allow_pickle=False) as z:
                    pred = z['prediction']
                require(f'cache_output_{split}_{key}_{origin}', pred.shape == (5, 96) and np.isfinite(pred).all())
                rows.append(pred)
                # Perturb all at/after-origin input values and rebuild the query.
                dd = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
                uu, rr, mm = u.copy(), resource.copy(), present.copy()
                for field in ['y', 'prod', 'valid', 'pvalid']:
                    dd[field][:, origin:] = False if dd[field].dtype == bool else 999
                uu[:, origin:], rr[:, origin:], mm[:, origin:] = 999, 999, False
                changed = input_arrays(dd, uu, rr, mm, ix, origin, cfg)
                require(f'no_future_input_{split}_{key}_{origin}', changed[2] == channels and
                        np.array_equal(target, changed[0], equal_nan=True) and np.array_equal(cov, changed[1], equal_nan=True))
            data[key] = np.stack(rows, axis=1)
        native[split] = data
        with np.load(result_dir / f'{split}_native_predictions.npz', allow_pickle=False) as z:
            require(split + '_native_archives_match_caches', set(z.files) == set(keys) and
                    all(np.array_equal(z[k], data[k]) for k in keys))
    # Independently verify saved selected arrays and frozen DEV choice.
    for split, origins, summary in [('dev', DEV, dev), ('test', TEST, test)]:
        result_dir = r / 'results' if split == 'dev' else repaired
        with np.load(result_dir / f'{split}_controlled_predictions.npz', allow_pickle=False) as z:
            saved = {key: z[key] for key in z.files}
        if split == 'test':
            with np.load(b / 'round8b/results/test_predictions.npz', allow_pickle=False) as z:
                original_prior = z['selected']
        else:
            with np.load(b / 'round5/results/calibration_dev.npz', allow_pickle=False) as z:
                residual = z['residual_ridge100000']
            with np.load(b / 'round3/results/chronos_dev.npz', allow_pickle=False) as z:
                chronos = z['native']
            with np.load(b / 'round3/results/tirex_dev.npz', allow_pickle=False) as z:
                tirex = z['native']
            with np.load(b / 'round3/results/tf3_dev.npz', allow_pickle=False) as z:
                timesfm = z['tf3_cellprod']
            anchor = np.clip(.25 * chronos + .25 * tirex + .5 * timesfm, 0, 1)
            old = .5 * anchor + .5 * residual
            original_choice = read(b / 'round8b/results/selection.json')
            original_weight = original_choice['chosen']['alpha']
            with np.load(b / 'round8b/results/dev_predictions.npz', allow_pickle=False) as z:
                original_prior = ((1 - original_weight) * old +
                                  original_weight * z[original_choice['config']])
        require(split + '_prior_exact_original_round8b', np.array_equal(saved['prior'], original_prior))
        reconstructed = corrected(saved['prior'], native[split][candidate], native[split][control], ix, alpha)
        require(split + '_selected_matches_frozen_formula', np.array_equal(saved['selected'], reconstructed))
        require(split + '_52_domains_exact_prior', np.array_equal(saved['selected'][d['cells'] != 'a'], saved['prior'][d['cells'] != 'a']))
        require(split + '_selected_finite_bounds', np.isfinite(saved['selected']).all() and saved['selected'].min() >= 0 and saved['selected'].max() <= 1)
        if alpha == 0:
            require(split + '_alpha0_whole_forecast_unchanged', np.array_equal(saved['selected'], saved['prior']))
        y, mask = d['y'][:, origins], d['valid'][:, origins]
        for key, cfg in protocol['candidates'].items():
            candidate_prediction = corrected(original_prior, native[split][key],
                                             native[split][cfg['backbone'] + '_control'], ix, 1)
            require(split + '_candidate_formula_' + key, np.array_equal(saved[key], candidate_prediction))
            for label, indices in [('all57', np.arange(57)), ('cella5', ix)]:
                require(split + '_candidate_wape_' + key + '_' + label,
                        math.isclose(wape(y[indices], saved[key][indices], mask[indices]),
                                     summary['candidate_scores'][key]['controlled_delta'][label]['wape'],
                                     rel_tol=1e-12, abs_tol=1e-12))
        for label, indices in [('all57', np.arange(57)), ('cella5', ix)]:
            require(split + '_wape_' + label, math.isclose(wape(y[indices], saved['selected'][indices], mask[indices]),
                    summary['selected_scores'][label]['wape'], rel_tol=1e-12, abs_tol=1e-12))
        if split == 'dev':
            prior = saved['prior']
            y5, m5 = y[ix, :4], mask[ix, :4]
            best = min(protocol['candidates'], key=lambda k: wape(y5,
                       corrected(prior, native['dev'][k], native['dev'][registry[k]['backbone'] + '_control'], ix, 1)[ix, :4], m5))
            require('independent_development_candidate_replay', best == candidate)
            grid = []
            for weight in protocol['alpha_grid']:
                p = corrected(prior, native['dev'][candidate], native['dev'][control], ix, weight)
                den = (np.abs(y[ix, 4:]) * mask[ix, 4:]).sum(-1).clip(1e-12)
                dif = (100 * ((np.abs(y[ix, 4:] - prior[ix, 4:]) - np.abs(y[ix, 4:] - p[ix, 4:])) * mask[ix, 4:]).sum(-1) / den).mean(0)
                grid.append((weight, wape(y[ix, 4:], p[ix, 4:], mask[ix, 4:]), float(dif.std(ddof=1) / 2)))
            win = min(grid, key=lambda x: x[1])
            picked = min(x[0] for x in grid if x[1] <= win[1] + win[2] + 1e-12)
            require('independent_development_alpha_replay', picked == alpha)
    csv = repaired / 'selected_deployment_origin31_1day.csv'
    out = pd.read_csv(csv)
    require('deployment_file_sha', sha(csv) == deploy['output_sha256'])
    require('deployment_frozen_choice', deploy['candidate'] == candidate and deploy['selection_alpha'] == alpha and
            deploy['input_end'] == '2019-05-31' and deploy['forecast_start'] == '2019-06-01' and
            deploy['future_truth_available'] is False and deploy['not_live_forecast'] is True)
    require('5472_rows_57x96_unique', len(out) == 5472 and out.series.nunique() == 57 and
            out.groupby('series').size().eq(96).all() and not out.duplicated(['series', 'local_trace_time_PT']).any())
    slots = pd.date_range('2019-06-01 00:00', periods=96, freq='15min')
    require('complete_nextday_time_grid', all(pd.DatetimeIndex(pd.to_datetime(g.local_trace_time_PT)).sort_values().equals(slots)
            for _, g in out.groupby('series')))
    require('deployment_domain_order', np.array_equal(out.series.to_numpy(), np.repeat(d['names'], 96)))
    vals = out[['predicted_util', 'native_covariate_candidate_util', 'prior_round8b_util', 'lower80_empirical', 'upper80_empirical']].to_numpy()
    require('finite_bounded_deployment_and_interval', np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all())
    require('interval_contains_point', (out.lower80_empirical <= out.predicted_util).all() and (out.predicted_util <= out.upper80_empirical).all())
    old = pd.read_csv(b / 'round8b/results/selected_deployment_origin31_1day.csv')
    require('deployment_prior_matches_original_keys', out[['series', 'cell', 'local_trace_time_PT']].equals(old[['series', 'cell', 'local_trace_time_PT']]))
    require('deployment_frozen_prior_matches_round8b', np.array_equal(out.prior_round8b_util.to_numpy(), old.predicted_util.to_numpy()))
    prior = old.predicted_util.to_numpy().reshape(57, 1, 96)
    reconstruction = corrected(prior, native['deploy'][candidate], native['deploy'][control], ix, alpha).ravel()
    require('deployment_matches_frozen_native_formula', np.allclose(out.predicted_util, reconstruction, rtol=1e-15, atol=1e-15))
    amask = out.cell.eq('a').to_numpy()
    require('deployment_other52_unchanged', np.array_equal(out.predicted_util.to_numpy()[~amask], old.predicted_util.to_numpy()[~amask]))
    if alpha == 0:
        require('deployment_alpha0_whole_model_unchanged', np.array_equal(out.predicted_util.to_numpy(), old.predicted_util.to_numpy()))
    for column, index in [('history_cpu_mean7_NCU', 0), ('history_memory_mean7_normalized', 1)]:
        expected = np.full(57, np.nan)
        expected[ix] = u[ix, 24:31, :, index].mean((1, 2))
        actual = out[column].to_numpy()
        require(column + '_only_observed_seven_day_mean', np.allclose(actual, np.repeat(expected, 96), rtol=1e-14, atol=1e-14, equal_nan=True))
        require(column + '_five_domains_finite_other52_missing', np.isfinite(actual[amask]).all() and (actual[amask] >= 0).all() and np.isnan(actual[~amask]).all())
    require('cpu_history_masks', np.array_equal(out.has_complete_cpu_history.to_numpy(), amask) and
            np.array_equal(out.history_cpu_missing.to_numpy(), ~amask))
    require('cpu_is_history_not_future_prediction', 'Observed historical seven-day mean' in deploy['cpu_columns'])
    migration_dir = cluster / 'results/migration_to_d'
    migration, compatibility = read(migration_dir / 'migration_verification.json'), read(migration_dir / 'migration_compatibility.json')
    passed('whole_tree_migration', migration)
    passed('migration_compatibility', compatibility)
    require('migration_correct_roots', Path(migration['source']) == DESKTOP and Path(migration['destination']) == top and
            migration['all_names_lengths_and_sha256_equal'] is True)
    hashes = migration_dir / 'all_files_sha256.jsonl'
    require('migration_full_hash_evidence_sha', sha(hashes) == migration['evidence_sha256'])
    count, total, seen = 0, 0, set()
    with hashes.open(encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            rel = Path(record['file'])
            if rel.is_absolute() or '..' in rel.parts or record['file'] in seen or not re.fullmatch('[0-9a-f]{64}', record['sha256']):
                raise RuntimeError('Invalid/duplicate migration hash record')
            if not (top / rel).is_file():
                raise RuntimeError('Migrated original file missing: ' + str(rel))
            seen.add(record['file']); count += 1; total += record['bytes']
    require('migration_all_file_hashes_count_and_bytes', count == migration['files'] > 0 and total == migration['bytes'] > 0)
    require('actual_desktop_junction_to_D', DESKTOP.is_junction() and DESKTOP.resolve(strict=True) == top)
    keys = ['desktop_junction', 'real_root_D', 'python_runs_from_D', 'python_runs_via_old_alias',
            'torch_cuda_available', 'data_source_hashes_equal', 'source_shape_names_equal']
    require('migration_runtime_compatibility_proof', all(compatibility['checks'].get(k) is True for k in keys))
    require('old_alias_current_power_hash', sha(DESKTOP / 'experiment/results/data.npz') == fp['power_sha256'])
    qa_dir = r / 'docs/report_qa'
    qa = read(qa_dir / 'verification.json')
    passed('final_word_qa', qa)
    docx = top / TITLE
    require('final_docx_sha', sha(docx) == qa['document_sha256'])
    pages = qa['pages']
    require('all_pages_actually_reviewed_proof', isinstance(pages, int) and pages > 0 and
            qa['all_pages_visually_reviewed'] is True and qa['layout_passed'] is True and
            qa['page_numbers_reviewed'] == list(range(1, pages + 1)))
    require('all_page_image_hashes', len(qa['images']) == pages and
            all(sha(qa_dir / row['file']) == row['sha256'] for row in qa['images']))
    pdf = Path(qa['pdf'])
    require('final_pdf_evidence_sha', pdf.resolve(strict=True).parent == qa_dir.resolve() and sha(pdf) == qa['pdf_sha256'])
    from pypdf import PdfReader
    require('pdf_actual_page_count', len(PdfReader(pdf).pages) == pages)
    require('final_qa_after_test', date(qa['reviewed_utc']) >= date(test['completed_at_utc']))
    return dict(frozen_wape=test['selected_scores']['all57']['wape'], new_weight=alpha,
                selected_candidate=candidate, selection_sha256=freeze_hash,
                report=str(docx), report_pages=pages, report_sha256=qa['document_sha256'],
                migration_files=count, migration_bytes=total,
                migration_evidence_scope='Historical whole-tree SHA256 equality verified by the retained manifest/proof; current sources and native caches independently rehashed. Later reports/entries may differ from the move snapshot.',
                execution_fingerprint=execution, original_failure_evidence_preserved=True,
                cpu_scope='Five cell-a domains; deployment CPU/memory columns contain historical means only.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    details, failure = {}, None
    try:
        details = verify(args.root)
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    passed_all = failure is None and bool(CHECKS) and all(CHECKS.values())
    print(json.dumps(dict(verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
          read_only=True, checks=CHECKS, all_passed=passed_all, failure=failure,
          details=details), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if passed_all else 1


if __name__ == '__main__':
    sys.exit(main())
