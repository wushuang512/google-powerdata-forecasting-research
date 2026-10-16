"""Frozen native historical CPU/memory covariates, around the round8b prior.

Deploy this file and its protocol next to existing cluster-data code. The root
uses the resolved real script path, so the complete research folder can reside
on D and the previous Desktop path can remain a compatibility junction. All
outputs remain within that research root's experiment/round10. --data-root can
override source discovery only; no source data is rewritten.
The dev phase never computes a test metric; no backbone is instantiated at import.
"""
import argparse
import ctypes
import datetime as dt
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd

GIB = 2 ** 30
DEV = list(range(14, 22))
TEST = list(range(22, 31))
HORIZON = 96
CONTEXT = 1344
ROOT = Path(__file__).resolve().parents[1]
TOP = ROOT.parent
B = TOP / "experiment"
R = B / "round10"
TEMPLATE = Path(__file__).with_name("native_covariate_protocol.json")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **arrays)
    temporary.replace(path)


def register():
    for folder in ("docs", "results", "models", "results/cache"):
        (R / folder).mkdir(parents=True, exist_ok=True)
    protocol = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    stored = R / "docs/protocol.json"
    if stored.exists():
        if json.loads(stored.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Registered protocol differs: do not silently change this round")
    else:
        save_json(stored, protocol)
        save_json(R / "docs/registration.json", dict(registered_at_utc=utc(),
                  protocol_sha256=digest(stored), before_foundation_inference=True))
    return protocol


def find_source(default, data_root, relatives):
    candidates = []
    if data_root:
        candidates.extend(Path(data_root) / name for name in relatives)
    candidates.append(Path(default))
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError("Source not found: " + str(candidates))


def load_sources(data_root=None):
    power = find_source(B / "results/data.npz", data_root,
                        ["experiment/results/data.npz", "data.npz"])
    usage = find_source(ROOT / "aggregated/complete_usage_a_15min.npz", data_root,
                        ["google-clusterdata-2019/aggregated/complete_usage_a_15min.npz",
                         "aggregated/complete_usage_a_15min.npz", "complete_usage_a_15min.npz"])
    resources = find_source(ROOT / "aggregated/resources_15min.npz", data_root,
                        ["google-clusterdata-2019/aggregated/resources_15min.npz",
                         "aggregated/resources_15min.npz", "resources_15min.npz"])
    with np.load(power, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    with np.load(usage, allow_pickle=False) as z:
        if not bool(z["complete"]) or int(z["source_files"]) != 12085 or int(z["expected_source_files"]) != 12085:
            raise RuntimeError("Require complete audited 12085-file cell-a projection")
        u, unames, ufeatures = z["aggregate"], z["names"], z["features"].tolist()
    with np.load(resources, allow_pickle=False) as z:
        rr, rnames, rfeatures = z["resource"], z["names"], z["features"].tolist()
    if not np.array_equal(unames, d["names"]) or not np.array_equal(rnames, d["names"]):
        raise RuntimeError("Source domain ordering differs")
    protocol = json.loads((R / "docs/protocol.json").read_text(encoding="utf-8"))
    if ufeatures != protocol["complete_usage_schema"]:
        raise RuntimeError("Usage schema differs; third column is job-window equivalent, not variance")
    if rr.shape != (57, 31, 96, 8) or u.shape != (57, 31, 96, 6):
        raise RuntimeError("Unexpected aggregation shape")
    if rfeatures[1] != "active_fraction" or rfeatures[2] != "mean_cpu_capacity_NCU":
        raise RuntimeError("Resource column definitions differ")
    audit = json.loads((ROOT / "results/complete_projection_audit.json").read_text(encoding="utf-8"))
    if not audit.get("all_passed"):
        raise RuntimeError("Complete source audit must have passed")
    ix = np.where(d["cells"] == "a")[0]
    if len(ix) != 5 or d["y"].shape != (57, 31, 96):
        raise RuntimeError("Require exactly five cell-a domains and 57-domain month")
    present = u[..., 3] > 0
    if present[d["cells"] != "a"].any():
        raise RuntimeError("Task-usage source unexpectedly includes other cells")
    # Bind every existing archive read by previous() / reference_control() and
    # the prior June1 export. A changed prior must invalidate cached corrections
    # even when the raw inputs and foundation weights themselves are unchanged.
    legacy_files = [
        "round8b/results/selection.json", "round8b/results/selection.sha256",
        "round8b/results/dev_predictions.npz", "round8b/results/test_predictions.npz",
        "round8b/results/selected_deployment_origin31_1day.csv",
        "round5/results/calibration_dev.npz",
        "round3/results/chronos_dev.npz", "round3/results/tirex_dev.npz",
        "round3/results/tf3_dev.npz", "round3/results/tf3_test.npz",
        "round2/results/development.npz", "round2/results/final_chronos24.npz",
    ]
    prior_selection = B / "round8b/results/selection.json"
    if digest(prior_selection) != (B / "round8b/results/selection.sha256").read_text().strip():
        raise RuntimeError("Prior frozen selection hash differs before source registration")
    legacy_hashes = {name: digest(B / name) for name in legacy_files}
    fingerprints = dict(protocol_sha256=digest(R / "docs/protocol.json"),
                        script_sha256=digest(__file__),
                        power_sha256=digest(power), usage_sha256=digest(usage),
                        resource_sha256=digest(resources),
                        audit_sha256=digest(ROOT / "results/complete_projection_audit.json"),
                        prior_artifact_sha256=legacy_hashes)
    save_json(R / "docs/source_audit.json", dict(checked_at_utc=utc(),
              paths=dict(power=str(power), usage=str(usage), resources=str(resources)),
              fingerprints=fingerprints, source_files=12085, source_complete=True,
              cpu_domains=d["names"][ix].tolist(), usage_features=ufeatures,
              resource_features=rfeatures, source_data_read_only=True))
    return d, dict(u=u, resource=rr, present=present, ix=ix), fingerprints


def input_arrays(d, v, origin, cfg):
    end = origin * 96
    start = max(0, end - CONTEXT)
    ix = v["ix"]
    y = d["y"].reshape(57, -1)[ix, start:end].astype("float32")
    proxy = np.where(d["pvalid"].reshape(57, -1)[:, start:end],
                     d["prod"].reshape(57, -1)[:, start:end], np.nan).astype("float32")
    names, covs = [], []
    for i in ix:
        if np.isfinite(proxy[i]).mean() >= .2:
            names.append(f"production_{i}")
            covs.append(proxy[i])
    if cfg["usage"]:
        usage = v["u"].reshape(57, -1, 6)
        mask = v["present"].reshape(57, -1)
        for i in ix:
            for j, name in ((0, "completed_cpu_NCU"), (1, "completed_memory_normalized")):
                a = np.where(mask[i, start:end], usage[i, start:end, j], np.nan).astype("float32")
                if not np.isfinite(a).any() or (a[np.isfinite(a)] < 0).any():
                    raise RuntimeError("No finite nonnegative historical usage channel")
                covs.append(a)
                names.append(f"{name}_{i}")
    if cfg["resources"]:
        rr = v["resource"].reshape(57, -1, 8)
        for i in ix:
            for j, name in ((1, "active_fraction"), (2, "mean_cpu_capacity_NCU")):
                covs.append(rr[i, start:end, j].astype("float32"))
                names.append(f"{name}_{i}")
    cov = np.stack(covs) if covs else np.empty((0, end - start), dtype="float32")
    if len(ix) + len(cov) > 30 or y.shape != (5, min(CONTEXT, end)):
        raise RuntimeError("Query size exceeds registered schema")
    if np.isinf(y).any() or np.isinf(cov).any() or not np.isfinite(y).all():
        raise RuntimeError("Invalid historical input")
    return y, cov, names


def input_digest(y, cov, names):
    h = hashlib.sha256(json.dumps(names).encode())
    for a in (y, cov):
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def causal_checks(d, v, registry, origins):
    result = {}
    for o in origins:
        # Every modified element is at or after the forecast boundary.
        dd = {k: (a.copy() if isinstance(a, np.ndarray) else a) for k, a in d.items()}
        vv = {k: (a.copy() if isinstance(a, np.ndarray) else a) for k, a in v.items()}
        for k in ("y", "prod", "valid", "pvalid"):
            dd[k][:, o:] = 999 if dd[k].dtype != bool else False
        vv["u"][:, o:] = 999
        vv["resource"][:, o:] = 999
        vv["present"][:, o:] = False
        for key, cfg in registry.items():
            a = input_arrays(d, v, o, cfg)
            b = input_arrays(dd, vv, o, cfg)
            result[f"future_input_invariance_{key}_origin{o}"] = bool(
                a[2] == b[2] and np.array_equal(a[0], b[0], equal_nan=True)
                and np.array_equal(a[1], b[1], equal_nan=True))
    if not all(result.values()):
        raise RuntimeError("Future-input perturbation changed the historical query")
    return result


def memory_state():
    if os.name != "nt":
        raise RuntimeError("This desktop protocol requires Windows memory checks")
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MemoryStatus()
    m.dwLength = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise RuntimeError("Cannot inspect physical RAM and commit reserve")
    return dict(available_physical_bytes=int(m.ullAvailPhys),
                available_commit_bytes=int(m.ullAvailPageFile),
                memory_load_percent=int(m.dwMemoryLoad))


def resource_check(protocol, label, before_load=False):
    state = memory_state()
    policy = protocol["resources"]
    suffix = "before_load" if before_load else "between_predictions"
    physical = policy[f"minimum_available_physical_gib_{suffix}"]
    commit = policy[f"minimum_available_commit_gib_{suffix}"]
    state.update(at_utc=utc(), label=label, thresholds_gib=dict(physical=physical, commit=commit))
    state["passed"] = (state["available_physical_bytes"] >= physical * GIB
                       and state["available_commit_bytes"] >= commit * GIB)
    save_json(R / "results/resource_guard_status.json", state)
    if not state["passed"]:
        raise MemoryError("Research paused before next operation; existing caches retained. "
                          "Need available RAM %.2f GiB and commit %.2f GiB" % (physical, commit))
    if shutil.disk_usage(R).free < GIB:
        raise RuntimeError("Require one GiB research-output disk reserve")
    return state


def reference_control(backbone, split, ix):
    if split == "deploy":
        return None
    if backbone == "timesfm":
        path, key = B / f"round3/results/tf3_{split}.npz", "tf3_cellprod"
    else:
        path = B / "round2/results" / ("development.npz" if split == "dev" else "final_chronos24.npz")
        key = "zero_cellprod"
    with np.load(path, allow_pickle=False) as z:
        return z[key][ix]


def validate_cache(cache, expected):
    if not cache.is_file() or not cache.with_suffix(".json").is_file():
        return None
    meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("identity") != expected or meta.get("prediction_archive_sha256") != digest(cache):
        raise RuntimeError("Cached prediction identity differs: " + str(cache))
    with np.load(cache, allow_pickle=False) as z:
        p = z["prediction"]
    if p.shape != (5, 96) or not np.isfinite(p).all():
        raise RuntimeError("Invalid cached native prediction")
    return p


def predict_backbones(d, v, fingerprints, protocol, split, origins):
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refuse slow high-RAM CPU fallback")
    registry = {**protocol["matched_controls"], **protocol["candidates"]}
    checks = causal_checks(d, v, registry, origins)
    predictions, regressions = {}, {}
    for backbone in ("timesfm", "chronos"):
        keys = [k for k, c in registry.items() if c["backbone"] == backbone]
        ref = reference_control(backbone, split, v["ix"])
        origin_index = {o: j for j, o in enumerate(origins)}

        def check_control_origin(key, origin, prediction):
            if ref is None or key != f"{backbone}_control":
                return
            expected = ref[:, origin_index[origin]]
            passed = bool(np.allclose(prediction, expected, atol=2e-6, rtol=1e-6))
            row = dict(maximum_absolute_difference=float(np.abs(prediction - expected).max()),
                       atol=2e-6, rtol=1e-6, passed=passed)
            evidence = regressions.setdefault(backbone, dict(origins={}, atol=2e-6, rtol=1e-6))
            evidence["origins"][str(origin)] = row
            evidence["maximum_absolute_difference"] = max(
                r["maximum_absolute_difference"] for r in evidence["origins"].values())
            evidence["passed"] = all(r["passed"] for r in evidence["origins"].values())
            checks[f"matched_control_{backbone}_{split}_origin{origin}"] = passed
            save_json(R / f"results/{split}_control_regression.json", regressions)
            if not passed:
                save_json(R / f"results/{split}_causality_checks.json", checks)
                raise RuntimeError("Matched control differs at origin %s; cache and regression evidence retained" % origin)

        directory = B / "models" / ("timesfm3-base" if backbone == "timesfm" else "chronos2-base")
        model_id = dict(checkpoint_sha256=digest(directory / "model.safetensors"),
                        config_sha256=digest(directory / "config.json"))
        todo, rows = [], {}
        for key in keys:
            for o in origins:
                yy, cov, names = input_arrays(d, v, o, registry[key])
                identity = dict(**fingerprints, **model_id, origin=o, split=split,
                                config=registry[key], covariate_names=names,
                                input_sha256=input_digest(yy, cov, names),
                                target_names=d["names"][v["ix"]].tolist())
                path = R / "results/cache" / f"{split}_{key}_origin{o}.npz"
                pred = validate_cache(path, identity)
                if pred is None:
                    todo.append((key, o, yy, cov, names, identity, path))
                else:
                    rows[key, o] = pred
                    # A previously completed control gets the same early gate.
                    check_control_origin(key, o, pred)
        model = None
        if todo:
            resource_check(protocol, f"load_{backbone}", before_load=True)
            free, total = torch.cuda.mem_get_info()
            if free < protocol["resources"]["minimum_available_cuda_gib_before_load"] * GIB:
                raise MemoryError("Insufficient free CUDA memory for sequential foundation inference")
            torch.cuda.reset_peak_memory_stats()
            try:
                if backbone == "timesfm":
                    from low_memory_timesfm import load_forecaster
                    model, load_audit = load_forecaster(directory, batch=1)
                    save_json(R / f"results/{split}_timesfm_load_audit.json", load_audit)
                    if not load_audit.get("all_passed"):
                        raise RuntimeError("Low-memory foundation loading validation failed")
                else:
                    from chronos import Chronos2Pipeline
                    model = Chronos2Pipeline.from_pretrained(str(directory), device_map="cuda",
                                                            local_files_only=True)
                torch.set_num_threads(2)
                for key, o, yy, cov, names, identity, path in todo:
                    resource_check(protocol, f"predict_{key}_origin{o}")
                    started = time.perf_counter()
                    with torch.inference_mode():
                        if backbone == "timesfm":
                            pred = np.asarray(model.predict(yy, horizon=96,
                                              past_only_covariates=cov if len(cov) else None,
                                              use_symmetric_averaging=False).forecast)
                        else:
                            inp = {"target": yy, "past_covariates": dict(zip(names, cov))}
                            quant, _ = model.predict_quantiles([inp], prediction_length=96,
                                       quantile_levels=[.5], context_length=1344,
                                       batch_size=32, cross_learning=False)
                            pred = quant[0][:, :, 0].float().cpu().numpy()
                            del quant
                    if pred.shape != (5, 96) or not np.isfinite(pred).all():
                        raise RuntimeError("Invalid native output shape/values")
                    save_npz(path, prediction=pred)
                    save_json(path.with_suffix(".json"), dict(identity=identity,
                              prediction_archive_sha256=digest(path), predicted_at_utc=utc(),
                              seconds=time.perf_counter() - started,
                              cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                              cuda_allocated_bytes=int(torch.cuda.memory_allocated())))
                    rows[key, o] = pred
                    # Both fresh and resumed controls must match immediately;
                    # never spend the remaining candidate calls after a mismatch.
                    check_control_origin(key, o, pred)
                    print("NATIVE_PREDICTED", split, key, o, flush=True)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
        for key in keys:
            predictions[key] = np.stack([rows[key, o] for o in origins], axis=1)
        if ref is not None:
            actual = predictions[f"{backbone}_control"]
            passed = bool(np.allclose(actual, ref, atol=2e-6, rtol=1e-6))
            regressions[backbone].update(maximum_absolute_difference=float(np.abs(actual - ref).max()),
                                        atol=2e-6, rtol=1e-6, passed=passed)
            checks[f"matched_control_{backbone}_{split}"] = passed
            if not passed:
                save_json(R / f"results/{split}_control_regression.json", regressions)
                raise RuntimeError("Matched control does not reproduce existing cache; do not score")
    save_npz(R / f"results/{split}_native_predictions.npz", **predictions)
    save_json(R / f"results/{split}_control_regression.json", regressions)
    save_json(R / f"results/{split}_causality_checks.json", checks)
    return predictions, checks


def previous(split):
    if split == "dev":
        with np.load(B / "round5/results/calibration_dev.npz") as z:
            residual = z["residual_ridge100000"]
        with np.load(B / "round3/results/chronos_dev.npz") as z:
            chronos = z["native"]
        with np.load(B / "round3/results/tirex_dev.npz") as z:
            tirex = z["native"]
        with np.load(B / "round3/results/tf3_dev.npz") as z:
            tf3 = z["tf3_cellprod"]
        anchor = np.clip(.25 * chronos + .25 * tirex + .5 * tf3, 0, 1)
        old = .5 * anchor + .5 * residual
        path = B / "round8b/results/selection.json"
        if digest(path) != (path.with_name("selection.sha256")).read_text().strip():
            raise RuntimeError("Prior selection hash differs")
        s = json.loads(path.read_text())
        with np.load(B / "round8b/results/dev_predictions.npz") as z:
            return (1 - s["chosen"]["alpha"]) * old + s["chosen"]["alpha"] * z[s["config"]]
    if split == "test":
        with np.load(B / "round8b/results/test_predictions.npz") as z:
            return z["selected"]
    raise ValueError(split)


def truth(d, origins):
    return d["y"][:, origins], d["valid"][:, origins]


def metrics(y, p, mask):
    if y.shape != p.shape or y.shape != mask.shape or not np.isfinite(p).all():
        raise RuntimeError("Metric arrays differ or contain invalid forecasts")
    error = np.abs(y - p)
    denominator = (np.abs(y) * mask).sum((1, 2))
    if (denominator <= 0).any() or not mask.any():
        raise RuntimeError("No valid scoring denominator")
    per = 100 * (error * mask).sum((1, 2)) / denominator
    count = mask.sum(-1)
    ya = (y * mask).sum(-1) / np.maximum(count, 1)
    pa = (p * mask).sum(-1) / np.maximum(count, 1)
    dv = count > 0
    daily = np.mean(100 * (np.abs(ya - pa) * dv).sum(1) / (np.abs(ya) * dv).sum(1))
    ym, pm = np.where(mask, y, -np.inf).max(-1), np.where(mask, p, -np.inf).max(-1)
    peak = 100 * np.mean((np.abs(ym - pm) / np.maximum(ym, 1e-8))[dv])
    return dict(wape=float(per.mean()), mae=float(error[mask].mean()),
                p95_ape=float(np.percentile(100 * error[mask] / np.maximum(np.abs(y[mask]), 1e-8), 95)),
                daily_wape=float(daily), peak_mape=float(peak), valid_points=int(mask.sum()))


def corrected(prior, native, control, ix, alpha=1):
    p = prior.copy()
    p[ix] = np.clip(prior[ix] + alpha * (native - control), 0, 1)
    return p


def scoped_scores(y, p, mask, ix):
    return dict(all57=metrics(y, p, mask), cella5=metrics(y[ix], p[ix], mask[ix]))


def daily_improvement(y, p, ref, mask, ix):
    den = (np.abs(y[ix]) * mask[ix]).sum(-1).clip(1e-12)
    candidate = 100 * (np.abs(y[ix] - p[ix]) * mask[ix]).sum(-1) / den
    baseline = 100 * (np.abs(y[ix] - ref[ix]) * mask[ix]).sum(-1) / den
    return (baseline - candidate).mean(0)


def freeze_selection(protocol, fingerprints, native, prior, d, v):
    y, mask = truth(d, DEV)
    early = []
    for key, cfg in protocol["candidates"].items():
        candidate = corrected(prior, native[key], native[cfg["backbone"] + "_control"], v["ix"])
        score = scoped_scores(y[:, :4], candidate[:, :4], mask[:, :4], v["ix"])
        early.append(dict(candidate=key, scores=score))
    best = min(early, key=lambda a: a["scores"]["cella5"]["wape"])["candidate"]
    cfg = protocol["candidates"][best]
    alphas = []
    for alpha in protocol["alpha_grid"]:
        p = corrected(prior, native[best], native[cfg["backbone"] + "_control"], v["ix"], alpha)
        differences = daily_improvement(y[:, 4:], p[:, 4:], prior[:, 4:], mask[:, 4:], v["ix"])
        score = scoped_scores(y[:, 4:], p[:, 4:], mask[:, 4:], v["ix"])
        alphas.append(dict(alpha=alpha, primary_wape=score["cella5"]["wape"],
                           scores=score, paired_day_se=float(differences.std(ddof=1) / np.sqrt(4)),
                           paired_day_improvement_pp=differences.tolist()))
    win = min(alphas, key=lambda a: a["primary_wape"])
    picked = min((a for a in alphas if a["primary_wape"] <= win["primary_wape"] + win["paired_day_se"] + 1e-12),
                 key=lambda a: a["alpha"])
    selection = dict(frozen_at_utc=utc(), protocol_sha256=fingerprints["protocol_sha256"],
                     source_fingerprints=fingerprints, candidate=best, chosen=picked,
                     early_development=early, late_development=alphas,
                     test_scored_before_freeze=False, independent_confirmation=False,
                     retained_previous_best=picked["alpha"] == 0)
    path = R / "results/selection.json"
    if path.exists():
        original = read_selection(fingerprints)
        if original["candidate"] != best or original["chosen"]["alpha"] != picked["alpha"]:
            raise RuntimeError("Recomputed development choice differs; preserve existing freeze and investigate")
        save_json(R / "results/selection_replay_audit.json", dict(recomputed=selection,
                  original_selection_sha256=digest(path), preserved_original=True))
        return original
    save_json(path, selection)
    path.with_name("selection.sha256").write_text(digest(path) + "\n", encoding="utf-8")
    return selection


def read_selection(fingerprints):
    path = R / "results/selection.json"
    if digest(path) != (path.with_name("selection.sha256")).read_text().strip():
        raise RuntimeError("Frozen selection hash differs")
    selection = json.loads(path.read_text())
    if selection["source_fingerprints"] != fingerprints:
        raise RuntimeError("Frozen source, protocol or implementation hash differs")
    return selection


def calibration(d, selected, origins):
    y, mask = truth(d, origins)
    scale = np.stack([d["y"][:, o - 7:o].mean((1, 2)).clip(.05) for o in origins], axis=1)
    normalized = np.abs(y - selected) / scale[:, :, None]
    width = float(np.quantile(normalized[mask], .8))
    half = width * scale[:, :, None]
    coverage = float(((y >= np.maximum(selected - half, 0)) &
                      (y <= np.minimum(selected + half, 1)))[mask].mean() * 100)
    return dict(relative_halfwidth=width, nominal_coverage_percent=80,
                development_calibration_coverage_percent=coverage,
                calibration_points=int(mask.sum()), calibration_origins=origins,
                scope="Pooled development residual calibration for selected predictor only; not independent coverage")


def phase_dev(protocol, d, v, fingerprints):
    native, checks = predict_backbones(d, v, fingerprints, protocol, "dev", DEV)
    prior = previous("dev")
    selection = freeze_selection(protocol, fingerprints, native, prior, d, v)
    y, mask = truth(d, DEV)
    scores, arrays, rows = {}, {}, []
    for key, cfg in protocol["candidates"].items():
        candidate = corrected(prior, native[key], native[cfg["backbone"] + "_control"], v["ix"])
        scores[key] = dict(raw_native_cella5=metrics(y[v["ix"]], native[key], mask[v["ix"]]),
                          controlled_delta=scoped_scores(y, candidate, mask, v["ix"]))
        arrays[key] = candidate
        rows.append(dict(candidate=key, **scores[key]["controlled_delta"]["cella5"],
                         all57_wape=scores[key]["controlled_delta"]["all57"]["wape"],
                         raw_native_cella5_wape=scores[key]["raw_native_cella5"]["wape"]))
    for key in protocol["matched_controls"]:
        scores[key] = dict(raw_native_cella5=metrics(y[v["ix"]], native[key], mask[v["ix"]]))
    cfg = protocol["candidates"][selection["candidate"]]
    selected = corrected(prior, native[selection["candidate"]], native[cfg["backbone"] + "_control"],
                         v["ix"], selection["chosen"]["alpha"])
    checks["other52_exact_prior"] = bool(np.array_equal(selected[d["cells"] != "a"], prior[d["cells"] != "a"]))
    checks["finite_bounded_selected"] = bool(np.isfinite(selected).all() and selected.min() >= 0 and selected.max() <= 1)
    uncertainty = calibration(d, selected, DEV)
    save_npz(R / "results/dev_controlled_predictions.npz", **arrays, selected=selected, prior=prior)
    pd.DataFrame(rows).to_csv(R / "results/dev_metrics.csv", index=False)
    summary = dict(completed_at_utc=utc(), selection=selection, candidate_scores=scores,
                   selected_scores=scoped_scores(y, selected, mask, v["ix"]),
                   prior_scores=scoped_scores(y, prior, mask, v["ix"]), uncertainty=uncertainty,
                   checks=checks, all_passed=all(checks.values()), test_scored=False,
                   independent_confirmation=False)
    save_json(R / "results/dev_summary.json", summary)
    if not summary["all_passed"]:
        raise RuntimeError("Development validation failed")
    print("DEV_FROZEN", selection["candidate"], selection["chosen"]["alpha"], flush=True)


def test_comparison(y, selected, prior, mask, ix):
    rng = np.random.default_rng(20261001)
    samples, five = [], []
    # Resample precomputed masked daily sums; do not recompute percentile/peak
    # metrics thousands of times or materialize large bootstrap forecast arrays.
    denominators = (np.abs(y) * mask).sum(-1)
    old_errors = (np.abs(y - prior) * mask).sum(-1)
    new_errors = (np.abs(y - selected) * mask).sum(-1)
    for _ in range(2000):
        days = rng.integers(0, len(TEST), len(TEST))
        den = denominators[:, days].sum(1)
        gain = 100 * (old_errors[:, days].sum(1) - new_errors[:, days].sum(1)) / den
        samples.append(float(gain.mean()))
        five.append(float(gain[ix].mean()))
    return dict(all57_improvement_pp=metrics(y, prior, mask)["wape"] - metrics(y, selected, mask)["wape"],
                cella5_improvement_pp=metrics(y[ix], prior[ix], mask[ix])["wape"] - metrics(y[ix], selected[ix], mask[ix])["wape"],
                all57_paired_date_bootstrap95_pp=np.quantile(samples, [.025, .975]).tolist(),
                cella5_paired_date_bootstrap95_pp=np.quantile(five, [.025, .975]).tolist(),
                cella5_each_day_improvement_pp=daily_improvement(y, selected, prior, mask, ix).tolist(),
                bootstrap_units="Nine forecast dates; all domains resampled together. Exploratory, already-reused test.")


def phase_test(protocol, d, v, fingerprints):
    selection = read_selection(fingerprints)
    dev = json.loads((R / "results/dev_summary.json").read_text(encoding="utf-8"))
    if not dev["all_passed"]:
        raise RuntimeError("Require passed development phase before test scoring")
    frozen_hash = digest(R / "results/selection.json")
    native, checks = predict_backbones(d, v, fingerprints, protocol, "test", TEST)
    # Test truth becomes an evaluation input only after the freeze was verified.
    y, mask = truth(d, TEST)
    prior = previous("test")
    scores, arrays, rows = {}, {}, []
    for key, cfg in protocol["candidates"].items():
        p = corrected(prior, native[key], native[cfg["backbone"] + "_control"], v["ix"])
        scores[key] = dict(raw_native_cella5=metrics(y[v["ix"]], native[key], mask[v["ix"]]),
                          controlled_delta=scoped_scores(y, p, mask, v["ix"]))
        arrays[key] = p
        rows.append(dict(candidate=key, **scores[key]["controlled_delta"]["cella5"],
                         all57_wape=scores[key]["controlled_delta"]["all57"]["wape"],
                         raw_native_cella5_wape=scores[key]["raw_native_cella5"]["wape"]))
    for key in protocol["matched_controls"]:
        scores[key] = dict(raw_native_cella5=metrics(y[v["ix"]], native[key], mask[v["ix"]]))
    cfg = protocol["candidates"][selection["candidate"]]
    selected = corrected(prior, native[selection["candidate"]], native[cfg["backbone"] + "_control"],
                         v["ix"], selection["chosen"]["alpha"])
    checks.update(selection_hash_preserved=digest(R / "results/selection.json") == frozen_hash,
                  other52_exact_prior=bool(np.array_equal(selected[d["cells"] != "a"], prior[d["cells"] != "a"])),
                  finite_bounded_selected=bool(np.isfinite(selected).all() and selected.min() >= 0 and selected.max() <= 1))
    if selection["chosen"]["alpha"] == 0:
        checks["alpha_zero_exact_prior"] = bool(np.array_equal(selected, prior))
    compare = test_comparison(y, selected, prior, mask, v["ix"])
    width = dev["uncertainty"]["relative_halfwidth"]
    scale = np.stack([d["y"][:, o-7:o].mean((1, 2)).clip(.05) for o in TEST], axis=1)
    half = width * scale[:, :, None]
    uncertainty = dict(dev["uncertainty"], test_empirical_coverage_percent=float(
        ((y >= np.maximum(selected - half, 0)) & (y <= np.minimum(selected + half, 1)))[mask].mean() * 100))
    save_npz(R / "results/test_controlled_predictions.npz", **arrays, selected=selected, prior=prior)
    pd.DataFrame(rows).to_csv(R / "results/test_metrics.csv", index=False)
    summary = dict(completed_at_utc=utc(), selection=selection, candidate_scores=scores,
                   selected_scores=scoped_scores(y, selected, mask, v["ix"]),
                   prior_scores=scoped_scores(y, prior, mask, v["ix"]), comparison=compare,
                   uncertainty=uncertainty, checks=checks, all_passed=all(checks.values()),
                   independent_confirmation=False, test_reselection=False,
                   large_improvement=False,
                   conclusion="Frozen development choice evaluated; no test-rank selection. New independent dates required for confirmatory claims.")
    save_json(R / "results/summary.json", summary)
    if not summary["all_passed"]:
        raise RuntimeError("Test implementation validation failed")
    print("TEST_SELECTED", summary["selected_scores"], flush=True)


def phase_deploy(protocol, d, v, fingerprints):
    selection = read_selection(fingerprints)
    summary = json.loads((R / "results/summary.json").read_text(encoding="utf-8"))
    if not summary["all_passed"]:
        raise RuntimeError("Passed frozen test implementation is required before deployment export")
    key = selection["candidate"]
    cfg = protocol["candidates"][key]
    # Only the frozen candidate and its control need a June1 prediction.
    subset = dict(protocol)
    subset["candidates"] = {key: cfg}
    subset["matched_controls"] = {cfg["backbone"] + "_control": protocol["matched_controls"][cfg["backbone"] + "_control"]}
    native, checks = predict_backbones(d, v, fingerprints, subset, "deploy", [31])
    old = pd.read_csv(B / "round8b/results/selected_deployment_origin31_1day.csv")
    expected_names = np.repeat(d["names"], 96)
    if len(old) != 5472 or not np.array_equal(old["series"].to_numpy(), expected_names):
        raise RuntimeError("Prior deployment row ordering differs")
    prior = old.predicted_util.to_numpy().reshape(57, 1, 96)
    alpha = selection["chosen"]["alpha"]
    selected = corrected(prior, native[key], native[cfg["backbone"] + "_control"], v["ix"], alpha)[:, 0]
    candidate = corrected(prior, native[key], native[cfg["backbone"] + "_control"], v["ix"], 1)[:, 0]
    width = summary["uncertainty"]["relative_halfwidth"]
    half = width * d["y"][:, 24:31].mean((1, 2)).clip(.05)
    cpu, mem = np.full(57, np.nan), np.full(57, np.nan)
    cpu[v["ix"]] = v["u"][v["ix"], 24:31, :, 0].mean((1, 2))
    mem[v["ix"]] = v["u"][v["ix"], 24:31, :, 1].mean((1, 2))
    has = np.zeros(57, dtype=bool)
    has[v["ix"]] = v["present"][v["ix"], 17:31].all((1, 2))
    out = old[["series", "cell", "local_trace_time_PT"]].copy()
    out["predicted_util"] = selected.ravel()
    out["native_covariate_candidate_util"] = candidate.ravel()
    out["prior_round8b_util"] = prior.ravel()
    out["lower80_empirical"] = np.maximum(selected - half[:, None], 0).ravel()
    out["upper80_empirical"] = np.minimum(selected + half[:, None], 1).ravel()
    out["history_cpu_mean7_NCU"] = np.repeat(cpu, 96)
    out["history_memory_mean7_normalized"] = np.repeat(mem, 96)
    out["has_complete_cpu_history"] = np.repeat(has, 96)
    out["history_cpu_missing"] = np.repeat(~has, 96)
    times = pd.to_datetime(out.local_trace_time_PT)
    checks.update(rows5472=len(out) == 5472, domains57=out.series.nunique() == 57,
                  june1_time_boundary=bool(times.min() == pd.Timestamp("2019-06-01 00:00:00") and
                                          times.max() == pd.Timestamp("2019-06-01 23:45:00")),
                  finite_bounded_power=bool(np.isfinite(selected).all() and selected.min() >= 0 and selected.max() <= 1),
                  other52_exact_prior=bool(np.array_equal(selected[d["cells"] != "a"], prior[d["cells"] != "a", 0])),
                  cpu_missing_outside_a=bool(np.isnan(cpu[d["cells"] != "a"]).all()),
                  finite_nonnegative_history_cpu_in_a=bool(np.isfinite(cpu[v["ix"]]).all() and (cpu[v["ix"]] >= 0).all()),
                  five_complete_history_flags=bool(has.sum() == 5),
                  interval_order=bool(((out.lower80_empirical <= out.predicted_util) &
                                       (out.predicted_util <= out.upper80_empirical)).all()),
                  selection_hash_preserved=digest(R / "results/selection.json") ==
                     (R / "results/selection.sha256").read_text().strip())
    if not all(checks.values()):
        raise RuntimeError("Deployment verification failed: " + str(checks))
    path = R / "results/selected_deployment_origin31_1day.csv"
    out.to_csv(path, index=False)
    save_json(R / "results/deployment_verification.json", dict(checks=checks, all_passed=True,
              output=str(path), output_sha256=digest(path), selection_alpha=alpha,
              candidate=key, input_end="2019-05-31", forecast_start="2019-06-01",
              future_truth_available=False, not_live_forecast=True,
              cpu_columns="Observed historical seven-day mean, not future CPU prediction or physical kW",
              interval_scope=summary["uncertainty"]["scope"]))
    print("DEPLOY_VERIFIED", path, flush=True)


def main(default_phase="dev"):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["dev", "test", "deploy", "all"], default=default_phase)
    ap.add_argument("--data-root", type=Path, help="Optional separate data storage root; defaults to the resolved research folder")
    args = ap.parse_args()
    protocol = register()  # Persist the exact registry before importing/instantiating a foundation model.
    d, v, fingerprints = load_sources(args.data_root)
    actions = dict(dev=phase_dev, test=phase_test, deploy=phase_deploy)
    for phase in (list(actions) if args.phase == "all" else [args.phase]):
        started = utc()
        try:
            actions[phase](protocol, d, v, fingerprints)
        except Exception as error:
            save_json(R / f"results/{phase}_last_failure.json", dict(at_utc=utc(), started_at_utc=started,
                      error_type=type(error).__name__, error=str(error), caches_retained=True,
                      selection_preserved=(R / "results/selection.json").exists()))
            raise
    print("ROUND10_PHASE_COMPLETE", args.phase, flush=True)


if __name__ == "__main__":
    main()
