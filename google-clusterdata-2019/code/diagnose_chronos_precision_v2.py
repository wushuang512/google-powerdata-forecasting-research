"""Origin22 legacy-control numerical-state diagnosis; no truth scoring.

Explicitly contrast torch >=2.9 global FP32 settings and per-backend inheritance.
Uses only new fp32_precision APIs, never the deprecated allow_tf32 API. Run under
the root's single-process memory guard, with a new output directory.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np
from diagnose_chronos_control import comparison, sha


def flags(torch):
    return dict(torch_version=torch.__version__,
                default_dtype=str(torch.get_default_dtype()),
                global_fp32_precision=torch.backends.fp32_precision,
                cuda_matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
                cudnn_fp32_precision=torch.backends.cudnn.fp32_precision,
                cpu_threads=torch.get_num_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--research-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    top, out = args.research_root.resolve(), args.output_dir.resolve()
    b, root = top / "experiment", top / "google-clusterdata-2019"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Use a new diagnostic directory; do not overwrite existing evidence")
    out.mkdir(parents=True, exist_ok=True)
    frozen = b / "round10/results/selection.json"
    unchanged_files = [frozen, frozen.with_name("selection.sha256"),
                       b / "round10/docs/protocol.json",
                       b / "round10/results/cache/test_chronos_control_origin22.npz",
                       b / "round2/results/final_chronos24.npz"]
    unchanged = {str(p.relative_to(b)): sha(p) for p in unchanged_files}
    if sha(frozen) != frozen.with_name("selection.sha256").read_text().strip():
        raise RuntimeError("Original frozen selection checksum is inconsistent")
    with np.load(b / "results/data.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    with np.load(root / "aggregated/complete_usage_a_15min.npz", allow_pickle=False) as z:
        u = z["aggregate"]
    with np.load(root / "aggregated/resources_15min.npz", allow_pickle=False) as z:
        resource = z["resource"]
    ix = np.where(d["cells"] == "a")[0]
    from native_covariate_experiment import input_arrays, input_digest, memory_state
    target, cov, names = input_arrays(d, dict(ix=ix, u=u, resource=resource, present=u[..., 3] > 0),
                                    22, dict(backbone="chronos", usage=False, resources=False))
    item = dict(target=target, past_covariates=dict(zip(names, cov)))
    with np.load(b / "round2/results/final_chronos24.npz", allow_pickle=False) as z:
        old = z["zero_cellprod"][ix, 0]
    with np.load(b / "round10/results/cache/test_chronos_control_origin22.npz", allow_pickle=False) as z:
        failure = z["prediction"]
    state = memory_state()
    if state["available_physical_bytes"] < 3 * 2**30 or state["available_commit_bytes"] < 6 * 2**30:
        raise MemoryError("Need 3 GiB RAM and 6 GiB commit reserve before diagnostic load")
    import torch
    from chronos import Chronos2Pipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refuse CPU fallback")
    original = dict(global_fp32=torch.backends.fp32_precision,
                    matmul_fp32=torch.backends.cuda.matmul.fp32_precision,
                    cudnn_fp32=torch.backends.cudnn.fp32_precision,
                    cpu_threads=torch.get_num_threads())
    original_flags = flags(torch)
    pipe, rows, arrays = None, [], {}
    # Parent's source audit identified Trainer enabling GLOBAL tf32, while the
    # successful Chronos fit restores only per-backend values (often 'none').
    # The inheritance conditions therefore distinguish this exact hypothesis.
    modes = [
        ("inherit_ieee", "ieee", "none", "none"),
        ("inherit_tf32", "tf32", "none", "none"),
        ("explicit_ieee", "ieee", "ieee", "ieee"),
        ("explicit_tf32", "tf32", "tf32", "tf32"),
    ]
    report = dict(origin=22, future_target_truth_scored=False, test_reselection=False,
                  purpose="Only reproduction of an existing prediction archive",
                  immutable_before=unchanged, original_flags=original_flags,
                  input_sha256=input_digest(target, cov, names), calls=rows)
    try:
        torch.set_num_threads(2)
        pipe = Chronos2Pipeline.from_pretrained(str(b / "models/chronos2-base"),
                                               device_map="cuda", local_files_only=True)
        report["model"] = dict(dtype=str(pipe.model.dtype), training=pipe.model.training,
                    attention_implementation=pipe.model.config._attn_implementation,
                    checkpoint_sha256=sha(b / "models/chronos2-base/model.safetensors"),
                    config_sha256=sha(b / "models/chronos2-base/config.json"))
        for label, global_precision, matmul_precision, cudnn_precision in modes:
            for threads in (2, 4):
                # A local 'none' means inherit the global mode. Do not use
                # torch.set_float32_matmul_precision / deprecated allow_tf32.
                torch.backends.fp32_precision = global_precision
                torch.backends.cuda.matmul.fp32_precision = matmul_precision
                torch.backends.cudnn.fp32_precision = cudnn_precision
                torch.set_num_threads(threads)
                for repeat in (1, 2):
                    state = memory_state()
                    if state["available_physical_bytes"] < 1.5 * 2**30 or state["available_commit_bytes"] < 2 * 2**30:
                        raise MemoryError("Resource reserve reached between short numerical checks")
                    call = f"{label}_threads{threads}_repeat{repeat}"
                    started = time.perf_counter()
                    with torch.no_grad():
                        quant, mean = pipe.predict_quantiles([item], prediction_length=96,
                                    quantile_levels=[.5], context_length=1344,
                                    batch_size=64, cross_learning=False)
                        prediction = quant[0][:, :, 0].float().cpu().numpy()
                        del quant, mean
                    arrays[call] = prediction
                    row = dict(call=call, seconds=time.perf_counter()-started, flags=flags(torch),
                               vs_old_test_cache=comparison(prediction, old),
                               vs_failed_round10_cache=comparison(prediction, failure))
                    if repeat == 2:
                        row["vs_adjacent_repeat"] = comparison(prediction,
                                      arrays[f"{label}_threads{threads}_repeat1"])
                    rows.append(row)
                    np.savez_compressed(out / "precision_calls.npz", **arrays)
                    (out / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    print("PRECISION_DIAGNOSTIC", call, row["vs_old_test_cache"], flush=True)
    finally:
        # Restore the exact caller settings even when a memory guard raises.
        torch.backends.fp32_precision = original["global_fp32"]
        torch.backends.cuda.matmul.fp32_precision = original["matmul_fp32"]
        torch.backends.cudnn.fp32_precision = original["cudnn_fp32"]
        torch.set_num_threads(original["cpu_threads"])
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        report["flags_after_restoration"] = flags(torch)
        report["immutable_after"] = {str(p.relative_to(b)): sha(p) for p in unchanged_files}
        report["immutable_preserved"] = report["immutable_after"] == unchanged
        (out / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["all_adjacent_repeats_equal"] = all(r.get("vs_adjacent_repeat", {}).get("bitwise_values_equal", True) for r in rows)
    report["matching_legacy_conditions"] = [r["call"] for r in rows if r["vs_old_test_cache"]["registered_tolerance_passed"]]
    (out / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PRECISION_DIAGNOSTIC_COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
