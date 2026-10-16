"""Origin22 numerical regression only: no target-truth scoring or model choice.

Run under the root's single research-process memory guard. Writes only a new
diagnostic folder, never overwrites frozen protocol/selection/main caches.
The default factorial contrasts original batch64/no_grad and batch32/inference,
including two adjacent repeats per condition. It changes no numerical flags.
"""
import argparse
import ast
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def comparison(prediction, reference):
    difference = np.abs(prediction - reference)
    return dict(maximum_absolute_difference=float(difference.max()),
                mean_absolute_difference=float(difference.mean()),
                median_absolute_difference=float(np.median(difference)),
                nonzero_points=int((difference != 0).sum()), points=int(difference.size),
                bitwise_values_equal=bool(np.array_equal(prediction, reference)),
                registered_tolerance_passed=bool(np.allclose(prediction, reference, atol=2e-6, rtol=1e-6)),
                per_domain_maximum_absolute_difference=difference.max(1).tolist())


def flags(torch):
    result = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  default_dtype=str(torch.get_default_dtype()),
                  matmul_precision=torch.get_float32_matmul_precision(),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  cpu_threads=torch.get_num_threads(),
                  cudnn_enabled=torch.backends.cudnn.enabled,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                  memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                  math_sdp=torch.backends.cuda.math_sdp_enabled(),
                  cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    for name, getter in {
        "cuda_matmul_fp32_precision": lambda: torch.backends.cuda.matmul.fp32_precision,
        "cudnn_fp32_precision": lambda: torch.backends.cudnn.fp32_precision,
        "global_fp32_precision": lambda: torch.backends.fp32_precision,
    }.items():
        try:
            result[name] = getter()
        except (AttributeError, RuntimeError) as error:
            result[name] = "unavailable:" + type(error).__name__
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--research-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Fresh diagnostic directory; refuse an existing populated directory")
    args = ap.parse_args()
    top, out = args.research_root.resolve(), args.output_dir.resolve()
    b, root = top / "experiment", top / "google-clusterdata-2019"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Use a new output directory; existing evidence is preserved")
    out.mkdir(parents=True, exist_ok=True)
    selection = b / "round10/results/selection.json"
    immutable = dict(selection_sha256=sha(selection),
                     protocol_sha256=sha(b / "round10/docs/protocol.json"),
                     failure_cache_sha256=sha(b / "round10/results/cache/test_chronos_control_origin22.npz"),
                     old_test_cache_sha256=sha(b / "round2/results/final_chronos24.npz"))
    if sha(selection) != (selection.with_name("selection.sha256")).read_text().strip():
        raise RuntimeError("Frozen selection hash is already inconsistent")
    with np.load(b / "results/data.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    with np.load(root / "aggregated/complete_usage_a_15min.npz", allow_pickle=False) as z:
        u = z["aggregate"]
    with np.load(root / "aggregated/resources_15min.npz", allow_pickle=False) as z:
        resource = z["resource"]
    ix = np.where(d["cells"] == "a")[0]
    v = dict(ix=ix, u=u, resource=resource, present=u[..., 3] > 0)
    from native_covariate_experiment import input_arrays, input_digest, memory_state
    target, covariates, names = input_arrays(d, v, 22,
                            dict(backbone="chronos", usage=False, resources=False))
    # Execute only the historical old inputs() function's AST, avoiding legacy
    # module imports, training helpers, global Torch state and source mutation.
    original_file = b / "round2/code/round2_lib.py"
    module = ast.parse(original_file.read_text(encoding="utf-8"))
    original_function = next(node for node in module.body
                             if isinstance(node, ast.FunctionDef) and node.name == "inputs")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[original_function], type_ignores=[]), str(original_file), "exec"), namespace)
    old_groups = namespace["inputs"](d, 22, kind="cellprod", h=96, context=1344)
    a_group = list(np.unique(d["cells"])).index("a")
    old_item = old_groups[a_group]
    exact_inputs = dict(target_equal=bool(np.array_equal(old_item["target"], target, equal_nan=True)),
                        covariate_names_equal=list(old_item["past_covariates"]) == names,
                        covariate_values_equal=bool(np.array_equal(
                            np.stack(list(old_item["past_covariates"].values())), covariates, equal_nan=True)),
                        target_shape=list(target.shape), covariate_shape=list(covariates.shape),
                        input_sha256=input_digest(target, covariates, names), covariate_names=names)
    if not all(exact_inputs[k] for k in ("target_equal", "covariate_names_equal", "covariate_values_equal")):
        raise RuntimeError("Old and new input construction differs; stop before model load")
    with np.load(b / "round2/results/final_chronos24.npz", allow_pickle=False) as z:
        old_reference = z["zero_cellprod"][ix, 0]
    with np.load(b / "round10/results/cache/test_chronos_control_origin22.npz", allow_pickle=False) as z:
        failure_reference = z["prediction"]
    state = memory_state()
    if state["available_physical_bytes"] < 3 * 2 ** 30 or state["available_commit_bytes"] < 6 * 2 ** 30:
        raise MemoryError("Need 3 GiB physical and 6 GiB commit reserve before diagnostic model load")
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    from chronos import Chronos2Pipeline
    before = flags(torch)
    pipe = None
    arrays, rows = {}, []
    try:
        pipe = Chronos2Pipeline.from_pretrained(str(b / "models/chronos2-base"),
                                               device_map="cuda", local_files_only=True)
        model_info = dict(dtype=str(pipe.model.dtype), device=str(pipe.model.device),
                          training=pipe.model.training,
                          attention_implementation=pipe.model.config._attn_implementation,
                          checkpoint_sha256=sha(b / "models/chronos2-base/model.safetensors"),
                          config_sha256=sha(b / "models/chronos2-base/config.json"))
        # Four conditions distinguish batch argument and outer inference context;
        # both legacy and new APIs ultimately use the same no-gradient pipeline.
        for batch, context in ((64, "no_grad"), (32, "inference"),
                               (32, "no_grad"), (64, "inference")):
            for repeat in (1, 2):
                state = memory_state()
                if state["available_physical_bytes"] < 1.5 * 2 ** 30 or state["available_commit_bytes"] < 2 * 2 ** 30:
                    raise MemoryError("Resource floor reached between diagnostic calls")
                call = f"batch{batch}_{context}_repeat{repeat}"
                started = time.perf_counter()
                manager = torch.no_grad() if context == "no_grad" else torch.inference_mode()
                with manager:
                    q, mean = pipe.predict_quantiles([old_item], prediction_length=96,
                              quantile_levels=[.5], context_length=1344, batch_size=batch,
                              cross_learning=False)
                    p = q[0][:, :, 0].float().cpu().numpy()
                    del q, mean
                arrays[call] = p
                row = dict(call=call, seconds=time.perf_counter()-started,
                           vs_old_test_cache=comparison(p, old_reference),
                           vs_failed_round10_cache=comparison(p, failure_reference),
                           flags=flags(torch))
                if repeat == 2:
                    row["vs_adjacent_repeat"] = comparison(p, arrays[f"batch{batch}_{context}_repeat1"])
                rows.append(row)
                np.savez_compressed(out / "control_calls.npz", **arrays)
                report = dict(origin=22, future_target_truth_scored=False,
                              purpose="Numerical regression only; not model selection", input_exactness=exact_inputs,
                              immutable_before=immutable, flags_before=before, model=model_info, calls=rows)
                (out / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print("DIAGNOSTIC", call, row["vs_old_test_cache"], flush=True)
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    report["flags_after"] = flags(torch)
    report["immutable_after"] = dict(selection_sha256=sha(selection),
            protocol_sha256=sha(b / "round10/docs/protocol.json"),
            failure_cache_sha256=sha(b / "round10/results/cache/test_chronos_control_origin22.npz"),
            old_test_cache_sha256=sha(b / "round2/results/final_chronos24.npz"))
    report["immutable_preserved"] = report["immutable_before"] == report["immutable_after"]
    report["maximum_between_condition_difference"] = float(max(
        np.abs(a-bb).max() for a in arrays.values() for bb in arrays.values()))
    (out / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DIAGNOSTIC_COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
