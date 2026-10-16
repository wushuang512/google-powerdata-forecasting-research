"""Write a separate approved implementation repair record after real diagnosis.

No model runs. Source project files and original freeze are read-only. --output
must explicitly name the new provenance file; refuse any existing destination.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

FROZEN_SCRIPT = "fd0e90d490ab947dc155aff95e7e646dc9c95f86f3c2404fa6423dfd549cbfeb"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--repair-script", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    top = a.research_root.resolve()
    b, root = top / "experiment", top / "google-clusterdata-2019"
    r = b / "round10"
    if a.output.exists():
        raise RuntimeError("Refuse to overwrite an existing provenance record")
    selection_path = r / "results/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if sha(selection_path) != (r / "results/selection.sha256").read_text().strip():
        raise RuntimeError("Original selection hash differs")
    if selection["candidate"] != "chronos_usage_resource" or selection["chosen"]["alpha"] != 0:
        raise RuntimeError("Preserve the actual original candidate and alpha-zero choice")
    if sha(root / "code/native_covariate_experiment.py") != FROZEN_SCRIPT:
        raise RuntimeError("Original scientific script bytes changed")
    fp = selection["source_fingerprints"]
    exact_sources = {
        "script_sha256": root / "code/native_covariate_experiment.py",
        "protocol_sha256": r / "docs/protocol.json",
        "power_sha256": b / "results/data.npz",
        "usage_sha256": root / "aggregated/complete_usage_a_15min.npz",
        "resource_sha256": root / "aggregated/resources_15min.npz",
        "audit_sha256": root / "results/complete_projection_audit.json",
    }
    for key, path in exact_sources.items():
        if sha(path) != fp[key]:
            raise RuntimeError("Original source fingerprint differs: " + key)
    for relative, expected in fp["prior_artifact_sha256"].items():
        if sha(b / relative) != expected:
            raise RuntimeError("Existing prior/control archive differs: " + relative)
    diagnostic = json.loads(a.diagnostic.read_text(encoding="utf-8"))
    if diagnostic.get("immutable_preserved") is not True or diagnostic.get("future_target_truth_scored") is not False:
        raise RuntimeError("Actual diagnostic must preserve immutable files and avoid truth scoring")
    wanted = [f"inherit_tf32_threads{threads}_repeat{repeat}" for threads in (2, 4) for repeat in (1, 2)]
    for call in wanted:
        row = next((r for r in diagnostic.get("calls", []) if r["call"] == call), None)
        if not row or row["vs_old_test_cache"]["maximum_absolute_difference"] != 0 or not row["vs_old_test_cache"]["bitwise_values_equal"]:
            raise RuntimeError("Need all four inherited TF32 numerical repetitions to match old cache exactly")
    if diagnostic.get("all_adjacent_repeats_equal") is not True:
        raise RuntimeError("Numerical repeated predictions were not stable")
    original_weights = {}
    for backbone in ("timesfm", "chronos"):
        meta = json.loads((r / f"results/cache/dev_{backbone}_control_origin14.json").read_text())
        original_weights[backbone] = {key: meta["identity"][key] for key in ("checkpoint_sha256", "config_sha256")}
        folder = b / "models" / ("timesfm3-base" if backbone == "timesfm" else "chronos2-base")
        if sha(folder / "model.safetensors") != original_weights[backbone]["checkpoint_sha256"] or sha(folder / "config.json") != original_weights[backbone]["config_sha256"]:
            raise RuntimeError("Foundation weights/config differ from the original development run")
    dev_files = sorted((r / "results").glob("dev_*.json")) + sorted((r / "results").glob("dev_*.npz")) + sorted((r / "results").glob("dev_*.csv"))
    dev_files += sorted((r / "results/cache").glob("dev_*.*"))
    if len(list((r / "results/cache").glob("dev_*.npz"))) != 48:
        raise RuntimeError("Require preserved original 48 development prediction caches")
    dev_hashes = {str(p.relative_to(r)).replace("\\", "/"): sha(p) for p in dev_files}
    failure_files = [r / "results/test_control_regression.json", r / "results/test_causality_checks.json",
                     r / "results/test_last_failure.json"]
    failure_files += sorted((r / "results/cache").glob("test_*.*"))
    failure_hashes = {str(p.relative_to(r)).replace("\\", "/"): sha(p) for p in failure_files}
    source_files = [root / "code/low_memory_timesfm.py",
                    b / ".venv/Lib/site-packages/chronos/chronos2/pipeline.py",
                    b / ".venv/Lib/site-packages/chronos/chronos2/model.py",
                    b / ".venv/Lib/site-packages/chronos/chronos2/layers.py",
                    b / ".venv/Lib/site-packages/chronos/chronos2/dataset.py",
                    b / ".venv/Lib/site-packages/transformers/utils/import_utils.py"]
    record = dict(repair_id="repair1", status="approved_for_execution",
            created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            authorization="Root approved implementation-only repair after protected source and prediction diagnosis",
            original_script_sha256=FROZEN_SCRIPT, new_script_sha256=sha(a.repair_script),
            original_selection_sha256=sha(selection_path), original_protocol_sha256=sha(r / "docs/protocol.json"),
            original_candidate=selection["candidate"], original_alpha=selection["chosen"]["alpha"],
            original_freeze_time=selection["frozen_at_utc"], frozen_source_fingerprints=fp,
            scientific_configuration_unchanged=True, source_and_weight_hashes_unchanged=True,
            test_target_truth_used=False, test_reselection=False,
            original_tolerance=dict(atol=2e-6, rtol=1e-6),
            confirmed_hypothesis="Trainer sets global FP32 TF32 on Torch>=2.9; successful Chronos fit restores local backend values but not global inheritance. Existing test cache is reproduced bitwise by global TF32 with local inheritance. This is an execution-state repair, not fine-tuning.",
            changed_behavior="Explicitly fix and restore global/local precision for sequential backbones. Keep original IEEE development caches/freeze. Reproduce legacy Chronos TEST in TF32 with local inheritance; keep TimesFM TEST and both new June1 deployment queries IEEE. All repair outputs use a new namespace.",
            global_fp32_policy={"timesfm_test": "ieee", "chronos_test": "tf32",
                               "timesfm_deploy": "ieee", "chronos_deploy": "ieee"},
            local_fp32_policy="none",
            diagnostic_evidence=[dict(path=str(a.diagnostic.resolve()), sha256=sha(a.diagnostic))],
            original_dev_sha256=dev_hashes, original_failure_sha256=failure_hashes,
            original_weights=original_weights,
            implementation_sources_sha256={str(p): sha(p) for p in source_files},
            limitations="Different historical split numerical states are preserved transparently. Alpha remains zero; no new point forecast is adopted. No independent May test confirmation.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(a.output)
    print("PROVENANCE_SHA256", sha(a.output))


if __name__ == "__main__":
    main()
