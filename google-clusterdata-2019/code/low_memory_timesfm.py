"""Load local TimesFM3 without a full CPU or duplicate GPU parameter copy.

This changes allocation placement only.  The ordinary forecaster loader still
reads the original configuration and adopts the original float32 checkpoint.
Callers must enforce and monitor their own RAM/commit/VRAM reserve policy.
"""

import dataclasses
import json
import time
from functools import wraps
from pathlib import Path
from unittest.mock import patch


def _plain(value):
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value


def load_forecaster(model_dir, batch=4):
    """Return ``(forecaster, actual_validation)``; refuse CPU fallback.

    No checkpoint is downloaded, rewritten, converted, quantized, or fine-tuned.
    Weight comparisons hold one CPU tensor copy at a time.  Safetensors uses a
    file mapping, so touched checkpoint pages may still briefly occupy RAM; this
    function is not a guarantee that a particular RAM reserve will be met.
    """
    import torch
    from packaging.version import Version
    from safetensors import safe_open, __version__ as safetensors_version
    from safetensors.torch import load_file
    from timesfm3 import TimesFM3Forecaster
    from timesfm3.torch.model import TimesFM3Torch
    from timesfm3.torch.transformer import RotaryPositionalEmbedding

    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise ValueError("batch must be a positive integer")
    if not torch.cuda.is_available():
        raise RuntimeError("Low-memory TimesFM3 loading requires CUDA; no CPU fallback")
    if Version(safetensors_version) < Version("0.4.3"):
        raise RuntimeError("Direct CUDA checkpoint loading requires safetensors >= 0.4.3")
    directory = Path(model_dir).resolve(strict=True)
    configuration_path = directory / "config.json"
    checkpoint_path = directory / "model.safetensors"
    if not directory.is_dir() or not configuration_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Require the existing local config.json and model.safetensors")
    original_configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    device = torch.device("cuda", torch.cuda.current_device())

    restored_buffers = []
    buffer_values_equal = True
    load_checks = {}
    original_init = TimesFM3Torch.__init__
    original_safetensor_loader = TimesFM3Torch._load_as_safetensor

    @wraps(original_init)
    def meta_init(self, *args, **kwargs):
        # Restrict the default-device override to parameter construction.  In
        # particular, safetensors' native storage creation runs after it exits.
        with torch.device("meta"):
            original_init(self, *args, **kwargs)

    def assign_cuda_checkpoint(cls, model, model_file, map_location, strict):
        nonlocal buffer_values_equal
        if torch.get_default_device().type != "cpu":
            raise RuntimeError("Native checkpoint loading requires the normal CPU default device")
        load_checks["all_parameters_meta_before_assignment"] = all(
            parameter.is_meta for parameter in model.parameters()
        )
        if not load_checks["all_parameters_meta_before_assignment"]:
            raise RuntimeError("Refuse checkpoint loading into a materialized parameter copy")
        # assign=True transfers storage ownership to the module instead of
        # copying into another initialized CPU/GPU parameter set.
        checkpoint_state = load_file(model_file, device=str(device))
        result = model.load_state_dict(checkpoint_state, strict=True, assign=True)
        load_checks["strict_assignment_keys_equal"] = (
            not result.missing_keys and not result.unexpected_keys
        )
        assigned_state = model.state_dict()
        load_checks["assigned_parameters_share_checkpoint_storage"] = all(
            assigned_state[name].data_ptr() == value.data_ptr()
            for name, value in checkpoint_state.items()
        )
        del assigned_state, checkpoint_state

        # RoPE constants are nonpersistent and therefore absent from the
        # checkpoint.  Restore them before Forecaster calls model.to(cuda),
        # using exactly the original CPU expression and float32 rounding.
        for name, module in model.named_modules():
            if not isinstance(module, RotaryPositionalEmbedding):
                continue
            fraction = (
                2.0
                * torch.arange(module.embedding_dims // 2, dtype=torch.float32, device="cpu")
                / module.embedding_dims
            )
            expected = module.min_timescale * (
                module.max_timescale / module.min_timescale
            ) ** fraction
            buffer_name = f"{name}.timescale" if name else "timescale"
            module.timescale = expected.to(device=device)
            buffer_values_equal &= torch.equal(module.timescale.detach().cpu(), expected)
            restored_buffers.append(buffer_name)
            del expected, fraction
        remaining_meta = [
            name for name, value in (*model.named_parameters(), *model.named_buffers())
            if value.is_meta
        ]
        if remaining_meta:
            raise RuntimeError("Unrestored meta tensors: " + ", ".join(remaining_meta))
        model.eval()
        return model

    # Retain the original directory loader and all forecaster configuration
    # synchronization.  Both temporary class patches are restored on exceptions
    # as well as success.  Only this serial load may run while patches are active.
    with patch.object(TimesFM3Torch, "__init__", new=meta_init), patch.object(
        TimesFM3Torch, "_load_as_safetensor", new=classmethod(assign_cuda_checkpoint)
    ):
        forecaster = TimesFM3Forecaster.from_pretrained(
            str(directory),
            device=str(device),
            per_core_batch_size=batch,
            local_files_only=True,
        )
    model = forecaster.model
    load_checks["temporary_class_methods_restored"] = (
        TimesFM3Torch.__init__ is original_init
        and TimesFM3Torch._load_as_safetensor.__func__ is original_safetensor_loader.__func__
    )
    torch.cuda.synchronize(device)

    actual_configuration = {
        name: _plain(getattr(model, name)) for name in original_configuration
    }
    state = model.state_dict()
    parameter_names = {name for name, _ in model.named_parameters()}
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    shape_mismatches = []
    dtype_mismatches = []
    value_mismatches = []
    missing_keys = []
    unexpected_keys = []
    checked_tensors = 0
    with torch.inference_mode(), safe_open(
        str(checkpoint_path), framework="pt", device="cpu"
    ) as checkpoint:
        checkpoint_names = set(checkpoint.keys())
        missing_keys = sorted(checkpoint_names - set(state))
        unexpected_keys = sorted(set(state) - checkpoint_names)
        for name in sorted(checkpoint_names & set(state)):
            # Do not collect these views/copies into a CPU state dictionary.
            expected = checkpoint.get_tensor(name)
            actual = state[name]
            if actual.shape != expected.shape:
                shape_mismatches.append(name)
            elif actual.dtype != expected.dtype or expected.dtype != torch.float32:
                dtype_mismatches.append(name)
            else:
                actual_cpu = actual.detach().cpu()
                if not torch.equal(actual_cpu, expected):
                    value_mismatches.append(name)
                del actual_cpu
            checked_tensors += 1
            del expected, actual
    buffers = list(model.named_buffers())
    all_parameters_cuda = all(parameter.device == device for parameter in model.parameters())
    all_buffers_cuda = all(buffer.device == device for _, buffer in buffers)
    all_float32 = all(parameter.dtype == torch.float32 for parameter in model.parameters())
    floating_buffers_float32 = all(
        not buffer.is_floating_point() or buffer.dtype == torch.float32
        for _, buffer in buffers
    )
    nonpersistent_buffers = sorted(
        f"{name}.{buffer_name}" if name else buffer_name
        for name, module in model.named_modules()
        for buffer_name in module._non_persistent_buffers_set
        if getattr(module, buffer_name) is not None
    )
    checks = {
        **load_checks,
        "meta_assignment_loader_executed": "all_parameters_meta_before_assignment" in load_checks,
        "original_configuration_equal": actual_configuration == original_configuration,
        "checkpoint_keys_equal": not missing_keys and not unexpected_keys,
        "all_parameter_keys_in_checkpoint": parameter_names <= checkpoint_names,
        "checkpoint_shapes_equal": not shape_mismatches,
        "checkpoint_dtypes_float32_equal": not dtype_mismatches,
        "checkpoint_values_equal": not value_mismatches,
        "all_parameters_cuda": all_parameters_cuda,
        "all_buffers_cuda": all_buffers_cuda,
        "parameters_float32": all_float32,
        "floating_buffers_float32": floating_buffers_float32,
        "all_nonpersistent_buffers_restored": sorted(restored_buffers) == nonpersistent_buffers,
        "restored_buffers_equal_original_cpu_expression": bool(buffer_values_equal),
        "model_in_eval_mode": not model.training,
    }
    validation = {
        "method": "Meta module initialization; strict assign of one CUDA checkpoint; original local float32 weights; CPU RoPE constants restored",
        "model_directory": str(directory),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "batch": batch,
        "device": str(device),
        "checkpoint_map_location": str(device),
        "checkpoint_load_strict": True,
        "checkpoint_load_assign": True,
        "safetensors_version": safetensors_version,
        "parameter_count": parameter_count,
        "checked_tensors": checked_tensors,
        "restored_nonpersistent_buffers": restored_buffers,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "value_mismatches": value_mismatches,
        "checks": checks,
        "all_passed": all(checks.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "process_cuda_allocated_bytes_after_load": torch.cuda.memory_allocated(device),
        "process_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "validation_scope": "Loaded weights, configuration and constants only; inference regression must be checked separately",
    }
    if not validation["all_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("Low-memory TimesFM3 integrity verification failed: " + ", ".join(failed))
    return forecaster, validation
