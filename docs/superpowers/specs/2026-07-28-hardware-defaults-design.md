# Hardware Defaults: Auto-enable torch.compile and AMP on CUDA

**Date:** 2026-07-28
**Status:** Approved for implementation

---

## Problem

`compile_model` and `native_amp` are opt-in flags that default to off. Users who don't know to pass `--compile-model` or `--native-amp` get no benefit from torch.compile or AMP, even on hardware that supports them well. The infrastructure to use both already exists in `train_base.py`; it just isn't activated by default.

---

## Goal

Auto-enable `torch.compile` and `native_amp` when CUDA is available, without changing behaviour for CPU or MPS users and without overriding explicit user choices in YAML config.

---

## Scope

- **Phase 1 (this PR):** timeseries task type only — `compile_model` and `native_amp` already exist in `timeseries/params.py` and the argv plumbing is in place.
- **Phase 2 (separate PR):** vision and audio task types — add the flags to their `params.py` files, wire through their argv builders, then call `apply_hardware_defaults` from their params constructors. No changes to `hardware_defaults.py` required.

---

## Approach

Detection lives in a single shared utility function called from each task's `TrainingParams` constructor. The function is a no-op on non-CUDA hardware and skips any field the user explicitly set in their YAML.

Detection condition: `torch.cuda.is_available()` only. MPS is excluded — AMP float16 support on MPS is unreliable, and `aot_eager` torch.compile on MPS is slower to compile with inconsistent speedup. MPS users can still opt in via YAML.

---

## Architecture

```
YAML config loaded
       ↓
TrainingParams.__init__(params_dict, explicitly_set)
       ↓
apply_hardware_defaults(params, explicitly_set)
       ↓  if CUDA available AND key not in explicitly_set:
              params.training.compile_model = 1
              params.training.native_amp = True
       ↓
argv built from params (timeseries_base.py — unchanged)
       ↓
train script receives --compile-model 1 --native-amp as before
```

---

## New File: `hardware_defaults.py`

**Path:** `tinyml-modelmaker/tinyml_modelmaker/utils/hardware_defaults.py`

```python
import torch

def apply_hardware_defaults(params, explicitly_set: set):
    """Auto-enable compile_model and native_amp when CUDA is available.

    Skips any field present in explicitly_set — those represent deliberate
    user choices from the YAML config and must not be overridden.
    hasattr guards make this safe to call from task types whose params
    don't yet carry these fields (vision, audio in Phase 2).
    """
    if not torch.cuda.is_available():
        return
    if 'compile_model' not in explicitly_set and hasattr(params.training, 'compile_model'):
        if getattr(params.training, 'compile_model', 0) == 0:
            params.training.compile_model = 1
    if 'native_amp' not in explicitly_set and hasattr(params.training, 'native_amp'):
        if not getattr(params.training, 'native_amp', False):
            params.training.native_amp = True
```

---

## Changes to Existing Files

### `tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/params.py`

`params.py` exposes a module-level `init_params(*args, **kwargs)` function (no class). The user config arrives as the first positional arg (a dict). Changes:

- Before creating `ConfigDict`, extract the `training` keys the user supplied:
  ```python
  user_training_keys = set(args[0].get('training', {}).keys()) if args else set()
  ```
- After `params = utils.ConfigDict(default_params, *args, **kwargs)`, call:
  ```python
  apply_hardware_defaults(params, user_training_keys)
  ```
- Import `apply_hardware_defaults` from `tinyml_modelmaker.utils.hardware_defaults`.

No changes to the function signature — callers are unaffected.

No changes are required to `timeseries_base.py`, `train_base.py`, or any train script.

---

## YAML Override Behaviour

Any key present in the user's `training:` YAML block is treated as an explicit choice, regardless of its value:

```yaml
# This keeps AMP off even on CUDA — intentional override
training:
  native_amp: false
```

```yaml
# This keeps compile off even on CUDA — intentional override
training:
  compile_model: 0
```

```yaml
# Neither key present — both auto-enabled on CUDA
training:
  batch_size: 64
```

---

## Testing

**File:** `tinyml-modelmaker/tests/test_hardware_defaults.py`

Three cases, all using `unittest.mock.patch('torch.cuda.is_available')` — no GPU required in CI:

| Case | CUDA mock | explicitly_set | Expected compile_model | Expected native_amp |
|------|-----------|----------------|------------------------|---------------------|
| Auto-enable on CUDA | True | `set()` | 1 | True |
| YAML override respected | True | `{'native_amp'}` | 1 | False (unchanged) |
| No CUDA — no change | False | `set()` | 0 | False |

---

## Phase 2: Vision and Audio (out of scope for this PR)

When ready, the expansion requires:

1. Add `compile_model=0` and `native_amp=False` to `vision/params.py` and `audio/params.py` training defaults.
2. Wire `--compile-model` and `--native-amp` through each task's argv builder (same pattern as `timeseries_base.py:884`).
3. Call `apply_hardware_defaults(self, explicitly_set)` from `VisionTrainingParams.__init__` and `AudioTrainingParams.__init__`.

`hardware_defaults.py` requires no changes — the `hasattr` guards already handle the expansion safely.

---

## Non-Goals

- MPS auto-detection (excluded — float16 AMP unreliable on MPS; aot_eager compile inconsistent benefit)
- CPU autocast (available in PyTorch but no meaningful throughput benefit)
- Gradient accumulation (separate feature)
- Dynamic `num_workers` tuning (separate feature)
