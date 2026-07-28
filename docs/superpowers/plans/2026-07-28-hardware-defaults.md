# Hardware Defaults: Auto-enable torch.compile and AMP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-enable `torch.compile` and `native_amp` in timeseries training when CUDA is available, respecting explicit YAML overrides.

**Architecture:** A new `apply_hardware_defaults(params, explicitly_set)` utility function checks `torch.cuda.is_available()` and patches `params.training.compile_model` and `params.training.native_amp` when the user hasn't explicitly set them. It is called inside `init_params()` in `timeseries/params.py` after the `ConfigDict` is constructed, using keys extracted from the user's raw input dict before merging.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, pytest, `unittest.mock`

## Global Constraints

- Python `==3.10.*`
- No new package dependencies — `torch` and `unittest.mock` are already available
- `hardware_defaults.py` must be safe to call even when `params.training` lacks `compile_model` or `native_amp` (uses `hasattr` guards for future vision/audio expansion)
- MPS is explicitly excluded from auto-detection — CUDA only
- YAML override rule: any key present in the user's `training:` block must not be changed by auto-detection, even if its value matches the default

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `tinyml-modelmaker/tinyml_modelmaker/utils/hardware_defaults.py` | Detection logic + patching |
| Create | `tinyml-modelmaker/tests/test_hardware_defaults.py` | Unit tests for detection function |
| Modify | `tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/params.py:225` | Call `apply_hardware_defaults` in `init_params` |
| Create | `tinyml-modelmaker/tests/test_hardware_defaults_integration.py` | Integration test via `init_params` |

---

## Task 1: `apply_hardware_defaults` — tests and implementation

**Files:**
- Create: `tinyml-modelmaker/tinyml_modelmaker/utils/hardware_defaults.py`
- Create: `tinyml-modelmaker/tests/test_hardware_defaults.py`

**Interfaces:**
- Produces: `apply_hardware_defaults(params: ConfigDict, explicitly_set: set) -> None` — mutates `params.training` in place, returns nothing

---

- [ ] **Step 1: Write the failing tests**

Create `tinyml-modelmaker/tests/test_hardware_defaults.py`:

```python
from unittest.mock import patch
import pytest


def _make_params():
    """Build a minimal ConfigDict with the two training flags at their defaults."""
    from tinyml_modelmaker.utils.config_dict import ConfigDict
    return ConfigDict(dict(training=dict(compile_model=0, native_amp=False)))


def test_auto_enables_both_on_cuda_with_no_explicit_keys():
    from tinyml_modelmaker.utils.hardware_defaults import apply_hardware_defaults
    params = _make_params()
    with patch('torch.cuda.is_available', return_value=True):
        apply_hardware_defaults(params, set())
    assert params.training.compile_model == 1
    assert params.training.native_amp is True


def test_respects_explicit_native_amp_false():
    from tinyml_modelmaker.utils.hardware_defaults import apply_hardware_defaults
    params = _make_params()
    with patch('torch.cuda.is_available', return_value=True):
        apply_hardware_defaults(params, {'native_amp'})
    assert params.training.compile_model == 1   # still auto-enabled
    assert params.training.native_amp is False  # explicit choice respected


def test_respects_explicit_compile_model_zero():
    from tinyml_modelmaker.utils.hardware_defaults import apply_hardware_defaults
    params = _make_params()
    with patch('torch.cuda.is_available', return_value=True):
        apply_hardware_defaults(params, {'compile_model'})
    assert params.training.compile_model == 0   # explicit choice respected
    assert params.training.native_amp is True   # still auto-enabled


def test_no_change_without_cuda():
    from tinyml_modelmaker.utils.hardware_defaults import apply_hardware_defaults
    params = _make_params()
    with patch('torch.cuda.is_available', return_value=False):
        apply_hardware_defaults(params, set())
    assert params.training.compile_model == 0
    assert params.training.native_amp is False


def test_safe_when_params_lacks_flags():
    """hasattr guards: calling on a params dict without compile_model/native_amp must not raise."""
    from tinyml_modelmaker.utils.config_dict import ConfigDict
    from tinyml_modelmaker.utils.hardware_defaults import apply_hardware_defaults
    params = ConfigDict(dict(training=dict(batch_size=32)))
    with patch('torch.cuda.is_available', return_value=True):
        apply_hardware_defaults(params, set())  # must not raise
    assert params.training.batch_size == 32
```

- [ ] **Step 2: Run tests — confirm they all fail with ImportError**

```bash
cd tinyml-modelmaker
python -m pytest tests/test_hardware_defaults.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'apply_hardware_defaults'`

- [ ] **Step 3: Create `hardware_defaults.py`**

Create `tinyml-modelmaker/tinyml_modelmaker/utils/hardware_defaults.py`:

```python
import torch


def apply_hardware_defaults(params, explicitly_set: set) -> None:
    """Auto-enable compile_model and native_amp when CUDA is available.

    Skips fields present in explicitly_set — those are deliberate user
    choices from the YAML config and must not be overridden.
    hasattr guards keep this safe for params that don't carry these fields
    yet (vision, audio — Phase 2).
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

- [ ] **Step 4: Run tests — confirm all five pass**

```bash
python -m pytest tests/test_hardware_defaults.py -v
```

Expected output:
```
PASSED tests/test_hardware_defaults.py::test_auto_enables_both_on_cuda_with_no_explicit_keys
PASSED tests/test_hardware_defaults.py::test_respects_explicit_native_amp_false
PASSED tests/test_hardware_defaults.py::test_respects_explicit_compile_model_zero
PASSED tests/test_hardware_defaults.py::test_no_change_without_cuda
PASSED tests/test_hardware_defaults.py::test_safe_when_params_lacks_flags
5 passed
```

- [ ] **Step 5: Commit**

```bash
git add tinyml-modelmaker/tinyml_modelmaker/utils/hardware_defaults.py \
        tinyml-modelmaker/tests/test_hardware_defaults.py
git commit -m "feat: add apply_hardware_defaults — auto-enable compile/AMP on CUDA"
```

---

## Task 2: Wire into `timeseries/params.py` + integration test

**Files:**
- Modify: `tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/params.py`
- Create: `tinyml-modelmaker/tests/test_hardware_defaults_integration.py`

**Interfaces:**
- Consumes: `apply_hardware_defaults(params, explicitly_set)` from Task 1
- Produces: `init_params(*args, **kwargs)` now auto-sets flags when CUDA present and user hasn't overridden

---

- [ ] **Step 1: Write the failing integration tests**

Create `tinyml-modelmaker/tests/test_hardware_defaults_integration.py`:

```python
from unittest.mock import patch
import pytest


def test_init_params_auto_enables_on_cuda():
    """init_params with no training overrides auto-enables both flags on CUDA."""
    from tinyml_modelmaker.ai_modules.timeseries.params import init_params
    user_config = dict(common=dict(task_category='timeseries_classification'))
    with patch('torch.cuda.is_available', return_value=True):
        params = init_params(user_config)
    assert params.training.compile_model == 1
    assert params.training.native_amp is True


def test_init_params_respects_native_amp_false_override():
    """Explicit native_amp: false in user config is not overridden."""
    from tinyml_modelmaker.ai_modules.timeseries.params import init_params
    user_config = dict(
        common=dict(task_category='timeseries_classification'),
        training=dict(native_amp=False),
    )
    with patch('torch.cuda.is_available', return_value=True):
        params = init_params(user_config)
    assert params.training.native_amp is False
    assert params.training.compile_model == 1  # still auto-enabled


def test_init_params_no_change_without_cuda():
    """Without CUDA, both flags stay at defaults."""
    from tinyml_modelmaker.ai_modules.timeseries.params import init_params
    user_config = dict(common=dict(task_category='timeseries_classification'))
    with patch('torch.cuda.is_available', return_value=False):
        params = init_params(user_config)
    assert params.training.compile_model == 0
    assert params.training.native_amp is False
```

- [ ] **Step 2: Run tests — confirm they all fail**

```bash
python -m pytest tests/test_hardware_defaults_integration.py -v 2>&1 | head -20
```

Expected: all three fail with `AssertionError` (flags stay at defaults because `apply_hardware_defaults` not yet called from `init_params`).

- [ ] **Step 3: Modify `timeseries/params.py`**

In `tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/params.py`, make two changes:

**Add import** after the existing imports (line ~37):

```python
import os

from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion, TinyMLQuantizationMethod

from ... import utils
from . import constants
from ...utils.hardware_defaults import apply_hardware_defaults  # add this line
```

**Modify the end of `init_params`** — replace the existing two-line tail:

```python
    params = utils.ConfigDict(default_params, *args, **kwargs)
    return params
```

with:

```python
    user_training_keys = set(args[0].get('training', {}).keys()) if args else set()
    params = utils.ConfigDict(default_params, *args, **kwargs)
    apply_hardware_defaults(params, user_training_keys)
    return params
```

- [ ] **Step 4: Run integration tests — confirm all three pass**

```bash
python -m pytest tests/test_hardware_defaults_integration.py -v
```

Expected:
```
PASSED tests/test_hardware_defaults_integration.py::test_init_params_auto_enables_on_cuda
PASSED tests/test_hardware_defaults_integration.py::test_init_params_respects_native_amp_false_override
PASSED tests/test_hardware_defaults_integration.py::test_init_params_no_change_without_cuda
3 passed
```

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/params.py \
        tinyml-modelmaker/tests/test_hardware_defaults_integration.py
git commit -m "feat: wire apply_hardware_defaults into timeseries init_params"
```

---

## Self-Review Checklist

- **Spec coverage:**
  - ✅ `hardware_defaults.py` created with `apply_hardware_defaults`
  - ✅ CUDA-only detection (`torch.cuda.is_available`)
  - ✅ YAML override rule tested (explicit key → skip, implicit → auto-set)
  - ✅ `hasattr` guards for future vision/audio expansion tested
  - ✅ Wired into `timeseries/params.py` via user key extraction before `ConfigDict` construction
  - ✅ MPS excluded (condition is CUDA only — no MPS branch)
  - ✅ Phase 2 out of scope — not in plan
- **Placeholder scan:** None found
- **Type consistency:** `apply_hardware_defaults(params, explicitly_set: set)` used identically in Task 1 and Task 2
