# Unwrap torch.compile Before ONNX Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `export_model()` crashing with `RuntimeError: Detected that you are using FX to torch.jit.trace a dynamo-optimized function` whenever the model passed in was wrapped by `torch.compile` (i.e. whenever `compile_model=1` — the default `hardware-defaults` now auto-enables on CUDA — actually succeeds).

**Architecture:** `torch.compile()` wraps a model in `torch._dynamo.OptimizedModule`, which exposes the original, uncompiled module at `._orig_mod`. Neither `torch.onnx.export` (used for float export) nor `torch.jit.trace` (used for quantized export) can trace a dynamo-optimized module. `export_model()` (`tinyml-tinyverse/tinyml_tinyverse/common/utils/utils.py:1618`) never unwraps this, so it always crashes on a compiled model, in both the quantized and float export branches. Fix: unwrap via `getattr(model, '_orig_mod', model)` once, right before the existing `copy.deepcopy(model)` call — safe no-op for uncompiled models (`getattr` falls back to `model` itself when `_orig_mod` doesn't exist).

**Tech Stack:** Python 3.10, PyTorch 2.7.1, pytest

## Global Constraints

- Python `==3.10.*`
- No new dependencies
- Zero behavior change for uncompiled models (the common case today, since `compile_model` still defaults to 0 unless hardware-defaults auto-enables it)
- Must handle BOTH export branches in `export_model()` — the `quantization` truthy branch (`model_copy.export(...)` + `torch.jit.trace`) and the `else` branch (`torch.onnx.export`) — both call `torch.jit`-based tracing internally and both are broken by a compiled model today

## Context: How This Was Found

Discovered via real-hardware validation on the GX10 (NVIDIA GB10) after fixing a separate, unrelated bug (`persistent_workers`/deepcopy crash, commit `dc1509d`) that had been masking this one — training with `compile_model=1` would previously crash before ever reaching `export_model()`. With that masking bug fixed, `tinyml-modelmaker/tests/test_pipeline_smoke.py`'s three training smoke tests (classification, regression, forecasting) now fail with:
```
RuntimeError: Detected that you are using FX to torch.jit.trace a dynamo-optimized function. This is not supported at the moment.
```
This happens because `apply_hardware_defaults` (the hardware-defaults feature) auto-enables `compile_model=1` on any machine where `torch.cuda.is_available()` is True — including this GX10, where the pipeline smoke tests otherwise run on CPU (`num_gpus=0` in their config) but the machine itself has CUDA available, so compile still gets auto-enabled and successfully compiles (unlike the earlier Blackwell/ptxas failure, which is GPU-execution-specific and doesn't block CPU-side compilation attempts under `aot_eager`/whatever backend applies).

## File Map

| Action | Path | Responsibility |
|--------|------|-----------------|
| Modify | `tinyml-tinyverse/tinyml_tinyverse/common/utils/utils.py` | `export_model()` unwraps `_orig_mod` before use |
| Create | `tinyml-tinyverse/tests/test_export_model_unwrap.py` | Regression test for both export branches |

---

## Task 1: Unwrap fix + tests

**Files:**
- Modify: `tinyml-tinyverse/tinyml_tinyverse/common/utils/utils.py` (function starts at line 1618)
- Create: `tinyml-tinyverse/tests/test_export_model_unwrap.py`

**Interfaces:**
- No signature change to `export_model()` — the fix is internal

---

- [ ] **Step 1: Read the current implementation for exact line numbers**

```bash
sed -n '1618,1660p' tinyml-tinyverse/tinyml_tinyverse/common/utils/utils.py
```

Confirm the function body matches what's described in Context above (line numbers may have drifted slightly).

- [ ] **Step 2: Write the failing test**

Create `tinyml-tinyverse/tests/test_export_model_unwrap.py`:

```python
"""Regression test for: export_model() crashing on a torch.compile-wrapped
model with `RuntimeError: Detected that you are using FX to torch.jit.trace
a dynamo-optimized function. This is not supported at the moment.`

Root cause: torch.compile() wraps a model in torch._dynamo.OptimizedModule,
which exposes the original module at ._orig_mod. Neither torch.onnx.export
nor torch.jit.trace (both used inside export_model, depending on whether
quantization is enabled) can trace a dynamo-optimized module directly.
"""
import os
import tempfile

import torch
import torch.nn as nn

from tinyml_tinyverse.common.utils.utils import export_model


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def test_export_model_handles_compiled_model_float_path():
    """The non-quantized (else) branch: torch.onnx.export must not choke on
    a compiled model."""
    model = _TinyModel()
    compiled_model = torch.compile(model, backend='aot_eager')
    # Trigger real compilation so we're testing an actual OptimizedModule,
    # not just the lazy uncalled wrapper.
    compiled_model(torch.rand(1, 4))

    with tempfile.TemporaryDirectory() as tmpdir:
        export_model(compiled_model, input_shape=(1, 4), output_dir=tmpdir, quantization=0)
        assert os.path.exists(os.path.join(tmpdir, 'model.onnx'))


def test_export_model_uncompiled_model_still_works():
    """Backward compatibility: an ordinary, uncompiled model must still
    export exactly as before (the getattr fallback is a no-op)."""
    model = _TinyModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        export_model(model, input_shape=(1, 4), output_dir=tmpdir, quantization=0)
        assert os.path.exists(os.path.join(tmpdir, 'model.onnx'))
```

- [ ] **Step 3: Run the tests to confirm they fail for the expected reason**

```bash
cd tinyml-tinyverse
python -m pytest tests/test_export_model_unwrap.py -v 2>&1 | tail -30
```

Expected: `test_export_model_handles_compiled_model_float_path` FAILS with `RuntimeError: Detected that you are using FX to torch.jit.trace a dynamo-optimized function...` (or an equivalent dynamo/ONNX export error — the exact wording may vary slightly by backend, but the failure must originate from attempting to export a compiled model). `test_export_model_uncompiled_model_still_works` should already PASS (it's a regression guard for current behavior, not new functionality).

- [ ] **Step 4: Implement the fix**

In `tinyml-tinyverse/tinyml_tinyverse/common/utils/utils.py`, in `export_model()`, change:
```python
    model_copy = copy.deepcopy(model)
```
to:
```python
    # torch.compile() wraps a model in torch._dynamo.OptimizedModule, exposing
    # the original module at ._orig_mod. Neither torch.jit.trace (used below
    # for quantized export) nor torch.onnx.export (used for float export) can
    # trace a dynamo-optimized module directly, so unwrap first. This is a
    # no-op for uncompiled models (getattr falls back to model itself).
    model = getattr(model, '_orig_mod', model)
    model_copy = copy.deepcopy(model)
```

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
cd tinyml-tinyverse
python -m pytest tests/test_export_model_unwrap.py -v
```

Expected: both tests pass.

- [ ] **Step 6: Run the full existing test suite for regressions**

```bash
cd tinyml-tinyverse
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

```bash
cd tinyml-modelmaker
python -m pytest tests/ -v --tb=short --ignore=tests/test_protocols.py 2>&1 | tail -40
```

Expected: `tests/test_pipeline_smoke.py::TestClassificationSmoke::test_classification_trains`, `TestRegressionSmoke::test_regression_trains`, and `TestForecastingSmoke::test_forecasting_trains` now PASS (previously the 3 failures this plan targets). The only remaining pre-existing failures should be the 5 `F28E12` device-profile tests in `test_cross_device.py` (a separate, already-known, intentionally-deferred gap — real hardware values needed, not something this plan touches).

- [ ] **Step 7: Commit**

```bash
git add tinyml-tinyverse/tinyml_tinyverse/common/utils/utils.py \
        tinyml-tinyverse/tests/test_export_model_unwrap.py
git commit -m "fix: unwrap torch.compile wrapper before ONNX/TorchScript export"
```

---

## Task 2: Real-hardware verification on the GX10

**Files:** none (verification only)

---

- [ ] **Step 1: Sync the branch to the GX10**

From the local machine:
```bash
git push origin HEAD:integration
ssh martin@REDACTED-DEV-HOST "cd ~/repos/tinyml-tensorlab && git pull"
```

- [ ] **Step 2: Re-run the full modelmaker test suite on the GX10**

```bash
ssh martin@REDACTED-DEV-HOST "cd ~/repos/tinyml-tensorlab/tinyml-modelmaker && ~/repos/tinyml-tensorlab/.venv/bin/python -m pytest tests/ -v --tb=short --ignore=tests/test_protocols.py 2>&1 | tail -20"
```

Expected: only the 5 pre-existing, known `F28E12` failures remain — the 3 pipeline-smoke failures this plan targets are gone.

- [ ] **Step 3: Re-run the quantization + compile validation script**

The script from earlier hardware-defaults validation is at `tinyml-modelmaker/validate_compile_amp_quant.py` on the GX10.

```bash
ssh martin@REDACTED-DEV-HOST "cd ~/repos/tinyml-tensorlab/tinyml-modelmaker && ~/repos/tinyml-tensorlab/.venv/bin/python validate_compile_amp_quant.py > /tmp/validation_final.log 2>&1; echo EXIT:\$?"
ssh martin@REDACTED-DEV-HOST "grep -A 15 '^SUMMARY$' /tmp/validation_final.log"
```

Report the outcome — whether the quantized+compiled path (`treatment_compile_and_amp` config, which uses `quantization=2`) now succeeds end-to-end, or hits a still-different failure (in which case, report exactly what it is; do not assume success without reading the actual log).

- [ ] **Step 4: Report findings**

No code changes in this task — report the before/after comparison as the closing verification for this plan.

---

## Self-Review Checklist

- **Spec coverage:**
  - ✅ Unwrap fix covers both export branches (quantization truthy and the else/float branch) — both call the same `model = getattr(...)` line before either branch runs
  - ✅ Backward compatible: no-op for uncompiled models, verified by a dedicated test
  - ✅ Real-hardware verification step included, including the quantization+compile combination that originally surfaced the persistent_workers bug
- **Placeholder scan:** None found
- **Type consistency:** N/A (no new function signatures)
