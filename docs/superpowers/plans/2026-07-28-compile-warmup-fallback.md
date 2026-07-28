# torch.compile Warmup Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `compile_model_if_enabled`'s existing try/except actually catch `torch.compile` failures, by running a real warmup forward pass at model-setup time instead of only wrapping the (lazy, no-op) `torch.compile()` call itself.

**Architecture:** `compile_model_if_enabled` gains an `input_shape` parameter. When compilation is enabled, it wraps the model, builds a dummy input tensor of that shape, and runs one forward pass through the compiled model inside the existing try/except, in eval/no_grad mode. If the warmup pass raises (the actual point where Inductor/Triton/ptxas compile failures surface — confirmed via real hardware reproduction, see Context below), the function falls back to returning the original uncompiled model instead of the broken compiled wrapper. All 4 reference train scripts (timeseries classification, regression, forecasting, anomalydetection) pass `input_shape = (1,) + dataset.X.shape[1:]` at their call site — this expression is already used elsewhere in each script for ONNX export, so it's a known-correct pattern per task type.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, pytest

## Global Constraints

- Python `==3.10.*`
- No new dependencies
- The fix must preserve today's behavior when `compile_model` is disabled (the default) — zero change to the non-compile path
- The warmup pass must not mutate the model's parameters or training state (no optimizer step; restore original `.training` mode after the pass)
- Must not regress the 4 existing call sites' argument order — `input_shape` is a new keyword-only-by-convention parameter, added last, defaulting to `None` (preserves backward compatibility for any other caller)

## Context: Why This Is Needed

Reproduced on real CUDA hardware (NVIDIA GB10, `torch==2.7.1+cu128`): auto-enabling `compile_model=1` crashes with:
```
torch._inductor.exc.InductorError: RuntimeError: A compilation subprocess exited unexpectedly.
...
ptxas-blackwell fatal: Value 'sm_121a' is not defined for option 'gpu-name'
```
The crash surfaces at `output = model(data)` inside `train_one_epoch_classification` (`tinyml-tinyverse/tinyml_tinyverse/common/utils/utils.py:1494`) — **not** inside `compile_model_if_enabled`'s existing try/except (`train_base.py:687-690`), because `torch.compile()` is lazy: it doesn't compile anything until the first forward call. The existing try/except only guards the (always-succeeding) wrapping call, giving false confidence.

This fix moves the actual compilation trigger (a forward pass) inside the guarded block, so a hardware/toolchain incompatibility is caught at setup time and falls back to eager mode — instead of crashing mid-training.

---

## File Map

| Action | Path | Responsibility |
|--------|------|-----------------|
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py` | `compile_model_if_enabled` gains warmup pass + real fallback |
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/timeseries_classification/train.py` | Pass `input_shape` at call site |
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/timeseries_regression/train.py` | Pass `input_shape` at call site |
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/timeseries_forecasting/train.py` | Pass `input_shape` at call site |
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/timeseries_anomalydetection/train.py` | Pass `input_shape` at call site |
| Create | `tinyml-tinyverse/tests/test_compile_warmup_fallback.py` | Unit tests for the warmup/fallback logic |

---

## Task 1: `compile_model_if_enabled` warmup + fallback, with tests

**Files:**
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py` (function starts at line 661)
- Create: `tinyml-tinyverse/tests/test_compile_warmup_fallback.py`

**Interfaces:**
- Produces: `compile_model_if_enabled(model, args, logger, input_shape=None)` — same return type as today (a `torch.nn.Module`), new optional 4th parameter

---

- [ ] **Step 1: Read the current implementation for exact line numbers**

```bash
sed -n '655,695p' tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py
```

Confirm the function body matches what's described in Context above before editing (line numbers may have drifted slightly since this plan was written).

- [ ] **Step 2: Write the failing tests**

Create `tinyml-tinyverse/tests/test_compile_warmup_fallback.py`. This test file requires a CUDA-independent way to force compile failure — use `unittest.mock.patch('torch.compile')` to simulate both a wrap-time failure and a warmup-forward-time failure, since real Inductor/Triton failures are hardware-specific and can't be reproduced in CI:

```python
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import pytest


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class _FakeArgs:
    def __init__(self, compile_model):
        self.compile_model = compile_model


def _get_logger():
    import logging
    return logging.getLogger("test_compile_warmup_fallback")


def test_compile_disabled_returns_original_model():
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=0)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    assert result is model


def test_compile_success_with_warmup_returns_compiled_model():
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    # torch.compile wraps in an OptimizedModule; on CPU with a trivial model
    # this should succeed genuinely (no mocking needed — real compile on a
    # tiny CPU model is fast and should not fail).
    assert result is not None
    out = result(torch.rand(1, 4))
    assert out.shape == (1, 2)


def test_warmup_failure_falls_back_to_original_model():
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1)

    class _BrokenCompiledModel(nn.Module):
        def forward(self, x):
            raise RuntimeError("simulated Inductor/Triton compile failure")

    with patch('torch.compile', return_value=_BrokenCompiledModel()):
        result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))

    # Must fall back to the ORIGINAL model, not the broken compiled one.
    assert result is model
    out = result(torch.rand(1, 4))
    assert out.shape == (1, 2)


def test_wrap_time_failure_falls_back_to_original_model():
    """torch.compile() itself raising (not just the warmup forward) is still caught."""
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1)

    with patch('torch.compile', side_effect=RuntimeError("simulated wrap-time failure")):
        result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))

    assert result is model


def test_no_input_shape_skips_warmup_but_still_compiles():
    """Backward compatibility: callers that don't pass input_shape get the
    old behavior (compile attempted, no warmup, no new fallback coverage)."""
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1)
    result = compile_model_if_enabled(model, args, _get_logger())  # no input_shape
    assert result is not None


def test_warmup_restores_original_training_mode():
    """The warmup pass must not leave the model stuck in eval mode."""
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    model.train()
    args = _FakeArgs(compile_model=1)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    assert result.training is True
```

- [ ] **Step 3: Run tests to confirm they fail for the expected reason**

```bash
cd tinyml-tinyverse
python -m pytest tests/test_compile_warmup_fallback.py -v 2>&1 | tail -30
```

Expected: `test_compile_disabled_returns_original_model` may pass trivially (no behavior change needed there); the rest should fail — either with `TypeError: compile_model_if_enabled() got an unexpected keyword argument 'input_shape'` (parameter doesn't exist yet) or with assertion failures once the parameter is silently accepted by `**kwargs` (it isn't — confirm the exact failure mode matches "parameter doesn't exist").

- [ ] **Step 4: Implement the fix**

Replace the current `compile_model_if_enabled` function body in `tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py` with:

```python
def compile_model_if_enabled(model, args, logger, input_shape=None):
    """
    Apply torch.compile to the model if --compile-model is enabled.

    torch.compile (PyTorch 2.0+) fuses operations into optimized kernels,
    which can significantly speed up training (15-30% on supported backends).

    torch.compile() itself is lazy — it does not compile anything until the
    first forward call. To catch compile failures (e.g. a Triton/ptxas
    version that doesn't yet support the GPU's compute capability) before
    training starts rather than mid-epoch, this function runs one warmup
    forward pass through the compiled model when input_shape is provided.
    A failure at either the wrap step or the warmup step falls back to the
    original, uncompiled model.

    Args:
        model: The model to potentially compile
        args: Parsed arguments (uses args.compile_model)
        logger: Logger instance
        input_shape: Shape (including batch dim) of a representative input
            tensor, e.g. (1,) + dataset.X.shape[1:]. Used to run a warmup
            forward pass that validates compilation actually works on this
            hardware/toolchain. If None, no warmup is performed and a
            compile failure will surface later, unguarded, on the training
            loop's first real forward pass (legacy behavior).

    Returns:
        The (possibly compiled) model
    """
    if getattr(args, 'compile_model', 0) and hasattr(torch, 'compile'):
        # Determine the best backend for the current device
        device_type = str(next(model.parameters()).device).split(':')[0] if len(list(model.parameters())) > 0 else 'cpu'
        if device_type == 'mps':
            # MPS supports torch.compile via the 'aot_eager' backend
            backend = 'aot_eager'
        elif device_type == 'cuda':
            backend = 'inductor'
        else:
            backend = 'aot_eager'
        logger.info(f"Compiling model with torch.compile (backend={backend})")
        original_model = model
        try:
            compiled_model = torch.compile(model, backend=backend)
            if input_shape is not None:
                device = next(compiled_model.parameters()).device if len(list(compiled_model.parameters())) > 0 else torch.device('cpu')
                dummy_input = torch.rand(size=input_shape, device=device)
                was_training = compiled_model.training
                compiled_model.eval()
                with torch.no_grad():
                    compiled_model(dummy_input)
                compiled_model.train(was_training)
            model = compiled_model
        except Exception as e:
            logger.warning(f"torch.compile failed (or failed its warmup pass), falling back to eager mode: {e}")
            model = original_model
    return model
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd tinyml-tinyverse
python -m pytest tests/test_compile_warmup_fallback.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py \
        tinyml-tinyverse/tests/test_compile_warmup_fallback.py
git commit -m "fix: validate torch.compile with a warmup pass so failures actually fall back to eager mode"
```

---

## Task 2: Wire `input_shape` through all 4 reference train scripts

**Files:**
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/timeseries_classification/train.py` (call site at line 275)
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/timeseries_regression/train.py` (call site at line 190)
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/timeseries_forecasting/train.py` (call site at line 197)
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/timeseries_anomalydetection/train.py` (call site at line 232)

**Interfaces:**
- Consumes: `compile_model_if_enabled(model, args, logger, input_shape=None)` from Task 1

---

- [ ] **Step 1: Confirm `dataset.X.shape` is in scope at each call site**

```bash
cd tinyml-tinyverse/tinyml_tinyverse/references
for f in timeseries_classification timeseries_regression timeseries_forecasting timeseries_anomalydetection; do
  echo "=== $f ==="
  grep -n "compile_model_if_enabled(model, args, logger)\|dataset.X.shape" "$f/train.py"
done
```

Confirm each file has a `variables = dataset.X.shape[1]` line *before* its `compile_model_if_enabled` call (this establishes `dataset` is already loaded and in scope by that point). If line numbers have drifted from what's listed above, locate the call site by searching for the exact string `compile_model_if_enabled(model, args, logger)`.

- [ ] **Step 2: Update each of the 4 call sites**

In each of the 4 files, change:
```python
        model = compile_model_if_enabled(model, args, logger)
```
to:
```python
        model = compile_model_if_enabled(model, args, logger, input_shape=(1,) + dataset.X.shape[1:])
```

This is a mechanical one-line change repeated identically across all 4 files — the expression `(1,) + dataset.X.shape[1:]` is copied verbatim into each, matching the pattern already used for ONNX export input_shape construction elsewhere in these same files (e.g. `timeseries_anomalydetection/train.py:313`, `timeseries_classification/train.py:388`).

- [ ] **Step 3: Verify no other call sites were missed**

```bash
cd tinyml-tinyverse
grep -rn "compile_model_if_enabled(model, args, logger)$" tinyml_tinyverse/references/*/train.py
```

Expected: no output (all call sites now pass `input_shape=...`; any remaining bare `compile_model_if_enabled(model, args, logger)` without the new argument means a call site was missed).

- [ ] **Step 4: Run the full existing test suite for regressions**

```bash
cd tinyml-modelmaker
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: same pass/fail counts as the pre-existing baseline (no new failures introduced by this change — this change doesn't touch anything in `tinyml-modelmaker`, only `tinyml-tinyverse`, so no interaction is expected, but confirm nothing broke).

```bash
cd tinyml-tinyverse
python -m pytest tests/test_compile_warmup_fallback.py -v
```

Expected: all 6 tests from Task 1 still pass.

- [ ] **Step 5: Commit**

```bash
git add tinyml-tinyverse/tinyml_tinyverse/references/timeseries_classification/train.py \
        tinyml-tinyverse/tinyml_tinyverse/references/timeseries_regression/train.py \
        tinyml-tinyverse/tinyml_tinyverse/references/timeseries_forecasting/train.py \
        tinyml-tinyverse/tinyml_tinyverse/references/timeseries_anomalydetection/train.py
git commit -m "fix: pass input_shape to compile_model_if_enabled at all 4 call sites"
```

---

## Task 3: Real-hardware verification on the GX10

**Files:** none (verification only, no code changes)

**Interfaces:**
- Consumes: the fixed `compile_model_if_enabled` from Tasks 1-2

---

- [ ] **Step 1: Sync the branch to the GX10**

From the local machine (not the GX10):
```bash
git push origin HEAD:integration
ssh martin@REDACTED-DEV-HOST "cd ~/repos/tinyml-tensorlab && git pull"
```

- [ ] **Step 2: Re-run the validation script from the earlier hardware-defaults investigation**

The script that originally reproduced the crash is at `tinyml-modelmaker/validate_compile_amp_quant.py` on the GX10 (already copied there in a prior session). Re-run it:

```bash
ssh martin@REDACTED-DEV-HOST "cd ~/repos/tinyml-tensorlab/tinyml-modelmaker && ~/repos/tinyml-tensorlab/.venv/bin/python validate_compile_amp_quant.py > /tmp/validation_output_after_fix.log 2>&1; echo EXIT:\$?"
```

- [ ] **Step 3: Confirm the treatment run (compile_model=1, native_amp=True) no longer crashes**

```bash
ssh martin@REDACTED-DEV-HOST "grep -A 10 'treatment_compile_and_amp' /tmp/validation_output_after_fix.log | tail -20"
```

Expected: instead of an `InductorError` crash, the log should show a `logger.warning` message about torch.compile falling back to eager mode (from `train_base.py`'s except block), and `training_succeeded` in the summary should no longer show an `InductorError` exception — though note the pre-existing, unrelated `export_model`/`persistent_workers` deepcopy bug (tracked separately as `task_45097026`) may still cause the run to fail at a *later* stage (export). Confirm specifically that the failure mode has changed from "crash during training due to Inductor" to either "success" or "crash during export due to the separate, already-tracked bug" — either outcome confirms this fix worked; only a recurrence of the `InductorError` during training would mean the fix didn't work.

- [ ] **Step 4: Report findings**

No code changes in this task — report the before/after comparison (crash → graceful fallback) as the closing verification for this plan.

---

## Self-Review Checklist

- **Spec coverage:**
  - ✅ Warmup forward pass added inside the existing try/except
  - ✅ Falls back to original uncompiled model (not the broken compiled wrapper) on any failure — both wrap-time and warmup-time
  - ✅ Backward compatible: `input_shape=None` preserves today's behavior exactly
  - ✅ All 4 reference scripts wired
  - ✅ Training mode restored after warmup (no side effect on the model's `.training` flag)
  - ✅ Real-hardware verification step included (Task 3) — this is what makes the fix trustworthy, since the original bug was only found via real hardware, not mocks
- **Placeholder scan:** None found
- **Type consistency:** `compile_model_if_enabled(model, args, logger, input_shape=None)` signature used identically in Task 1 and Task 2
