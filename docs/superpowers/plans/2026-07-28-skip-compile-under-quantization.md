# Skip torch.compile When Quantization Is Enabled Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a crash where `compile_model=1` combined with FX-based quantization (`quantization=1` or `quantization=2`) crashes training before it starts, because `prepare_qat_fx` (FX symbolic tracing) cannot trace a `torch.compile`-wrapped module.

**Architecture:** `compile_model_if_enabled` (`tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py`) gains one additional guard condition: skip compiling (and log why) whenever `args.quantization` is truthy, regardless of `args.compile_model`. This is a single-point fix — all 4 reference train scripts call through this one function, so no per-call-site changes are needed (unlike the earlier `input_shape` wiring, which needed 4 edits).

**Tech Stack:** Python 3.10, PyTorch 2.7.1, pytest

## Global Constraints

- Python `==3.10.*`
- No new dependencies
- Zero behavior change when `compile_model=0` (today's default) or when `quantization=0` (float training) — this fix only changes behavior for the specific combination of `compile_model=1` AND `quantization` truthy
- Must not silently do nothing — log at INFO level why compile was skipped, so a user who explicitly set `--compile-model 1` on a quantized run understands why they got eager mode instead of a crash

## Context: Why This Is Needed

Reproduced independently (by a final whole-branch code reviewer, via direct execution with the exact production call ordering — real `compile_model_if_enabled`, real `NeuralNetworkWithPreprocess` wrapping, real `quantization_wrapped_model`) that `compile_model=1` combined with `quantization=1` or `quantization=2` crashes with:
```
RuntimeError: Detected that you are using FX to symbolically trace a dynamo-optimized function. This is not supported at the moment.
```
This happens because all 4 timeseries reference scripts call `compile_model_if_enabled` (which wraps the model in `torch._dynamo.OptimizedModule` when it succeeds) *before* calling `quantization_wrapped_model`, which runs `prepare_qat_fx` — FX symbolic tracing, which cannot trace a dynamo-optimized module at all (this is a fundamentally different incompatibility than the ONNX/TorchScript export tracing issue fixed separately; unwrapping the compiled model right before quantization prep would work mechanically, but would also mean torch.compile provides no benefit for the rest of training on a quantized run, since the un-compiled original model is what continues through training — so skipping compile entirely for quantized runs is the simpler, equally-effective choice that avoids relying on unwrap-then-somehow-recompile complexity).

This matters because `quantization: 2` (TINPU) appears in nearly every shipped `tinyml-modelzoo` example config — this is the mainstream configuration, not an edge case. It went undetected in this branch's own real-hardware validation because the validation hardware (an early-generation Blackwell GPU) couldn't successfully compile at all, so every "compile enabled" validation run silently exercised the eager-fallback path (from the separate warmup-fallback fix) rather than the actual compile-success path where this bug lives.

## File Map

| Action | Path | Responsibility |
|--------|------|-----------------|
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py` | `compile_model_if_enabled` skips compile when quantization is enabled |
| Create | `tinyml-tinyverse/tests/test_compile_skipped_under_quantization.py` | Regression test for the guard |

---

## Task 1: Guard + tests

**Files:**
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py` (function starts at line 661, per the compile-warmup-fallback plan's most recent edit)
- Create: `tinyml-tinyverse/tests/test_compile_skipped_under_quantization.py`

**Interfaces:**
- No signature change to `compile_model_if_enabled(model, args, logger, input_shape=None)` — the fix only adds an internal condition

---

- [ ] **Step 1: Read the current implementation**

```bash
sed -n '661,745p' tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py
```

Confirm the function currently starts compiling with:
```python
if getattr(args, 'compile_model', 0) and hasattr(torch, 'compile'):
```
(This is the state after the compile-warmup-fallback plan's Task 1 and its training-mode-restore fix — confirm the exact current line numbers before editing, they may have drifted.)

- [ ] **Step 2: Write the failing tests**

Create `tinyml-tinyverse/tests/test_compile_skipped_under_quantization.py`:

```python
"""Regression test for: compile_model=1 combined with FX-based quantization
(quantization=1 or 2) crashing at prepare_qat_fx time with
`RuntimeError: Detected that you are using FX to symbolically trace a
dynamo-optimized function. This is not supported at the moment.`

Root cause: torch.compile() wraps the model in torch._dynamo.OptimizedModule
before quantization_wrapped_model() runs prepare_qat_fx on it. FX symbolic
tracing cannot trace a dynamo-optimized module at all. This is a different
incompatibility than the ONNX/TorchScript export tracing issue (fixed
separately in export_model()) -- unwrapping right before quantization would
work mechanically but would mean compile provides no benefit for the rest
of a quantized run's training, so this fix skips compiling entirely
whenever quantization is enabled.
"""
import torch
import torch.nn as nn


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class _FakeArgs:
    def __init__(self, compile_model, quantization):
        self.compile_model = compile_model
        self.quantization = quantization


def _get_logger():
    import logging
    return logging.getLogger("test_compile_skipped_under_quantization")


def test_compile_skipped_when_quantization_enabled():
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1, quantization=2)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    # Must be the ORIGINAL model, not compiled -- an OptimizedModule here
    # would go on to crash prepare_qat_fx's FX symbolic trace.
    assert result is model
    assert not hasattr(result, '_orig_mod')


def test_compile_skipped_when_quantization_is_ptq_mode():
    """quantization=1 (generic PTQ/QAT) hits the same FX-trace incompatibility
    as quantization=2 (TINPU) -- both go through prepare_qat_fx."""
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1, quantization=1)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    assert result is model


def test_compile_still_happens_for_float_training():
    """Zero behavior change for the common case: quantization=0 (float
    training) still compiles exactly as before."""
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=1, quantization=0)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    assert result is not model  # genuinely compiled
    out = result(torch.rand(1, 4))
    assert out.shape == (1, 2)


def test_compile_disabled_and_quantization_enabled_is_still_a_noop():
    """When compile_model=0, quantization doesn't matter -- nothing changes."""
    from tinyml_tinyverse.references.common.train_base import compile_model_if_enabled
    model = _TinyModel()
    args = _FakeArgs(compile_model=0, quantization=2)
    result = compile_model_if_enabled(model, args, _get_logger(), input_shape=(1, 4))
    assert result is model
```

- [ ] **Step 3: Run the tests to confirm they fail for the expected reason**

```bash
cd tinyml-tinyverse
python -m pytest tests/test_compile_skipped_under_quantization.py -v 2>&1 | tail -30
```

Expected: `test_compile_skipped_when_quantization_enabled` and `test_compile_skipped_when_quantization_is_ptq_mode` FAIL (`result is model` assertion fails, because today's code compiles regardless of quantization). `test_compile_still_happens_for_float_training` and `test_compile_disabled_and_quantization_enabled_is_still_a_noop` should already PASS (they describe existing behavior).

- [ ] **Step 4: Implement the fix**

In `tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py`, change the `compile_model_if_enabled` function's opening condition from:
```python
    if getattr(args, 'compile_model', 0) and hasattr(torch, 'compile'):
```
to:
```python
    if getattr(args, 'quantization', 0):
        # FX-based quantization (prepare_qat_fx) symbolically traces the model,
        # which cannot trace a torch.compile-wrapped module at all -- a different
        # incompatibility than the ONNX/TorchScript export tracing issue handled
        # separately in export_model(). Skip compiling rather than compile and
        # then immediately discard the benefit before quantization prep runs.
        if getattr(args, 'compile_model', 0):
            logger.info(
                "compile_model is enabled but quantization is also enabled "
                "(FX-based quantization cannot trace a compiled model) -- "
                "skipping torch.compile for this run."
            )
        return model
    if getattr(args, 'compile_model', 0) and hasattr(torch, 'compile'):
```

(The rest of the function body — backend selection, warmup, fallback — stays exactly as-is beneath this new early-return guard.)

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
cd tinyml-tinyverse
python -m pytest tests/test_compile_skipped_under_quantization.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Run the full existing test suite for regressions**

```bash
cd tinyml-tinyverse
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

```bash
cd tinyml-modelmaker
python -m pytest tests/ -v --tb=short --ignore=tests/test_protocols.py 2>&1 | tail -20
```

Expected: no new failures. The only pre-existing failures should be the 5 known `F28E12` device-profile tests in `test_cross_device.py`.

- [ ] **Step 7: Commit**

```bash
git add tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py \
        tinyml-tinyverse/tests/test_compile_skipped_under_quantization.py
git commit -m "fix: skip torch.compile when quantization is enabled — FX tracing can't handle a compiled model"
```

---

## Task 2: Real-hardware verification — compile SUCCESS path with quantization

**Files:** none (verification only)

**Why this task matters:** every prior real-hardware validation in this branch's history ran on hardware where `torch.compile` could never actually succeed (an early-generation Blackwell GPU whose bundled Triton doesn't support its compute capability), so every "treatment" run silently exercised the eager-fallback path, not the actual compile-success path. This task must verify the ACTUAL fix on a path where compile would otherwise have succeeded and crashed at quantization — not another fallback run.

---

- [ ] **Step 1: Sync the branch to the GX10**

```bash
git push origin HEAD:integration
ssh martin@<gx10-host> "cd ~/repos/tinyml-tensorlab && git pull"
```

- [ ] **Step 2: Confirm the crash reproduces on unpatched CPU-side compilation (control)**

Before trusting the fix, confirm the bug is real and reachable in this exact environment by temporarily reverting the fix (e.g. `git stash` the Task 1 commit) and running a quantized pipeline smoke test with `compile_model=1` forced on. Since `tests/test_pipeline_smoke.py`'s existing configs use CPU (`num_gpus=0`) and `quantization=0`, this requires a small ad-hoc script (similar to the earlier `validate_compile_amp_quant.py` pattern used in this branch's history) that sets `training: {compile_model: 1, quantization: 2}` explicitly and runs a real training job — CPU-side `aot_eager` compilation succeeds even without CUDA, which is exactly what makes this a valid control: it does NOT depend on the GX10's broken CUDA compile path, so it proves the quantization-tracing incompatibility independently of the Blackwell/ptxas issue.

Expected (pre-fix, control): `RuntimeError: Detected that you are using FX to symbolically trace a dynamo-optimized function.`

- [ ] **Step 3: Confirm the fix resolves it**

Re-apply the fix (`git stash pop` or re-pull) and re-run the same ad-hoc script.

Expected (post-fix): training completes; log shows the new INFO message about skipping compile due to quantization; no crash.

- [ ] **Step 4: Re-run the full modelmaker test suite**

```bash
ssh martin@<gx10-host> "cd ~/repos/tinyml-tensorlab/tinyml-modelmaker && ~/repos/tinyml-tensorlab/.venv/bin/python -m pytest tests/ -v --tb=short --ignore=tests/test_protocols.py 2>&1 | tail -20"
```

Expected: only the 5 known `F28E12` failures remain.

- [ ] **Step 5: Report findings**

No code changes in this task — report the control-vs-fixed comparison, since this is the first validation in this branch's history that actually exercises the compile-success-then-quantize path rather than the eager-fallback path.

---

## Self-Review Checklist

- **Spec coverage:**
  - ✅ Guard added as a single early-return in `compile_model_if_enabled`, applies uniformly to all 4 call sites
  - ✅ Zero behavior change for `quantization=0` (float training, the common case) and for `compile_model=0`
  - ✅ Logs why compile was skipped (not silent)
  - ✅ Real-hardware verification explicitly targets the compile-SUCCESS path (via CPU-side `aot_eager`, which doesn't depend on the GX10's broken CUDA compile), not another fallback run — addressing the exact blind spot that let this bug through the branch's prior validation
- **Placeholder scan:** None found
- **Type consistency:** N/A (no new function signatures)
