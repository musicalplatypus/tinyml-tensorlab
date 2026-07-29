# Unwrap torch.compile Before Checkpoint Save/Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a silent correctness bug — checkpoints saved from a `torch.compile`-wrapped model carry `_orig_mod.`-prefixed keys, which the float→quantization weight-transfer path (`load_weights.py`) cannot match. It falls back to `strict=False` and silently discards the entire float-trained result, restarting quantization-aware training from random initialization — with no exception and no fatal warning, so the pipeline reports success while producing a materially worse model.

**Architecture:** `setup_distributed_model` sets `model_without_ddp = model` when not using DDP (the common case), so when `compile_model_if_enabled` succeeded upstream, `model_without_ddp` is the `OptimizedModule` wrapper itself. `save_checkpoint()` then calls `.state_dict()` directly on it, emitting `_orig_mod.`-prefixed keys. Fix: unwrap via `getattr(model_without_ddp, '_orig_mod', model_without_ddp)` before calling `.state_dict()`, and apply the same unwrap to `model_ema`. For symmetry, `resume_from_checkpoint()` (used by `--resume`, a separate code path from the float→quant `--weights` transfer) gets the same unwrap before `.load_state_dict()`, so a resumed run against a compiled model doesn't hit the same key mismatch in the other direction.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, pytest

## Global Constraints

- Python `==3.10.*`
- No new dependencies
- Zero behavior change for uncompiled models (the common case today) — the unwrap is a no-op via `getattr`'s fallback
- Do NOT use the existing `unwrap_compiled_submodules()` helper (from `tinyml_tinyverse/common/utils/utils.py`) for this fix — it mutates the model's submodule tree in place via `setattr`, which would silently un-compile the live training model. This fix must only affect what gets written to / read from the checkpoint dict, never the live model object still being trained.
- Save and load must be symmetric: whatever key convention `save_checkpoint` writes, `resume_from_checkpoint` must read using the same convention

## Context: Why This Is Needed

Found and reproduced by a final whole-branch code reviewer, via direct execution of the real production pipeline: a `compile_model=1` run (float phase compiles successfully) saves `checkpoint.pth` with `_orig_mod.`-prefixed keys. The pipeline then starts the quantization phase (which the separate `skip-compile-under-quantization` fix correctly leaves uncompiled) and loads that checkpoint via `--weights` through `tinyml-tinyverse/tinyml_tinyverse/common/utils/load_weights.py`. That loader only knows how to align a `module.` (DDP) prefix — never `_orig_mod.` (torch.compile) — so `load_state_dict(data, strict=True)` fails, falls into its fallback path, and `check_model_data` reports every single weight as "missing." The subsequent `load_state_dict(data, strict=False)` call then loads nothing useful. No exception is raised; a yellow warning is printed (`=> The following layers in the model could not be loaded from pre-trained:`) but training continues and the pipeline returns success.

Reproduced end-to-end:
```
loading pretrained checkpoint for training: .../training/base/checkpoint.pth
=> The following layers in the model could not be loaded from pre-trained:   (24 tensors)
=> The following weights in pre-trained were not used:   _orig_mod.features.0.weight ... (24 tensors)
PIPELINE_RETURNED: True
CKPT .../training/base/checkpoint.pth          keys=28  _orig_mod_prefixed=28
CKPT .../training/quantization/checkpoint.pth  keys=28  _orig_mod_prefixed=0
```

This is latent in `upstream/main` already (which has `compile_model_if_enabled` from an earlier merged PR, with `compile_model` defaulting to 0), but the `hardware-defaults` feature in this branch converts `compile_model` from opt-in to default-on for every CUDA host — turning a dormant bug into a default-path regression. This is more severe than the crash fixed separately (`skip-compile-under-quantization`): a crash is visible immediately; this silently degrades model quality with no error signal.

## File Map

| Action | Path | Responsibility |
|--------|------|-----------------|
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py` | `save_checkpoint` and `resume_from_checkpoint` unwrap before state_dict save/load |
| Create | `tinyml-tinyverse/tests/test_checkpoint_unwrap_compiled_model.py` | Regression test: compile → save → assert clean keys → round-trip load succeeds |

---

## Task 1: Unwrap fix + tests

**Files:**
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py` (`save_checkpoint` at line 798, `resume_from_checkpoint` at line 592 — confirm exact current line numbers before editing, they may have drifted)
- Create: `tinyml-tinyverse/tests/test_checkpoint_unwrap_compiled_model.py`

**Interfaces:**
- No signature change to either function — the fix is internal

---

- [ ] **Step 1: Read the current implementations**

```bash
sed -n '592,615p;798,826p' tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py
```

Confirm `save_checkpoint` does `'model': model_without_ddp.state_dict()` and, if `model_ema` is truthy, `checkpoint['model_ema'] = model_ema.state_dict()`. Confirm `resume_from_checkpoint` does `model_without_ddp.load_state_dict(checkpoint['model'])` and, if `model_ema`, `model_ema.load_state_dict(checkpoint['model_ema'])`.

- [ ] **Step 2: Write the failing test**

Create `tinyml-tinyverse/tests/test_checkpoint_unwrap_compiled_model.py`:

```python
"""Regression test for: checkpoints saved from a torch.compile-wrapped model
carrying _orig_mod.-prefixed keys, which the float->quantization weight
transfer path (load_weights.py) cannot match -- it falls back to
strict=False and silently discards the entire float-trained result. No
exception is raised; the pipeline reports success while quietly retraining
from random init.

Root cause: setup_distributed_model sets model_without_ddp = model when not
using DDP, so when compile_model_if_enabled succeeded upstream,
model_without_ddp IS the torch._dynamo.OptimizedModule wrapper. state_dict()
on it emits every key prefixed _orig_mod.
"""
import torch
import torch.nn as nn

from tinyml_tinyverse.references.common.train_base import save_checkpoint, resume_from_checkpoint


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class _FakeOptimizer:
    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass


class _FakeScheduler:
    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass


class _FakeArgs:
    def __init__(self, resume):
        self.resume = resume


def test_save_checkpoint_strips_orig_mod_prefix_from_compiled_model():
    model = _TinyModel()
    compiled_model = torch.compile(model, backend='aot_eager')
    compiled_model(torch.rand(1, 4))  # trigger real compilation

    checkpoint = save_checkpoint(
        compiled_model, _FakeOptimizer(), _FakeScheduler(), epoch=0, args=_FakeArgs(resume=None),
    )
    keys = list(checkpoint['model'].keys())
    assert keys, "checkpoint has no keys at all"
    assert not any(k.startswith('_orig_mod.') for k in keys), keys


def test_save_checkpoint_uncompiled_model_unaffected():
    """Backward compatibility: an ordinary, uncompiled model's checkpoint
    keys are unchanged (no _orig_mod. prefix ever existed to strip)."""
    model = _TinyModel()
    checkpoint = save_checkpoint(
        model, _FakeOptimizer(), _FakeScheduler(), epoch=0, args=_FakeArgs(resume=None),
    )
    assert set(checkpoint['model'].keys()) == set(model.state_dict().keys())


def test_checkpoint_round_trips_into_a_fresh_uncompiled_model():
    """The actual failure mode: save from a compiled model, load into the
    (uncompiled) model used for the next training phase, and confirm the
    real trained weights -- not random-init defaults -- are what land."""
    source = _TinyModel()
    with torch.no_grad():
        source.linear.weight.fill_(3.14)
    compiled_source = torch.compile(source, backend='aot_eager')
    compiled_source(torch.rand(1, 4))

    checkpoint = save_checkpoint(
        compiled_source, _FakeOptimizer(), _FakeScheduler(), epoch=0, args=_FakeArgs(resume=None),
    )

    target = _TinyModel()  # fresh, randomly initialized, NOT compiled
    assert not torch.allclose(target.linear.weight, torch.full_like(target.linear.weight, 3.14))
    target.load_state_dict(checkpoint['model'], strict=True)  # must not need strict=False
    assert torch.allclose(target.linear.weight, torch.full_like(target.linear.weight, 3.14))


def test_resume_from_checkpoint_symmetric_with_compiled_model():
    """resume_from_checkpoint (the --resume path) must be able to load a
    checkpoint saved by save_checkpoint back into a still-compiled model,
    using the same unwrap on both sides."""
    import tempfile
    import os

    source = _TinyModel()
    with torch.no_grad():
        source.linear.weight.fill_(2.71)
    compiled_source = torch.compile(source, backend='aot_eager')
    compiled_source(torch.rand(1, 4))

    checkpoint = save_checkpoint(
        compiled_source, _FakeOptimizer(), _FakeScheduler(), epoch=5, args=_FakeArgs(resume=None),
    )

    fresh = _TinyModel()
    compiled_fresh = torch.compile(fresh, backend='aot_eager')
    compiled_fresh(torch.rand(1, 4))

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'checkpoint.pth')
        torch.save(checkpoint, ckpt_path)
        args = _FakeArgs(resume=ckpt_path)
        args.device = 'cpu'
        resume_from_checkpoint(compiled_fresh, _FakeOptimizer(), _FakeScheduler(), None, args)

    assert torch.allclose(fresh.linear.weight, torch.full_like(fresh.linear.weight, 2.71))
```

- [ ] **Step 3: Run the tests to confirm they fail for the expected reason**

```bash
cd tinyml-tinyverse
python -m pytest tests/test_checkpoint_unwrap_compiled_model.py -v 2>&1 | tail -40
```

Expected: `test_save_checkpoint_strips_orig_mod_prefix_from_compiled_model` and `test_checkpoint_round_trips_into_a_fresh_uncompiled_model` FAIL (checkpoint keys still carry `_orig_mod.`, and `strict=True` load raises). `test_resume_from_checkpoint_symmetric_with_compiled_model` should also fail (key mismatch on the load side). `test_save_checkpoint_uncompiled_model_unaffected` should already PASS.

- [ ] **Step 4: Implement the fix**

In `tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py`, in `save_checkpoint`, change:
```python
    checkpoint = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args
    }
    if model_ema:
        checkpoint['model_ema'] = model_ema.state_dict()
```
to:
```python
    # torch.compile() wraps a model in torch._dynamo.OptimizedModule; when
    # not using DDP, model_without_ddp IS that wrapper (setup_distributed_model
    # only assigns model_without_ddp = model.module under DDP). state_dict() on
    # a compiled model emits every key prefixed _orig_mod., which downstream
    # weight-loading (load_weights.py, used for the float->quantization
    # transfer) cannot match -- it silently falls back to strict=False and
    # discards the entire result. Unwrap before saving so checkpoints always
    # carry the original, uncompiled key names.
    checkpoint_model = getattr(model_without_ddp, '_orig_mod', model_without_ddp)
    checkpoint = {
        'model': checkpoint_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args
    }
    if model_ema:
        checkpoint_ema = getattr(model_ema, '_orig_mod', model_ema)
        checkpoint['model_ema'] = checkpoint_ema.state_dict()
```

In the same file, in `resume_from_checkpoint`, change:
```python
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint['model_ema'])
    return args
```
to:
```python
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        # Symmetric with save_checkpoint's unwrap: checkpoints always carry
        # uncompiled key names, so load into the unwrapped model regardless
        # of whether it's currently wrapped by torch.compile.
        resume_model = getattr(model_without_ddp, '_orig_mod', model_without_ddp)
        resume_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if model_ema:
            resume_ema = getattr(model_ema, '_orig_mod', model_ema)
            resume_ema.load_state_dict(checkpoint['model_ema'])
    return args
```

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
cd tinyml-tinyverse
python -m pytest tests/test_checkpoint_unwrap_compiled_model.py -v
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

Expected: no new failures. Only the 5 known `F28E12` device-profile failures should remain in modelmaker.

- [ ] **Step 7: Commit**

```bash
git add tinyml-tinyverse/tinyml_tinyverse/references/common/train_base.py \
        tinyml-tinyverse/tests/test_checkpoint_unwrap_compiled_model.py
git commit -m "fix: strip torch.compile wrapper prefix from saved checkpoints"
```

---

## Task 2: Real-hardware verification — the exact failure scenario

**Files:** none (verification only)

**Why this task matters:** this bug was found via a genuine end-to-end pipeline run, not a unit test in isolation. The verification must reproduce that exact scenario: float training with `compile_model=1` succeeding, transitioning into quantization, and confirming the float-trained weights (not random init) actually land in the quantized model.

---

- [ ] **Step 1: Sync the branch to the GX10**

```bash
git push origin HEAD:integration
ssh martin@<gx10-host> "cd ~/repos/tinyml-tensorlab && git pull"
```

- [ ] **Step 2: Re-run a full compile+quantization pipeline and inspect the checkpoints directly**

Using the same style of ad-hoc validation script as the `skip-compile-under-quantization` plan's Task 2 (a config with `num_gpus: 0` to force CPU/`aot_eager` compilation, which succeeds independent of any GPU-specific compile issues, `compile_model: 1`, `quantization: 2`), after the run completes, inspect the saved checkpoints directly:

```python
import torch
ckpt = torch.load('<project_dir>/training/base/checkpoint.pth', weights_only=False)
prefixed = [k for k in ckpt['model'].keys() if k.startswith('_orig_mod.')]
print(f"keys={len(ckpt['model'])} _orig_mod_prefixed={len(prefixed)}")
```

Expected (post-fix): `_orig_mod_prefixed=0`.

- [ ] **Step 3: Confirm the float weights actually transfer into the quantized model**

Check the training log for the earlier failure signature and confirm it's absent:

```bash
grep -i "could not be loaded from pre-trained" <run_dir>/*.log
```

Expected: no output (or only the pre-existing, unrelated `num_batches_tracked` note, if any — not a full-model mismatch).

- [ ] **Step 4: Re-run the full modelmaker test suite**

```bash
ssh martin@<gx10-host> "cd ~/repos/tinyml-tensorlab/tinyml-modelmaker && ~/repos/tinyml-tensorlab/.venv/bin/python -m pytest tests/ -v --tb=short --ignore=tests/test_protocols.py 2>&1 | tail -20"
```

Expected: only the 5 known `F28E12` failures remain.

- [ ] **Step 5: Report findings**

No code changes in this task — report the before/after comparison (checkpoint key prefix, presence/absence of the "could not be loaded" warning) as the closing verification for this plan.

---

## Self-Review Checklist

- **Spec coverage:**
  - ✅ `save_checkpoint` unwraps both the main model and EMA model before `.state_dict()`
  - ✅ `resume_from_checkpoint` unwraps symmetrically before `.load_state_dict()`
  - ✅ Explicitly does NOT reuse `unwrap_compiled_submodules()` (documented reason: it mutates in place, which would be wrong here)
  - ✅ Zero behavior change for uncompiled models (no-op via `getattr` fallback)
  - ✅ Real-hardware verification targets the exact reproduced scenario (compile succeeds, float→quant weight transfer), not a synthetic proxy
- **Placeholder scan:** None found
- **Type consistency:** N/A (no new function signatures)
