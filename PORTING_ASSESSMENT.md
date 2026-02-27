# Porting Assessment: platypus_dev → platypus_dev_1.3

**Date:** 2026-02-26
**Base branch:** `platypus_dev_1.3` (from `origin/main` at `6bda92d`)
**Source branch:** `platypus_dev` (14 commits ahead of the old `origin/main`)
**Status:** ✅ **COMPLETE** — all phases executed, branch pushed to remote

## Upstream Changes Summary

Since `platypus_dev` diverged, `origin/main` received substantial updates
(version 1.2.0 → 1.3.0):

| Sub-package          | Files changed | Lines added | Lines removed |
|----------------------|---------------|-------------|---------------|
| tinyml-modelmaker    | 157           | +33,738     | -11,468       |
| tinyml-tinyverse     | 44            | +4,162      | -5,547        |
| tinyml-modeloptimization | 31        | +1,426      | -1,854        |

Key upstream changes:
- Version bumped to **1.3.0** across all sub-packages
- Created `timeseries_base.py` (925 lines) — their own `BaseModelTraining` extraction
- Dramatically reduced training subclass files (e.g. classification 426 → 106 lines)
- Path resolution remains inline in `runner.py` (not extracted)
- Updated model descriptions, added new models/presets
- Release cleanup (`7316768 updated for release`)

## platypus_dev Commits (oldest → newest)

### 1. `dcf7ee9` — Add ARCHITECTURE.md

✅ **DONE** — Cherry-picked as `b448116`

---

### 2. `73fa781` — Fix @classmethod methods using self instead of cls

✅ **DONE** — Re-applied to 5 remaining files as `e690110`

Upstream fixed 4 of the original 9 files. Applied the fix to:
- `ai_modules/common/compilation/tinyml_benchmark.py`
- `ai_modules/common/datasets/__init__.py`
- `ai_modules/timeseries/runner.py`
- `ai_modules/vision/runner.py`
- `ai_modules/vision/training/tinyml_tinyverse/image_classification.py`

---

### 3. `75bea31` — Replace magic strings with named constants

✅ **DONE** — Re-implemented as `868ffb9`

Added to both timeseries and vision `constants.py`:
- `TRAINING_BACKEND_TINYML_TINYVERSE`
- `DATA_DIR_CLASSES`, `DATA_DIR_FILES`, `DATA_DIR_IMAGES`
- Used existing `TRAINING_DEVICE_CUDA` in `params.py`

---

### 4. `2b2ba94` — Add pytest test suite for tinyml-modelmaker

✅ **DONE** — Re-implemented as `5f40eaa`

Created 5 test files adapted to the new 1.3.0 API:
- `test_config_dict.py` (6 tests)
- `test_constants.py` (16 tests)
- `test_dataset_utils.py` (6 tests)
- `test_misc_utils.py` (10 tests)
- `test_protocols.py` (updated with `importorskip("tvm")`)

Results: **47 passed, 1 skipped** (test_protocols requires TVM)

---

### 5. `9ba1339` — Replace print() with logging module

✅ **DONE** — Re-applied across 8 files as `ee684f8`

Files updated:
- `timeseries/runner.py`, `vision/runner.py`
- `timeseries/training/__init__.py`, `vision/training/__init__.py`
- `utils/misc_utils.py`, `utils/download_utils.py`
- `run_tinyml_modelmaker.py`
- `common/datasets/__init__.py`

---

### 6. `21e2fca` — Replace assert/sys.exit/raise-string with proper exceptions

✅ **DONE** — Re-implemented as `608c2d8`

Replaced ~25 `assert` statements, 1 `raise "string"`, and `assert False`
across 7 files with proper `ValueError`, `TypeError`, `FileNotFoundError`,
and `RuntimeError` exceptions.

---

### 7. `12bdf2c` — Extract resolve_paths() into misc_utils.py

✅ **DONE** — Re-implemented as `23117c1`

Extracted `resolve_paths()` and `resolve_run_name()` into `misc_utils.py`.
Updated both `timeseries/runner.py` and `vision/runner.py` to call the
shared function. Critical for mmcli's `builder.py` `train_output_path` support.

---

### 8. `98095af` — Add Protocol definitions for component interfaces

✅ **DONE** — Cherry-picked as `b0a7fd7`

---

### 9. `0406cc8` — Extract BaseModelTraining to deduplicate training files

❌ **DROPPED** — Superseded by upstream's `timeseries_base.py` (925 lines)

Our 419-line `base_training.py` is incompatible with the new upstream structure.
Upstream's version is more comprehensive.

---

### 10. `1634334` — Remove unused TVM imports from test_onnx files

❌ **DROPPED** — Upstream already fixed the TVM imports in 1.3.0

---

### 11. `e1b18df` — Update repo URLs from TexasInstruments to musicalplatypus

✅ **DONE** — Re-applied as `613dc01`

Replaced across 19 files.

---

### 12. `6236a3e` — Update git branch references from r1.2/main to platypus_dev

✅ **DONE** — Re-applied as `4184224`

Updated references to `platypus_dev_1.3` in 4 files.

---

### 13. `bc31bd6` — Bump sub-package versions to 1.2.0.dev0

✅ **DONE (superseded)** — New version bump to `1.3.0.dev0` as `5ff1ac9`

Bumped all 6 version files (3 pyproject.toml + 3 version.py).

---

### 14. `9f5769d` — Fix MPS compatibility: cast dtype before moving tensors to device

✅ **DONE** — Cherry-picked as `b3c606f` (one conflict resolved)

---

## Execution Summary

### Phase 1: Quick wins ✅
| Commit | Description | Result |
|--------|-------------|--------|
| `b3c606f` | MPS fix | Cherry-picked from platypus_dev |
| `b448116` | ARCHITECTURE.md | Cherry-picked |
| `b0a7fd7` | Protocol definitions | Cherry-picked |
| `5ff1ac9` | Version bump 1.3.0.dev0 | New commit |
| `613dc01` | URL replacement | New commit |
| `4184224` | Branch references | New commit |

### Phase 2: Verify and apply ✅
| Commit | Description | Result |
|--------|-------------|--------|
| — | TVM imports | Dropped (fixed upstream) |
| `e690110` | @classmethod fix | Re-applied to 5 files |
| `ee684f8` | print→logging | Re-applied across 8 files |

### Phase 3: Re-implement against new base ✅
| Commit | Description | Result |
|--------|-------------|--------|
| `23117c1` | resolve_paths() extraction | Re-implemented |
| `868ffb9` | Named constants | Re-implemented |
| `608c2d8` | Proper exceptions | Re-implemented |
| `5f40eaa` | Test suite | Re-implemented (47 pass, 1 skip) |

### Dropped
| Original | Description | Reason |
|----------|-------------|--------|
| `0406cc8` | BaseModelTraining | Superseded by upstream `timeseries_base.py` |
| `bc31bd6` | Version bump 1.2.0.dev0 | Superseded; bumped to 1.3.0.dev0 instead |
| `1634334` | TVM imports | Fixed upstream |

## Post-Porting: New Development

After the porting phase, additional work was done on the `platypus_dev_1.3` branch:

### 15. `f9a6c91` — Fix MPS compatibility: move torcheval metrics to CPU

✅ **DONE** — New commit

Moved `torcheval` metric computations (MulticlassAUROC) to CPU before computation,
fixing a crash on MPS where torcheval operators are not supported.

---

### 16. `36bd426` — Optimize training performance

✅ **DONE** — New commit

Training engine optimizations in tinyml-tinyverse:
- **O(n) eval**: Replaced O(n^2) `torch.cat` accumulation with list + single cat at epoch end (~2x eval speedup)
- **Epoch-end metrics**: F1 score and confusion matrix computed once at epoch end
- **`set_to_none=True`**: All `optimizer.zero_grad()` calls (~17% micro-benchmark improvement)
- **Persistent workers**: `persistent_workers=True` in DataLoaders
- **Fixed pin_memory**: Only enables when CUDA is available (was always False on MPS)
- **torch.compile support**: Opt-in via `--compile-model 1` (inductor on CUDA, aot_eager on MPS)
- **Native AMP support**: Opt-in via `--native-amp` (autocast + GradScaler on CUDA)

Benchmark results (MPS, Apple M3 Max):
- torch.cat fix: **+97.7%** improvement
- set_to_none: **+17%** improvement
- torch.compile: **-26%** (harmful on MPS, CUDA-only)
- native AMP: **-29%** (harmful on MPS, CUDA-only)

---

### 17. Thread performance flags through modelmaker config pipeline

✅ **DONE** — Committed with documentation update

- Added `compile_model` and `native_amp` defaults to `params.py`
- Updated `timeseries_base.py` to pass `--compile-model` and `--native-amp` to tinyml-tinyverse argv
- Updated ARCHITECTURE.md with Training Performance Optimizations section

---

## Branch Status

`platypus_dev_1.3`: **16 commits** ahead of `origin/main`.

```
(new)  Thread performance flags through modelmaker pipeline + update docs
36bd426 Optimize training performance: torch.compile, AMP, persistent workers, eval efficiency
f9a6c91 Fix MPS compatibility: move torcheval metrics to CPU before computation
8488b6f Update PORTING_ASSESSMENT.md: mark all phases complete
5f40eaa Add pytest test suite for tinyml-modelmaker core modules
608c2d8 Replace assert/sys.exit/raise-string with proper exceptions across tinyml-modelmaker
868ffb9 Replace magic strings with named constants
23117c1 Extract duplicated path resolution from ModelRunner into shared resolve_paths()
ee684f8 Replace print() with logging module across tinyml-modelmaker
e690110 Fix @classmethod methods using self instead of cls
4184224 Update git branch references to platypus_dev_1.3
613dc01 Update repo URLs from TexasInstruments to musicalplatypus fork
5ff1ac9 Bump sub-package versions to 1.3.0.dev0 for development fork
b0a7fd7 Add Protocol definitions for component interfaces
b448116 Add ARCHITECTURE.md
b3c606f Fix MPS compatibility: cast dtype before moving tensors to device
6bda92d (origin/main) ... upstream base
```
