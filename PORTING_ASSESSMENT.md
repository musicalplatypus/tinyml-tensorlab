# Porting Assessment: platypus_dev → platypus_dev_1.3

**Date:** 2026-02-26
**Base branch:** `platypus_dev_1.3` (from `origin/main` at `6bda92d`)
**Source branch:** `platypus_dev` (14 commits ahead of the old `origin/main`)

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

**Portability: EASY**

New standalone file, no conflicts possible. Cherry-pick directly.

```
git cherry-pick dcf7ee9
```

---

### 2. `73fa781` — Fix @classmethod methods using self instead of cls

**Portability: VERIFY FIRST**

Touches 9 files across modelmaker (compilation, datasets, runner, training files).
Upstream refactored many of these files heavily. The `self → cls` fixes may already
be resolved upstream, or the affected `@classmethod` methods may no longer exist.

**Action:** Check each of the 9 files in `origin/main` to see if the bug still
exists. If it does, re-apply manually. If upstream fixed it, drop this commit.

Files to check:
- `ai_modules/common/compilation/tinyml_benchmark.py`
- `ai_modules/common/datasets/__init__.py`
- `ai_modules/timeseries/runner.py`
- `ai_modules/timeseries/training/tinyml_tinyverse/timeseries_*.py` (4 files)
- `ai_modules/vision/runner.py`
- `ai_modules/vision/training/tinyml_tinyverse/image_classification.py`

---

### 3. `75bea31` — Replace magic strings with named constants

**Portability: CONFLICT-HEAVY — Re-implement**

Touches 13 files including `constants.py`, `params.py`, `training/__init__.py`
for both timeseries and vision. Upstream modified many of these same files with
different changes. Cherry-pick will produce extensive conflicts.

**Action:** Re-implement against the new upstream codebase. Audit `origin/main`
for remaining magic strings and apply the same pattern (extracting string literals
into named constants in `constants.py`).

---

### 4. `2b2ba94` — Add pytest test suite for tinyml-modelmaker

**Portability: MOSTLY PORTABLE**

Adds 7 new test files + pyproject.toml changes. The new files won't conflict,
but tests may reference APIs from other platypus_dev refactors (e.g.
`resolve_paths()`, `BaseModelTraining`) that don't exist in the upstream code.

**Action:** Cherry-pick, then review each test file. Fix imports and assertions
that reference platypus_dev-specific refactors. Key files:
- `tests/conftest.py` — may import our `resolve_paths`
- `tests/test_misc_utils.py` — may test `resolve_paths()`
- `tests/test_base_training.py` — references our `base_training.py` (superseded)

---

### 5. `9ba1339` — Replace print() with logging module

**Portability: VERIFY FIRST**

Upstream's major refactor likely addressed some or all `print()` usage.

**Action:** Search `origin/main` for remaining `print()` calls in modelmaker
source (excluding tests). If upstream already uses logging throughout, drop this
commit. If `print()` calls remain, re-apply the logging pattern to those files.

```
grep -rn "^\s*print(" origin/main:tinyml-modelmaker/tinyml_modelmaker/ --include="*.py"
```

---

### 6. `21e2fca` — Replace assert/sys.exit/raise-string with proper exceptions

**Portability: CONFLICT-HEAVY — Re-implement**

Touches 11 files. Upstream modified many of the same files. The specific
`assert` statements and `sys.exit()` calls may have been restructured.

**Action:** Audit `origin/main` for remaining `assert` (non-test), `sys.exit()`,
and bare `raise "string"` patterns. Apply proper exception handling to any that
remain.

---

### 7. `12bdf2c` — Extract resolve_paths() into misc_utils.py

**Portability: RE-IMPLEMENT**

Extracted ~127 lines of path resolution from `timeseries/runner.py` and
`vision/runner.py` into a shared `resolve_paths()` in `misc_utils.py`.

Upstream still has path resolution inline in both runners, but the code has
changed significantly (426 lines in runner.py now vs the version we refactored).
The `train_output_path` handling and project path computation logic differs.

**Action:** Re-implement the extraction against upstream's current `runner.py`.
Compare the two runners to identify the common path-resolution logic and extract
it into `misc_utils.py`. This is important for mmcli's `builder.py` which relies
on `train_output_path` being handled correctly.

**Note:** This is a high-priority port — the mmcli CLI depends on the path
resolution behavior, particularly `train_output_path` for separating working
copies from original data.

---

### 8. `98095af` — Add Protocol definitions for component interfaces

**Portability: EASY**

Adds 2 new files (`protocols.py`, `test_protocols.py`) and a 1-line import in
`__init__.py`. The protocols define interfaces (ModelRunner, Trainer, etc.) that
document the expected API shape.

**Action:** Cherry-pick directly. May need minor adjustment to the `__init__.py`
import if upstream changed that file.

---

### 9. `0406cc8` — Extract BaseModelTraining to deduplicate training files

**Portability: SUPERSEDED — Drop**

Created `base_training.py` (419 lines) and reduced the 4 training subclass files
by ~1,200 lines total.

Upstream independently created `timeseries_base.py` (925 lines) with the same
intent. The upstream version is more comprehensive and covers additional
functionality. Our version is incompatible with the new upstream structure.

**Action:** Drop this commit entirely. Use upstream's `timeseries_base.py`
instead. Verify that any unique improvements from our `base_training.py` that
aren't in upstream's version are noted and can be proposed separately.

---

### 10. `1634334` — Remove unused TVM imports from test_onnx files

**Portability: VERIFY FIRST**

Small change (3 files, 3 lines each) — guards TVM imports with try/except to
allow training without TVM installed.

**Action:** Check if the 3 test_onnx files in `origin/main` still have the
bare TVM imports. If so, cherry-pick (may apply cleanly). If upstream already
fixed this, drop.

Files:
- `tinyml_tinyverse/references/image_classification/test_onnx.py`
- `tinyml_tinyverse/references/timeseries_anomalydetection/test_onnx_cls.py`
- `tinyml_tinyverse/references/timeseries_classification/test_onnx.py`

---

### 11. `e1b18df` — Update repo URLs from TexasInstruments to musicalplatypus

**Portability: RE-APPLY**

URL replacements across 15 files. The file contents have changed upstream, so
the old diff won't apply cleanly, but the intent is simple: replace all
`github.com/TexasInstruments/tinyml-tensorlab` references with the fork URL.

**Action:** Run a project-wide search-and-replace on `platypus_dev_1.3`:
```
grep -rn "TexasInstruments/tinyml-tensorlab" --include="*.py" --include="*.toml" \
  --include="*.json" --include="*.yaml" --include="*.md" --include="*.sh" --include="*.ps1"
```
Then replace with `musicalplatypus/tinyml-tensorlab`.

---

### 12. `6236a3e` — Update git branch references from r1.2/main to platypus_dev

**Portability: RE-APPLY (adapt)**

Changed branch references in pyproject.toml, README, setup scripts. For the
new branch, references should point to `platypus_dev_1.3` instead.

**Action:** Search for `r1.3/main` or `main` branch references in install URLs
and replace with `platypus_dev_1.3`. Files to check:
- `README.md`
- `tinyml-modelmaker/pyproject.toml`
- `tinyml-modelmaker/setup_all.sh` / `setup_all.ps1`
- `tinyml-modeloptimization/README.md`
- `tinyml-tinyverse/pyproject.toml`

---

### 13. `bc31bd6` — Bump sub-package versions to 1.2.0.dev0

**Portability: SUPERSEDED — New version needed**

Upstream is now at 1.3.0. Need a fresh bump to `1.3.0.dev0`.

**Action:** Create a new commit that bumps all version declarations to
`1.3.0.dev0` in:
- `tinyml-modelmaker/pyproject.toml`
- `tinyml-modelmaker/tinyml_modelmaker/version.py`
- `tinyml-tinyverse/pyproject.toml`
- `tinyml-tinyverse/tinyml_tinyverse/references/version.py`
- `tinyml-modeloptimization/torchmodelopt/pyproject.toml`
- `tinyml-modeloptimization/torchmodelopt/version.py`

---

### 14. `9f5769d` — Fix MPS compatibility: cast dtype before moving tensors to device

**Portability: ALREADY APPLIED**

Cherry-picked as `b3c606f` on `platypus_dev_1.3` (with one conflict resolved).

---

## Recommended Porting Order

### Phase 1: Quick wins (no conflicts)
1. `dcf7ee9` — ARCHITECTURE.md (cherry-pick)
2. `98095af` — Protocol definitions (cherry-pick)
3. New commit: version bump to `1.3.0.dev0`
4. New commit: URL replacement (TexasInstruments → musicalplatypus)
5. New commit: branch references → `platypus_dev_1.3`

### Phase 2: Verify and apply
6. `1634334` — TVM imports (verify if still needed)
7. `73fa781` — @classmethod fix (verify if still needed)
8. `9ba1339` — print→logging (verify if still needed)

### Phase 3: Re-implement against new base
9. `12bdf2c` — resolve_paths() extraction (high priority for mmcli)
10. `75bea31` — Named constants (audit new codebase)
11. `21e2fca` — Proper exceptions (audit new codebase)
12. `2b2ba94` — Test suite (adapt to new API shapes)

### Drop
- `0406cc8` — BaseModelTraining (superseded by upstream's `timeseries_base.py`)
- `bc31bd6` — Version bump (superseded; need 1.3.0.dev0 instead)

## Risk Assessment

**High overlap areas** (most likely to cause merge conflicts):
- `tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/runner.py`
- `tinyml-modelmaker/tinyml_modelmaker/utils/misc_utils.py`
- `tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/training/tinyml_tinyverse/*.py`

**Safe areas** (new files, minimal overlap):
- Test files (`tests/`)
- Protocol definitions (`protocols.py`)
- Documentation (`ARCHITECTURE.md`)
