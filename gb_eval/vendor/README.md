# vendor

Files in this directory tree that were not written inside the workspace project are listed here with their origin.

| file (this repo) | original path (private lab storage, not available) | sha256 | last modified |
|---|---|---|---|
| `gb_eval/vendor/bin/runner_guards.py` | `<ORIGINAL_ROOT>/bin/runner_guards.py` | b701f1e7ed5f627958ba6fddcd43d35a444ed000d33838f5e8d83c89f796cbfe | 2026-10-02 |
| `gb_eval/vendor/dft_input_tools/make_vasp_inputs.py` | `<ORIGINAL_ROOT>/workspace/runs/<earlier internal run>/scripts/make_vasp_inputs.py` (DFT input helper of an earlier internal run; the run name is not kept in this repository) | 3660c293c5d0400677829e8df014c151732cecd1b0dc2c05d4036e04aa06b633 | 2026-09-15 |

The origin is a plain directory of lab utilities that is not a git repository, so no commit hash exists; the sha256 above identifies the content.
Lab-internal tools that the original sbatch scripts call (job pre-flight checks, the job scheduler registry) are NOT included; the lines that call them are kept as they were and are not needed to run the scripts documented in docs/reproduce.md. The calls of a model-specification check were removed from the sbatch generation scripts (see NOTICE). `make_vasp_inputs.py` comes from inside the analysis project (its commit is not recorded; the sha256 identifies the content).
