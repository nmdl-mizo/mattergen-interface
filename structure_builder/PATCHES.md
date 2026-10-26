# Differences from the original scripts

Each file was copied from the original run and changed only where listed. The changes are paths, removal of lab-internal references and the removal of
two mount-layout checks; no computation was changed.

* Roots of inputs and outputs: the absolute directories of the original run were replaced by `SB_WORK` (`lib/builder_common.py`, stage 0/1 scripts, `build_tilt_baseline_csv.py`).
  `GROUPS_CSV` and `STRUCT_G10` point to the outputs of stages 0 and 1 below `SB_WORK`.
* `sys.path` additions use paths relative to the script (`structure_builder/lib`, `stageD_gamma_dedup`, and the repository root for `stage_e_manifests.py`).
* Removed the mount-prefix layout checks: the `ValueError` for directories that do not start with the lab's mount name in `stageD_gamma_dedup/stage_d1_build_csv.py` and in `checks/check_data_rebuild.py`;
  `resolve_view()` of `stageE_split/build_tilt_baseline_csv.py` and `resolve()` / `to_remote()` of `lib/config.py` no longer translate mount names.
* `lib/config.py`: the absolute path constants were removed; the MACE model path is read from `MACE_MODEL_PATH` by `mace_model_path()` when stage C starts (not at import time).
* `stage0_templates/cnid_grid_sweep.py`: only `--build-only` (the construction of the 10 x 10 grid) is supported; the comparison part of the original script needed structures of
  earlier internal runs. `GRIDS = [10]`.
* `stage1_template_dedup/unified_chunk.py`, `unified_aggregate.py`: the comparison kinds against the ten structures of the earlier 4 x 4 grid (kinds `x`, `e`) were removed; the grouping of the
  100 structures (kind `p`: 4,950 pairs, union-find) is unchanged. The columns `matches_existing`, `prev_M-all_group_id`, `prev_M-bulk_group_id` of `groups_10x10_unified.csv` are empty.
* `lib/dx_lib.py`: `pilot_poscar_path()` (not used here) now raises. `lib/runner_guards.py`: the dashboard directories were removed.
* Names that follow the renaming of the model package (`SCM_HOST_Z`, `AL_S5_210_TRAIN`, `symmetry_constrained`) in `stageE_split/` and `checks/`.

Not included: the learning-side configuration writer and precheck (`stage_e_write_configs.py`, `stage_e_precheck.py`, `stage_e_stats.py`), their tests (`test_stage_e_units.py`,
`check_e_configs.py`), the check scripts that read lab-internal directories or the training environment (`check_e_manifests.py`, `dataset_regression.py`, `make_validationes.py`), the pilot-run and
relabel-run scripts of the earlier internal runs, and the earlier (non-final) grid-sweep scripts.
