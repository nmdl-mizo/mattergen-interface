#!/usr/bin/env python3
"""Data rebuild, stages E-5 and E-6: generates the set of settings of the proposed model and the baseline from the results of stage E-3.

  <python> stage_e_write_configs.py --out-dir DIR [--results-dir DIR] [--bulk-membership CSV] [--force]
  <python> stage_e_write_configs.py --production [...]      # output destination of production (MN/mattergen/conf and RUN/scripts). Not used this time

With --out-dir DIR, it writes under DIR with the same structure as MN (for tests. It can be resolved by adding <DIR>/mattergen/conf to hydra.searchpath):
  <DIR>/mattergen/conf/data_module/scm_al_proposed.yaml
  <DIR>/mattergen/conf/lightning_module/diffusion_module/corruption/scm_al_proposed.yaml
  <DIR>/mattergen/conf/data_module/scm_al_baseline.yaml
  <DIR>/scripts/sbatch_train_al_proposed.sh              training of the proposed model
  <DIR>/scripts/sbatch_build_baseline_csv.sh          creation of the CSV of the baseline (a call of build_tilt_baseline_csv.py)
  <DIR>/scripts/sbatch_prep_cache_baseline.sh         caching of the baseline (csv_to_dataset. The old cache is not touched)
  <DIR>/scripts/sbatch_train_al_baseline.sh           training of the baseline
  <DIR>/stage_e_write_configs_manifest.json               list of the products, input values and sha256
With --production, the conf is written directly under MN/mattergen/conf and the sbatch files to RUN/scripts.

Policy:
  - The yaml files of the proposed model and the baseline are made by replacing the text of the current yaml files (under MN/mattergen/conf; read only) line by line (an exception unless the replacement is exactly 1 time (or the specified number of times)).
    The leading comment (description of the old data) is replaced by a new comment.
  - The average density is the value of the model specification 0.05839622542107227 (stage_e_common.CARD_AVERAGE_DENSITY; recomputed from the 4,419 train rows.
    It is also checked that the string is in the yaml of the model specification). Same as the value of average_density.json.
  - mu_uncond and sigma2 are the values of dxvpsde_stats.json (stage E-3). The key of num_atoms is AL_S5_210_TRAIN.
  - For the callbacks, the callbacks list of trainer/default.yaml of MN is read and the list with a ModelCheckpoint that saves every 20 epochs appended at the end is
    written to the sbatch as a hydra override (trainer.callbacks=[...]) (hydra does not allow adding an index to a list; only replacing the whole list is possible).
    For the best-checkpoint saving (callbacks.1: monitor=loss_val, mode=min, save_top_k=1, save_last=true, every_n_epochs=1) an explicit override is also given.
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

import stage_e_common as E

MN = Path(_gbrel.CODE)
MN_CONF = MN / "mattergen" / "conf"
RUN = E.C.RUN
OLD_PROPOSED_RUN = Path(_gbrel.WORK + '/workspace/experiments/generation_hooks')
OLD_BASELINE_RUN = Path(_gbrel.WORK + '/workspace/experiments/baseline_generation')

SRC_PROPOSED_DM = MN_CONF / "data_module" / "scm_al_proposed.yaml"
SRC_PROPOSED_CORR = MN_CONF / "lightning_module" / "diffusion_module" / "corruption" / "scm_al_proposed.yaml"
SRC_VAN_DM = MN_CONF / "data_module" / "scm_al_baseline_dedup.yaml"
SRC_TRAINER = MN_CONF / "trainer" / "default.yaml"

# measurements of the old training (used for the conversion of --time. proposed 7h10m, baseline 6h37m, train 3,489 rows; factor 1.5)
OLD_TRAIN_ROWS = 3489
OLD_HOURS = {"proposed": 7 + 10 / 60, "baseline": 6 + 37 / 60}

PERIODIC_CB = {
    "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
    "every_n_epochs": 20, "monitor": None, "save_top_k": -1, "save_last": False, "filename": "{epoch}-periodic",
}


# ------------------------------------------------------------------ serialization of hydra overrides
_PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-/]*$")


def hydra_literal(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        if _PLAIN.match(v) and v.lower() not in ("true", "false", "null"):
            return v
        if "'" in v or '"' in v:
            raise ValueError(f"cannot quote {v!r}")
        return f"'{v}'"
    if isinstance(v, list):
        return "[" + ",".join(hydra_literal(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{hydra_literal(x)}" for k, x in v.items()) + "}"
    raise TypeError(type(v))


def callbacks_override(trainer_yaml: Path) -> tuple[str, list]:
    cbs = yaml.safe_load(trainer_yaml.read_text())["callbacks"]
    if not (isinstance(cbs, list) and cbs[1]["_target_"].endswith("ModelCheckpoint")):
        raise SystemExit("FATAL: trainer/default.yaml callbacks[1] is not a ModelCheckpoint (best-by-val)")
    new = list(cbs) + [dict(PERIODIC_CB)]
    return "trainer.callbacks=" + hydra_literal(new), new


# ------------------------------------------------------------------ line replacement of yaml
def strip_leading_comments(text: str) -> str:
    lines = text.splitlines()
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    return "\n".join(lines[i:]) + "\n"


def sub_exact(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"FATAL: expected {count} occurrence(s) of {old!r}, found {n}")
    return text.replace(old, new)


def to_node_path(p) -> str:
    s = str(p)
    return s[len("/mnt"):] if s.startswith('/mnt/CLUSTER_ROOT_A') else s


def proposed_data_module(res: str, bulk: str) -> str:
    body = strip_leading_comments(SRC_PROPOSED_DM.read_text())
    body = sub_exact(body, "average_density: 0.062\n", f"average_density: {E.CARD_AVERAGE_DENSITY!r}\n")
    body = sub_exact(body, ('gamma_csv_path: ' + _gbrel.WORK + '/workspace/experiments/earlier_experiment/results/integrated_gamma_dedup.csv'),
                     f"gamma_csv_path: {res}/gamma_dedup.csv", 2)
    body = sub_exact(body, ('bulk_membership_csv_path: ' + _gbrel.WORK + '/workspace/experiments/bulk_membership/bulk_membership.csv'),
                     f"bulk_membership_csv_path: {bulk}", 2)
    body = sub_exact(body, ('manifest_dir: ' + _gbrel.WORK + '/scm-work/experiments/earlier_experiment/manifests'),
                     f"manifest_dir: {res}/manifests", 2)
    body = sub_exact(body, ('split_indices_json_path: ' + _gbrel.WORK + '/workspace/experiments/generation_hooks/results/train_indices_final.json'),
                     f"split_indices_json_path: {res}/split.json", 2)
    head = (
        "# Al rebuilt data, data_module of the proposed model (symmetry constraints, variable length, no padding).\n"
        "# Stage E-5 of the data rebuild.\n"
        "# Generated by: workspace/experiments/data_rebuild/scripts/stage_e_write_configs.py\n"
        "# Transcribed from: data_module/scm_al_proposed.yaml (NOT edited). Only the inputs are replaced:\n"
        "#   gamma_csv_path = gamma_dedup.csv, split_indices_json_path = split.json (train / val; test is not used for the training),\n"
        "#   bulk_membership_csv_path = bulk_membership.csv, manifest_dir = manifests/ ({cnid}.json. 31 templates),\n"
        f"#   average_density = {E.CARD_AVERAGE_DENSITY!r} (value of the model specification, {E.CARD_HASH12}. Recomputed from the 4,419 train rows. Not referenced in the training of the proposed model).\n"
        "# Paths are from the viewpoint of compute nodes (/CLUSTER_ROOT_B/...). expected_size is not set (load_split_indices checks that the counts agree with n_train/n_val of the JSON and len).\n"
        "# Environment variables: export SCM_HOST_Z=13 and SCM_SOLUTE_ZS= (empty) before the training.\n"
    )
    return head + body


def proposed_corruption(mu: float, sigma2: float, n_rows: int) -> str:
    body = strip_leading_comments(SRC_PROPOSED_CORR.read_text())
    body = sub_exact(body, "mu_uncond: -0.09686156491831471\n", f"mu_uncond: {mu!r}\n")
    body = sub_exact(body, "sigma2: 0.15297381316590877\n", f"sigma2: {sigma2!r}\n")
    head = (
        "# Al rebuilt data, corruption config of the proposed model. Stage E-5 of the data rebuild.\n"
        "# Generated by: workspace/experiments/data_rebuild/scripts/stage_e_write_configs.py\n"
        "# Transcribed from: corruption/scm_al_proposed.yaml (NOT edited). Only the statistics of GBTranslationVPSDE of gb_dx (mu_uncond / sigma2) are replaced:\n"
        f"#   computed by stage_e_stats.py from the train rows of gamma_dedup.csv (n_rows={n_rows}) (results/dxvpsde_stats.json).\n"
        "#   mu_uncond = mean(dx_A), sigma2 = population variance (ddof=0).\n"
        "# dim of D3PM is 101 (the default of the original MatterGen; not 102 for padding).\n"
    )
    return head + body


def baseline_data_module() -> str:
    body = strip_leading_comments(SRC_VAN_DM.read_text())
    body = sub_exact(body, "average_density: 0.05836673615633607  # measured on the 3,489 train rows (header above). Not referenced in the training.\n",
                     f"average_density: {E.CARD_AVERAGE_DENSITY!r}  # value of the model specification (see the header). Not referenced in the training.\n")
    body = sub_exact(body, "root_dir: ${oc.env:PROJECT_ROOT}/../datasets/cache/scm_al_baseline_dedup\n",
                     "root_dir: ${oc.env:PROJECT_ROOT}/../datasets/cache/scm_al_baseline\n")
    head = (
        "# Al rebuilt data, data_module of the baseline.\n"
        "# Stage E-6 of the data rebuild.\n"
        "# Generated by: workspace/experiments/data_rebuild/scripts/stage_e_write_configs.py\n"
        "# Transcribed from: data_module/scm_al_baseline_dedup.yaml (NOT edited). Only the cache path and the value of average_density are replaced:\n"
        "#   root_dir: MN/datasets/cache/scm_al_baseline/{train,val,test}\n"
        "#     created by: build_tilt_baseline_csv.py (CSV. MN/datasets/csv/scm_al_baseline/)\n"
        "#           -> sbatch_prep_cache_baseline.sh (caching with mattergen.scripts.csv_to_dataset. The old cache is not overwritten)\n"
        "#     The row sets of train/val/test are train/val/test of split.json (same as the proposed model. test is not used for the training).\n"
        f"#   average_density: value of the model specification {E.CARD_AVERAGE_DENSITY!r} ({E.CARD_HASH12}. Recomputed from the 4,419 train rows. Not referenced in the training).\n"
        "# The conditioning property gb_gamma_J_m2, transforms, dataset_transforms and batch_size are the same as in the old baseline.\n"
    )
    return head + body


# ------------------------------------------------------------------ sbatch
def fmt_time(hours: float) -> str:
    total_min = int(-(-hours * 60 // 1))   # round up (minutes)
    d, rem = divmod(total_min, 24 * 60)
    h, m = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:00" if d else f"{h:02d}:{m:02d}:00"


def train_time(kind: str, n_train: int) -> tuple[str, float]:
    hours = OLD_HOURS[kind] * 1.5 * n_train / OLD_TRAIN_ROWS
    return fmt_time(hours), hours


COMMON_HEAD = r"""#!/bin/bash
# @@TITLE@@
# Stage E-@@SEC@@ of the data rebuild.
# Generated by: workspace/experiments/data_rebuild/scripts/stage_e_write_configs.py (--time converted from n_train=@@NTRAIN@@)
#
# [This file is not submitted] Submission is after the revision and re-approval of the model specification.
#   SMOKE=1 sbatch --time=00:40:00 --exclude=<EXCLUDE_NODES> @@NAME@@        # max_epochs=1, limit_train_batches=5
#   sbatch --exclude=... @@NAME@@                                                          # production training, 200 epochs
# Use shard:40 as the default gres (a whole GPU is often blocked by shard jobs). Always give --time also on the command line.
# Conversion of --time: earlier measurement @@OLDH@@ (train @@OLDN@@ rows) x 1.5 x n_train/@@OLDN@@ = @@HOURS@@ hours -> @@TIME@@.
#
# Transcribed from: @@SRC@@ (activation line, PYTHONPATH, cd, set -euo pipefail, venv reachability check, foreground execution, switch of CSVLogger, mechanism of the run-time print).
# Differences from the source (only what the plan requires):
#   (1) Data and settings: @@DIFF1@@
#   (2) num_atoms_distribution_key=AL_S5_210_TRAIN (a key for the record that the training does not reference. It is confirmed with resolved_config.yaml that the generation side resolves to the same value)
#   (3) callbacks (as in the model specification): trainer.callbacks is replaced by overriding the whole list (hydra cannot add an index to a list).
#       The content is the callbacks of trainer/default.yaml of MN plus a ModelCheckpoint that saves every 20 epochs
#       (every_n_epochs=20, monitor=null, save_top_k=-1, save_last=false, filename '{epoch}-periodic'; following the effective settings of an earlier model) appended
#       at the end (index 4). For the best-checkpoint saving trainer.callbacks.1 (monitor=loss_val, mode=min, save_top_k=1, save_last=true, every_n_epochs=1) an explicit override is also given.
#       The 6 lines of the earlier sbatch that keep epoch=199.ckpt (monitor removed, save_top_k=-1) are not used (the weights with the best val_loss are used for the evaluation).
#   (4) Output: @@MNRUN@@ (SMOKE=1 goes to smoke/ below it)
# [Author's choice] SMOKE=1 gives, in addition to max_epochs=1 and limit_train_batches=5, trainer.check_val_every_n_epoch=1 and limit_val_batches=5
#   (so that the validation runs once and the ModelCheckpoint route that monitors loss_val is really passed). DEMO=1 is the same as the source (stops just before trainer.fit).
#
#SBATCH --job-name=@@JOB@@
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=shard:40
#SBATCH --time=@@TIME@@
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=${GBREL_WORK}/workspace/experiments/data_rebuild/logs/@@LOGSTEM@@_%j.out
#SBATCH --error=${GBREL_WORK}/workspace/experiments/data_rebuild/logs/@@LOGSTEM@@_%j.err

set -euo pipefail

# Activation lines are transcribed verbatim from the source.
source ${GEN_VENV}/bin/activate
export PYTHONPATH=${GBREL_CODE}
@@ENVEXTRA@@
cd ${GBREL_CODE}

GC_RUN=${GBREL_WORK}/workspace/experiments/data_rebuild
OLD_PROPOSED_RUN=@@OLDPROPOSEDRUN@@   # location of run_with_hparam_dump.py (read only)
MN_RUN=@@MNRUN@@
SMOKE="${SMOKE:-0}"
if [ "${SMOKE}" = "1" ]; then
  RUN_DIR="${MN_RUN}/smoke"
  MAX_EPOCHS=1
  LOGGER_NAME=@@LOGGER@@_smoke
  HPARAM_BASE=@@HPARAM@@
  EXTRA_OVERRIDES=(+trainer.limit_train_batches=5 ++trainer.limit_val_batches=5 trainer.check_val_every_n_epoch=1 ++trainer.log_every_n_steps=1)
else
  RUN_DIR="${MN_RUN}"
  MAX_EPOCHS=200
  LOGGER_NAME=@@LOGGER@@_train
  HPARAM_BASE=@@HPARAM@@_full
  EXTRA_OVERRIDES=()
fi
# DEMO=1: demonstration of the run-time print. Runs the same command route as in production up to just before trainer.fit, writes the JSON and exits (--dump-only; no training).
DEMO="${DEMO:-0}"
DUMP_ONLY_FLAG=()
if [ "${DEMO}" = "1" ]; then
  RUN_DIR="${GC_RUN}/tmp/@@KIND@@_demo_${SLURM_JOB_ID:-local}"
  MAX_EPOCHS=2
  LOGGER_NAME=@@LOGGER@@_demo
  HPARAM_BASE=@@HPARAM@@_demo
  EXTRA_OVERRIDES=(+trainer.limit_train_batches=5)
  DUMP_ONLY_FLAG=(--dump-only)
fi
if [ -e "${RUN_DIR}/train_stdout.log" ]; then
  echo "[FAIL] ${RUN_DIR}/train_stdout.log already exists (an existing run is not overwritten)" >&2
  exit 1
fi
export OUTPUT_DIR="${RUN_DIR}/hydra_outputs"
mkdir -p "${OUTPUT_DIR}" "${RUN_DIR}/csv_logs" "${GC_RUN}/verification"

HPARAM_JSON="${GC_RUN}/verification/${HPARAM_BASE}.json"
if [ -e "${HPARAM_JSON}" ]; then HPARAM_JSON="${GC_RUN}/verification/${HPARAM_BASE}.${SLURM_JOB_ID:-nojob}.json"; fi

echo "host: $(hostname)"
echo "start: $(date -u)"
echo "SMOKE=${SMOKE} DEMO=${DEMO} RUN_DIR=${RUN_DIR} MAX_EPOCHS=${MAX_EPOCHS}"
@@ECHOEXTRA@@
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv || true

ACTIVATE_PATH=${GEN_VENV}/bin/activate
if [ ! -f "${ACTIVATE_PATH}" ]; then
  echo "[FAIL] activate path not reachable from compute node: ${ACTIVATE_PATH}"
  exit 1
fi
echo "[OK] activate path reachable on compute node: ${ACTIVATE_PATH}"

COMMON_OVERRIDES=(
  --config-name @@CONFIGNAME@@
  @@GROUPS@@
  num_atoms_distribution_key=@@NUMKEY@@
@@PRETRAINED@@  trainer.max_epochs=${MAX_EPOCHS}
  lightning_module.optimizer_partial.lr=5.0e-06
  +seed=20290000
  # logger: switched from the default WandbLogger to CSVLogger (as in the source. Wandb authentication on compute nodes is not presupposed)
  trainer.logger._target_=pytorch_lightning.loggers.CSVLogger
  +trainer.logger.name=${LOGGER_NAME}
  +trainer.logger.save_dir=${RUN_DIR}/csv_logs
  ~trainer.logger.project
  ~trainer.logger.entity
  ~trainer.logger.job_type
  ~trainer.logger.settings
  # callbacks: replacement of the whole list (best-checkpoint saving + saving every 20 epochs; see (3) in the header above). First the list, then the explicit override of index 1.
  "@@CALLBACKS@@"
  trainer.callbacks.1.every_n_epochs=1
  trainer.callbacks.1.monitor=loss_val
  trainer.callbacks.1.mode=min
  trainer.callbacks.1.save_top_k=1
  trainer.callbacks.1.save_last=true
  ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}
)

echo "COMMON_OVERRIDES: ${COMMON_OVERRIDES[*]}"

"""

PROPOSED_TAIL = r"""# Save the whole resolved config before the start (--cfg job --resolve; no GPU computation, training not started).
python -m mattergen.scripts.finetune_scm_cond "${COMMON_OVERRIDES[@]}" --resolve --cfg job \
  > "${RUN_DIR}/resolved_config.yaml" \
  2> "${RUN_DIR}/resolved_config.stderr.log"

# 5b(v): print the HOST_Z and the num_atoms key resolved by the training side (the generation side prints the same two values).
echo "resolved(config) num_atoms_distribution_key: $(grep -m1 '^num_atoms_distribution_key' "${RUN_DIR}/resolved_config.yaml")"

# Production training (run in the foreground as the body of the sbatch job. No background). run_with_hparam_dump.py writes the effective hyperparameters
# to ${HPARAM_JSON} just before trainer.fit and then calls the real fit.
python "${OLD_PROPOSED_RUN}/scripts/run_with_hparam_dump.py" \
  --entry mattergen.scripts.finetune_scm_cond --kind proposed --out "${HPARAM_JSON}" ${DUMP_ONLY_FLAG[@]+"${DUMP_ONLY_FLAG[@]}"} \
  -- "${COMMON_OVERRIDES[@]}" \
  > "${RUN_DIR}/train_stdout.log" 2>&1

rc=$?
echo "Exit code: ${rc}"
echo "end: $(date -u)"
find "${RUN_DIR}/csv_logs" -name "*.ckpt" 2>/dev/null
exit "${rc}"
"""

BASELINE_TAIL = r"""# The resolved config is saved with --cfg job (without --resolve: adapter.property_embeddings_adapt
# can be resolved only after finetune.py has taken in the denoiser config of the pretrained model,
# so --resolve gives InterpolationKeyError. Same as the measurement of the earlier baseline). The resolved effective values are seen in the run-time print.
python -m mattergen.scripts.finetune "${COMMON_OVERRIDES[@]}" --cfg job \
  > "${RUN_DIR}/resolved_config.yaml" \
  2> "${RUN_DIR}/resolved_config.stderr.log"

echo "resolved(config) num_atoms_distribution_key: $(grep -m1 '^num_atoms_distribution_key' "${RUN_DIR}/resolved_config.yaml")"

# Production training (foreground execution. No background). finetune.py (the entry point of the baseline) does not read +seed, so the wrapper's --seed 20290000
# calls pl.seed_everything before the instantiation to make it effective (same as the source).
python "${OLD_PROPOSED_RUN}/scripts/run_with_hparam_dump.py" \
  --entry mattergen.scripts.finetune --kind baseline --seed 20290000 --out "${HPARAM_JSON}" ${DUMP_ONLY_FLAG[@]+"${DUMP_ONLY_FLAG[@]}"} \
  -- "${COMMON_OVERRIDES[@]}" \
  > "${RUN_DIR}/train_stdout.log" 2>&1

rc=$?
echo "Exit code: ${rc}"
echo "end: $(date -u)"
find "${RUN_DIR}/csv_logs" -name "*.ckpt" 2>/dev/null
exit "${rc}"
"""


def fill(tpl: str, kv: dict) -> str:
    for k, v in kv.items():
        tpl = tpl.replace(f"@@{k}@@", v)
    left = re.findall(r"@@[A-Z0-9_]+@@", tpl)
    if left:
        raise SystemExit(f"FATAL: unfilled placeholders {sorted(set(left))}")
    return tpl


def proposed_sbatch(n_train: int, cb_override: str) -> str:
    time_s, hours = train_time("proposed", n_train)
    kv = {
        "TITLE": "Training sbatch for the proposed model (set of symmetry constraints, variable length, no padding) on the rebuilt Al data.", "SEC": "5",
        "NTRAIN": str(n_train), "NAME": "sbatch_train_al_proposed.sh", "OLDH": "7h10m", "OLDN": str(OLD_TRAIN_ROWS),
        "HOURS": f"{hours:.3f}", "TIME": time_s,
        "SRC": "GC/experiments/generation_hooks/scripts/sbatch_train_al_proposed.sh",
        "DIFF1": "--config-name stays that of the earlier proposed model, finetune-scm-al-proposed, and data_module=scm_al_proposed and\n"
                 "#       lightning_module/diffusion_module/corruption=scm_al_proposed are replaced by group overrides (the new yaml files are placed in MN/mattergen/conf).",
        "MNRUN": E.RUN_MN_PROPOSED, "JOB": "al_proposed_train", "LOGSTEM": "train_proposed", "KIND": "proposed", "LOGGER": "al_proposed",
        "HPARAM": "hparam_runtime_proposed",
        "ENVEXTRA": "\n# For Al, SCM_HOST_Z=13 is given explicitly (an exception if unset), and since the default of SCM_SOLUTE_ZS is {12}, it is given explicitly as empty for pure Al.\n"
                    "export SCM_HOST_Z=13\nexport SCM_SOLUTE_ZS=\n",
        "OLDPROPOSEDRUN": str(OLD_PROPOSED_RUN),
        "ECHOEXTRA": 'echo "SCM_HOST_Z=${SCM_HOST_Z} SCM_SOLUTE_ZS=${SCM_SOLUTE_ZS}"',
        "CONFIGNAME": "finetune-scm-al-proposed",
        "GROUPS": "data_module=scm_al_proposed\n  lightning_module/diffusion_module/corruption=scm_al_proposed",
        "NUMKEY": E.NUM_ATOMS_KEY,
        "PRETRAINED": "  pretrained_name=mattergen_base\n  model_path=null\n",
        "CALLBACKS": cb_override,
    }
    return fill(COMMON_HEAD, kv) + PROPOSED_TAIL


def baseline_sbatch(n_train: int, cb_override: str) -> str:
    time_s, hours = train_time("baseline", n_train)
    kv = {
        "TITLE": "Training sbatch for the baseline (original MatterGen, same split as the proposed model) on the rebuilt Al data.", "SEC": "6",
        "NTRAIN": str(n_train), "NAME": "sbatch_train_al_baseline.sh", "OLDH": "6h37m", "OLDN": str(OLD_TRAIN_ROWS),
        "HOURS": f"{hours:.3f}", "TIME": time_s,
        "SRC": "GC/experiments/baseline_generation/scripts/sbatch_train_al_baseline_dedup.sh",
        "DIFF1": "--config-name stays that of the earlier baseline, finetune-scm-al-baseline-dedup, and data_module=scm_al_baseline is replaced by a group override\n"
                 "#       (the cache is MN/datasets/cache/scm_al_baseline/{train,val,test}).",
        "MNRUN": E.RUN_MN_BASELINE, "JOB": "al_baseline_train", "LOGSTEM": "train_baseline", "KIND": "baseline",
        "LOGGER": "al_baseline", "HPARAM": "hparam_runtime_baseline",
        "ENVEXTRA": "", "OLDPROPOSEDRUN": str(OLD_PROPOSED_RUN), "ECHOEXTRA": "",
        "CONFIGNAME": "finetune-scm-al-baseline-dedup",
        "GROUPS": "data_module=scm_al_baseline",
        "NUMKEY": E.NUM_ATOMS_KEY, "PRETRAINED": "", "CALLBACKS": cb_override,
    }
    return fill(COMMON_HEAD, kv) + BASELINE_TAIL


BUILD_CSV_SBATCH = r"""#!/bin/bash
# Creation of the CSV of the baseline (data rebuild, stage E-6): a call of build_tilt_baseline_csv.py.
# Makes MN/datasets/csv/scm_al_baseline/{train,val,test}.csv from gamma_dedup.csv and split.json (train/val/test).
# Generated by: workspace/experiments/data_rebuild/scripts/stage_e_write_configs.py
# [This file is not submitted this time] To be submitted after the completion of stage D2 and stages E-1 to E-4. CPU partition. No --time (not needed on that cluster).
# python: the evaluation environment (pymatgen only; mattergen is not imported). An exception if the output destination has existing files (no overwriting).
#
#SBATCH --job-name=al_baseline_csv
#SBATCH --partition=hx1-16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=${GBREL_WORK}/workspace/experiments/data_rebuild/logs/build_baseline_csv_%j.out
#SBATCH --error=${GBREL_WORK}/workspace/experiments/data_rebuild/logs/build_baseline_csv_%j.err
set -uo pipefail
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
RUN=${GBREL_WORK}/workspace/experiments/data_rebuild
"$PY" -c "import pymatgen, ase" || { echo "[FAIL] env import"; exit 1; }
"$PY" "$RUN/scripts/build_tilt_baseline_csv.py" "$@"
rc=$?
echo "rc build_tilt_baseline_csv=${rc}"
exit "${rc}"
"""

PREP_CACHE_SBATCH = r"""#!/bin/bash
# Caching of the baseline (data rebuild, stage E-6): csv_to_dataset -> new cache name scm_al_baseline; the old cache scm_al_baseline_dedup is not touched.
# Transcribed from: GC/runs/baseline_generation/scripts/sbatch_prep_cache_and_check_dedup.sh (Steps 1, 3 and 4; based on verbatim copy).
# Differences: (1) CSV_DIR = MN/datasets/csv/scm_al_baseline, DSET = scm_al_baseline. Abort without overwriting if the cache already exists.
#       (2) Record the modification time of the old cache scm_al_baseline_dedup before and after and check that it is unchanged.
#       (3) Cross-check the number of rows of the CSV with n_* of split.json.
#       (4) The validation and DEMO point to the new baseline sbatch and data_module. The old Step 2 (check of the old split) is for the old fixed split, so it is not included
#           (the check of the row set of the new cache is done in the step above). Step 5 (DEMO of the proposed model) belongs to the sbatch of the proposed model, so it is not included.
# Reason: the import of mattergen needs GLIBC_2.32 of torch_scatter, which is not available on the login/CPU nodes, so a GPU is not used but the job is run on the smallest shard of the GPU partition.
# Generated by: workspace/experiments/data_rebuild/scripts/stage_e_write_configs.py
# [This file is not submitted this time] To be submitted after the completion of build_tilt_baseline_csv.py.
#   sbatch --exclude=<EXCLUDE_NODES> sbatch_prep_cache_baseline.sh
#
#SBATCH --job-name=al_baseline_prep
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=shard:8
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=${GBREL_WORK}/workspace/experiments/data_rebuild/logs/prep_cache_%j.out
#SBATCH --error=${GBREL_WORK}/workspace/experiments/data_rebuild/logs/prep_cache_%j.err

set -uo pipefail

# Activation lines are transcribed verbatim from the source.
source ${GEN_VENV}/bin/activate
export PYTHONPATH=${GBREL_CODE}

cd ${GBREL_CODE}

RUN_DIR=${GBREL_WORK}/workspace/experiments/data_rebuild
RESULTS_DIR="${RUN_DIR}/results"
VER_DIR="${RUN_DIR}/verification"
mkdir -p "${RESULTS_DIR}" "${VER_DIR}"

echo "host: $(hostname)"
echo "start: $(date -u)"
ACTIVATE_PATH=${GEN_VENV}/bin/activate
[ -f "${ACTIVATE_PATH}" ] || { echo "[FAIL] activate path not reachable: ${ACTIVATE_PATH}"; exit 1; }
echo "[OK] activate path reachable on compute node: ${ACTIVATE_PATH}"

CSV_DIR=${GBREL_WORK}/scm-work/datasets/csv/scm_al_baseline
CACHE_DIR=${GBREL_WORK}/scm-work/datasets/cache
DSET=scm_al_baseline
OLD_CACHE="${CACHE_DIR}/scm_al_baseline_dedup"
if [ -e "${CACHE_DIR}/${DSET}" ]; then
  echo "[FAIL] ${CACHE_DIR}/${DSET} already exists (not overwritten)"; exit 1
fi
OLD_MTIME_BEFORE=$(stat -c %Y "${OLD_CACHE}" 2>/dev/null || echo missing)
echo "old cache mtime before: ${OLD_MTIME_BEFORE}"

rc_cache=0; rc_rows=0; rc_oldcache=0; rc_resolve=0; rc_v2=0; rc_demo=0

# ---- Step 0: cross-check the number of rows of the CSV with n_* of split.json ----
python - "${CSV_DIR}" "${RESULTS_DIR}/split.json" <<'PYEOF' | tee "${VER_DIR}/check-12_csv_rows.log"
import csv, json, sys
csv.field_size_limit(sys.maxsize)
d, sp = sys.argv[1], json.load(open(sys.argv[2]))
ok = True
for s in ("train", "val", "test"):
    with open(f"{d}/{s}.csv", newline="") as f:
        n = sum(1 for _ in csv.DictReader(f))
    good = n == sp[f"n_{s}"]
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] {s}.csv rows={n} n_{s}={sp[f'n_{s}']}")
sys.exit(0 if ok else 1)
PYEOF
rc_rows=${PIPESTATUS[0]}
echo "rc csv rows=${rc_rows}"

# ---- Step 1: caching with csv_to_dataset.py ----
echo "=== csv_to_dataset.py: ${CSV_DIR} -> ${CACHE_DIR}/${DSET} ==="
python -m mattergen.scripts.csv_to_dataset \
  --csv-folder "${CSV_DIR}" --dataset-name "${DSET}" --cache-folder "${CACHE_DIR}" \
  2>&1 | tee "${RESULTS_DIR}/csv_to_dataset.log"
rc_cache=${PIPESTATUS[0]}
echo "rc csv_to_dataset=${rc_cache}"
find "${CACHE_DIR}/${DSET}" -maxdepth 2 2>/dev/null

OLD_MTIME_AFTER=$(stat -c %Y "${OLD_CACHE}" 2>/dev/null || echo missing)
echo "old cache mtime after: ${OLD_MTIME_AFTER}"
[ "${OLD_MTIME_BEFORE}" = "${OLD_MTIME_AFTER}" ] || rc_oldcache=1
echo "rc old cache unchanged=${rc_oldcache}"

# ---- Step 3: validation (the resolved config of the baseline has no GB mechanism / it fails with the config of this method) ----
export OUTPUT_DIR="${RUN_DIR}/tmp/v2a_resolve"
mkdir -p "${OUTPUT_DIR}"
python -m mattergen.scripts.finetune \
  --config-name finetune-scm-al-baseline-dedup data_module=scm_al_baseline \
  num_atoms_distribution_key=AL_S5_210_TRAIN +seed=20290000 --cfg job \
  > "${RESULTS_DIR}/resolved_config_baseline.yaml" \
  2> "${RESULTS_DIR}/resolved_config_baseline.stderr.log"
rc_resolve=$?
echo "rc resolve=${rc_resolve} lines=$(wc -l < "${RESULTS_DIR}/resolved_config_baseline.yaml")"
VALIDATION_SCRIPT=${GBREL_CODE}/gb_eval/experiments/baseline_reference/scripts/validation_v2_gb_zero_check.py
C38FIXN_RESOLVED=${GBREL_WORK}/scm-work/experiments/earlier_experiment/resolved_config.yaml
{
  echo "=== V2(a): zero-GB-mechanism check of baseline ==="
  python "${VALIDATION_SCRIPT}" --resolved-config "${RESULTS_DIR}/resolved_config_baseline.yaml" \
    --context "scm_al_baseline (--cfg job)"
  echo "exit code: $?"
  echo "=== V2(b): resolved_config.yaml of the reference model (the side that fails; non-zero is the expected value) ==="
  python "${VALIDATION_SCRIPT}" --resolved-config "${REFERENCE_RESOLVED}" --context "reference model (the side that fails)"
  echo "exit code (expected nonzero): $?"
} > "${VER_DIR}/V2_gb_zero_check.log" 2>&1
cat "${VER_DIR}/V2_gb_zero_check.log"
grep -q "^exit code: 0" "${VER_DIR}/V2_gb_zero_check.log" || rc_v2=1

# ---- Step 4: demonstration of the run-time print (the training sbatch with DEMO=1; it stops just before trainer.fit) ----
DEMO=1 bash "${RUN_DIR}/scripts/sbatch_train_al_baseline.sh" > "${VER_DIR}/hparam_demo_run_baseline.log" 2>&1
rc_demo=$?
echo "rc demo=${rc_demo}"; tail -40 "${VER_DIR}/hparam_demo_run_baseline.log"

echo "rc summary: rows=${rc_rows} cache=${rc_cache} old_cache_unchanged=${rc_oldcache} resolve=${rc_resolve} v2a=${rc_v2} demo=${rc_demo}"
echo "end: $(date -u)"
[ "${rc_rows}" -eq 0 ] && [ "${rc_cache}" -eq 0 ] && [ "${rc_oldcache}" -eq 0 ] && [ "${rc_resolve}" -eq 0 ] && [ "${rc_v2}" -eq 0 ] && [ "${rc_demo}" -eq 0 ]
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out-dir", default=None, help="output destination for tests (written with the same structure as MN)")
    g.add_argument("--production", action="store_true", help="production output destinations (MN/mattergen/conf and RUN/scripts)")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--bulk-membership", default=str(RUN / "results" / "bulk_membership.csv"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    res = E.results_dir(a.results_dir)
    res_node = to_node_path(res)
    bulk_node = to_node_path(a.bulk_membership)
    card_state = E.check_card_density()
    stats = json.loads((res / "dxvpsde_stats.json").read_text())
    split = E.load_split(res)
    n_train = int(split["n_train"])
    if stats["n_rows"] != n_train:
        raise SystemExit(f"FATAL: dxvpsde_stats n_rows={stats['n_rows']} != split n_train={n_train}")

    if a.production:
        conf_root, script_root = MN_CONF, RUN / "scripts"
    else:
        conf_root, script_root = Path(a.out_dir) / "mattergen" / "conf", Path(a.out_dir) / "scripts"

    cb_override, cb_list = callbacks_override(SRC_TRAINER)
    files = {
        conf_root / "data_module" / "scm_al_proposed.yaml": proposed_data_module(res_node, bulk_node),
        conf_root / "lightning_module" / "diffusion_module" / "corruption" / "scm_al_proposed.yaml":
            proposed_corruption(stats["mu_uncond"], stats["sigma2"], stats["n_rows"]),
        conf_root / "data_module" / "scm_al_baseline.yaml": baseline_data_module(),
        script_root / "sbatch_train_al_proposed.sh": proposed_sbatch(n_train, cb_override),
        script_root / "sbatch_build_baseline_csv.sh": BUILD_CSV_SBATCH,
        script_root / "sbatch_prep_cache_baseline.sh": PREP_CACHE_SBATCH,
        script_root / "sbatch_train_al_baseline.sh": baseline_sbatch(n_train, cb_override),
    }
    for p in files:
        if p.exists() and not a.force:
            raise SystemExit(f"FATAL: {p} exists (overwriting is prohibited; use --force to overwrite)")
    manifest = {"design": "data rebuild stages E-5 and E-6", "results_dir": str(res), "n_train": n_train,
                "card_hash12": E.CARD_HASH12, "card_average_density": E.CARD_AVERAGE_DENSITY, "card_density_check": card_state,
                "mu_uncond": stats["mu_uncond"], "sigma2": stats["sigma2"], "num_atoms_key": E.NUM_ATOMS_KEY,
                "callbacks_override": cb_override, "callbacks_list": cb_list, "files": {}}
    for p, text in files.items():
        E.write_text(p, text, a.force)
        if p.suffix == ".sh":
            p.chmod(0o755)
        manifest["files"][str(p)] = hashlib.sha256(text.encode()).hexdigest()
        print(f"WROTE {p}")
    root = Path(a.out_dir) if a.out_dir else RUN / "results"
    E.write_json(root / "stage_e_write_configs_manifest.json", manifest, a.force)
    print(f"callbacks override: {cb_override}")
    print(f"WROTE {root / 'stage_e_write_configs_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
