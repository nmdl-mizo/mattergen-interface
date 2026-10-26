#!/usr/bin/env python3
"""Common parts of stage E of the data rebuild (rounding of the split, input and output). Runs on compute nodes. Paths are written with the <CLUSTER_MOUNT> notation.

The results directory is switched with --results-dir of each script or the environment variable DATA_RESULTS_DIR (the default is the production RUN/results).
Outputs never overwrite existing files (an exception unless --force is given).
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import builder_common as C  # noqa: E402
import stage_d_common as D  # noqa: E402

SPLIT_SEED = 42                                   # stage E-1
SPLIT_RATIO = (8, 1, 1)                           # train:val:test (split.recipe of the model specification)
SPLIT_JSON = "split.json"
CHUNK_SEED_BASE = 20301000                        # stage E-3 seed of the generation
N_CHUNKS = 14
# The average density is the value of the model specification (bbd1307e34eb; recomputed from the 4,419 training rows). The same as the value of average_density.json.
CARD_AVERAGE_DENSITY = 0.05839622542107227
CARD_YAML = Path(_gbrel.WORK + '/workspace/model_spec.yaml')
CARD_HASH12 = "bbd1307e34eb"
NUM_ATOMS_KEY = "AL_S5_210_TRAIN"
ALLOWED_NATOMS = {112, 114, 116, 118, 120}
RUN_MN_PROPOSED = _gbrel.WORK + '/scm-work/experiments/proposed_train'
RUN_MN_BASELINE = _gbrel.WORK + '/scm-work/experiments/baseline_train'


def results_dir(arg=None) -> Path:
    return D.results_dir(arg)


def split_counts(n_templates: int, ratio=SPLIT_RATIO, min_each: int = 1) -> tuple[int, int, int]:
    """Gives (n_train, n_val, n_test) from the number of templates.
    Rule: n_val = n_test = max(min_each, floor(n * r / sum(ratio))) (integer arithmetic), n_train = n - n_val - n_test.
    For n=31, floor(31*1/10)=3 -> 25/3/3. min_each is a floor for small tests with n<10 and does not matter for n>=10.
    An exception for an n that gives n_train < 1."""
    tr, va, te = ratio
    tot = tr + va + te
    n_val = max(min_each, n_templates * va // tot)
    n_test = max(min_each, n_templates * te // tot)
    n_train = n_templates - n_val - n_test
    if n_train < 1:
        raise ValueError(f"split_counts: n_templates={n_templates} too small (train={n_train}, val={n_val}, test={n_test})")
    return n_train, n_val, n_test


def check_card_density() -> str:
    """Checks that the string CARD_AVERAGE_DENSITY is in the model-specification yaml (prevents a discrepancy). An exception if absent, 'card_not_readable' if the file is absent."""
    if not CARD_YAML.exists():
        return "card_not_readable"
    if repr(CARD_AVERAGE_DENSITY) not in CARD_YAML.read_text():
        raise SystemExit(f"FATAL: card {CARD_YAML} does not contain average_density {CARD_AVERAGE_DENSITY!r}")
    return "card_contains_value"


def write_json(path: Path, obj, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise SystemExit(f"FATAL: {path} exists (overwriting is prohibited; use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def write_text(path: Path, text: str, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise SystemExit(f"FATAL: {path} exists (overwriting is prohibited; use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def read_dedup(res: Path) -> list[dict]:
    """Reads gamma_dedup.csv with types. Requires that the space of the row positions (the list of converged rows only) agrees with the whole CSV."""
    p = Path(res) / C_GAMMA_DEDUP
    rows = D.read_gamma(p)
    bad = [r["row"] for r in rows if not r["converged"]]
    if bad:
        raise SystemExit(f"FATAL: {p} has non-converged rows (positions would differ from dataset.py's converged-filtered list): {bad[:5]}")
    return rows


C_GAMMA_DEDUP = D.GAMMA_DEDUP


def load_split(res: Path) -> dict:
    with open(Path(res) / SPLIT_JSON) as f:
        return json.load(f)


def train_rows(rows: list[dict], split: dict) -> list[dict]:
    pos = split["train_indices"]
    if any((not isinstance(p, int)) or p < 0 or p >= len(rows) for p in pos):
        raise SystemExit("FATAL: train_indices out of range")
    return [rows[p] for p in pos]
