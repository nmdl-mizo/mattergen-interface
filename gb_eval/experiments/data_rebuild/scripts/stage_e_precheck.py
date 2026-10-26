#!/usr/bin/env python3
"""stage_e_precheck.py -- data rebuild, stage E-7 (pre-check of the inputs of the proposed model).

Transcribed from (read only): GC/runs/generation_hooks/scripts/precheck_al_variable_length_rows.py. The differences are only what the plan requires:
  (1) Input: gamma_dedup.csv and bulk_membership.csv of <results-dir> (bulk_membership can be replaced with --bulk-membership; the default is the
      production RUN/results), manifests/ and split.json.
  (2) EXPECT is read from n_train, n_val and n_test of split.json (the old one was fixed at 3489/33/356). The split loop is ("train","val","test").
  (3) The output is precheck_rows.csv and precheck_summary.json in <out-dir> (default <results-dir>).

For all rows of gamma_dedup.csv it goes through ``__getitem__`` of the dataset of the variable-length proposed model (SymmetricTiltGBDataset of MN/mattergen/symmetry_constrained/variable_length/dataset.py,
the row-position indices JSON form), and measures the following for each row. Failing rows are not discarded; the results of all rows are given (it does not stop).

  - success of the derivation of the FIXED_B and SLAVE correspondences (``_derive_fixed_b_correspondence`` / ``_derive_slave_correspondence``.
    The same functions as in __getitem__ are called individually with the same inputs and decided independently, so that SLAVE can be decided even if FIXED_B fails)
  - reconstruction residual of the projection (A): the production SymmetryProjector is applied to the ChemGraph (float64, CPU) produced by the dataset, and
    the maximum distance in the minimum image when each atom of FIXED_A, FIXED_B and SLAVE is compared "with the atom at the same position" (per-atom index agreement,
    not set matching) and the maximum of the cell difference. `proj_resid_max_A` = the maximum of these.
    In addition, the nearest-neighbor residual of the set-matching side returned by the correspondence derivation (fixed_b_nn_resid_A / slave_nn_resid_A) is also recorded.
  - success of ds[i] (the production route itself). If there is an exception, its type and string.

The dataset is built with dtype=torch.float64 and on CPU (the default float32 does not reach 1e-9 A).
The environment variables SCM_HOST_Z=13 and SCM_SOLUTE_ZS= (empty) are set explicitly. An exception if other values are already set.

Output (existing files are not overwritten and raise an exception; overwrite with --force):
  <out-dir>/precheck_rows.csv, <out-dir>/precheck_summary.json

Execution environment (do not start a plain python on the login node; mattergen is imported, so the venv of MN is used):
  ssh <LOGIN_NODE> "srun -p <CPU_PARTITION> -n1 -c2 --immediate=60 ${GEN_PY} <this file> --results-dir DIR"
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# --- the environment variables are set explicitly before the import of mattergen (constants.py. Unset is an exception; the default solute set is {12}) ---
for _k, _v in (("SCM_HOST_Z", "13"), ("SCM_SOLUTE_ZS", "")):
    if _k in os.environ and os.environ[_k] != _v:
        raise SystemExit(f"FATAL: {_k}={os.environ[_k]!r} (this precheck requires {_v!r})")
    os.environ[_k] = _v

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)

GC = Path(_gbrel.WORK + '/workspace')
MN = Path(_gbrel.CODE)
RUN = GC / "experiments/data_rebuild"
RESULTS_DEFAULT = Path(os.environ.get("DATA_RESULTS_DIR") or RUN / "results")
SPLITS = ("train", "val", "test")
CHECK2_THRESHOLD_A = 1e-9  # absolute tolerance as a sufficient condition of PASS. Rows exceeding it are not FAIL but are listed and returned to the parent


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mic_cart_dist(a: torch.Tensor, b: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    diff = a - b
    diff = diff - torch.round(diff)
    return torch.linalg.norm(diff @ cell, dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DEFAULT))
    ap.add_argument("--bulk-membership", default=str(RUN / "results" / "bulk_membership.csv"))
    ap.add_argument("--out-dir", default=None, help="default: --results-dir")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    res = Path(args.results_dir)
    GAMMA_CSV = res / "gamma_dedup.csv"
    BULK_MEMBERSHIP = Path(args.bulk_membership)
    MANIFEST_DIR = res / "manifests"
    indices_json = res / "split.json"
    with open(indices_json) as f:
        _sp = json.load(f)
    EXPECT = {"train": int(_sp["n_train"]), "val": int(_sp["n_val"]), "test": int(_sp["n_test"])}
    EXPECT_TOTAL = sum(EXPECT.values())

    out_dir = Path(args.out_dir) if args.out_dir else res
    out_rows = out_dir / "precheck_rows.csv"
    out_summary = out_dir / "precheck_summary.json"
    for p in (out_rows, out_summary):
        if p.exists() and not args.force:
            raise SystemExit(f"FATAL: the output already exists (not overwritten. Use --force to overwrite): {p}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from mattergen.common.data.collate import collate
    from mattergen.symmetry_constrained.constants import SCM_HOST_Z, SCM_SOLUTE_ZS
    from mattergen.symmetry_constrained.layout import build_canonical_layout, read_poscar_frac_species
    from mattergen.symmetry_constrained.variable_length import dataset as dsmod
    from mattergen.symmetry_constrained.variable_length.dataset import (
        SymmetricTiltGBDataset,
        _derive_fixed_b_correspondence,
        _derive_slave_correspondence,
        resolve_view,
    )
    from mattergen.symmetry_constrained.variable_length.layout import (
        GB_BLOCK_FIXED_A,
        GB_BLOCK_FIXED_B,
        GB_BLOCK_MASTER,
        GB_BLOCK_SLAVE,
    )
    from mattergen.symmetry_constrained.variable_length.projection import SymmetryProjector

    assert SCM_HOST_Z == 13, SCM_HOST_Z
    assert SCM_SOLUTE_ZS == frozenset(), SCM_SOLUTE_ZS

    projector = SymmetryProjector()
    t0 = time.time()
    records: list[dict] = []

    for split in SPLITS:
        ds = SymmetricTiltGBDataset(
            GAMMA_CSV, BULK_MEMBERSHIP, MANIFEST_DIR,
            split_indices_json_path=str(indices_json), split=split,
            expected_size=EXPECT[split], dtype=torch.float64,
        )
        assert ds.dtype == torch.float64
        for i in range(len(ds)):
            row = ds.rows[i]
            rec = {
                "row_pos": ds.row_positions[i], "split": split, "cnid": row.cnid, "k": row.k, "seed": row.seed,
                "natoms_csv": row.natoms, "n": "", "m": "",
                "fixed_b_ok": "", "slave_ok": "", "fixed_b_nn_resid_A": "", "slave_nn_resid_A": "",
                "fixed_b_error": "", "slave_error": "",
                "getitem_ok": False, "proj_resid_max_A": "",
                "resid_fixed_a_A": "", "resid_fixed_b_A": "", "resid_slave_A": "", "resid_cell_A": "",
                "error_type": "", "error": "",
            }
            ctx = f" ({row.cnid} k={row.k} seed={row.seed})"

            # --- individual probes (FIXED_B / SLAVE are decided independently with the same functions and inputs as __getitem__) ---
            try:
                poscar_path = Path(resolve_view(str(Path(row.dir) / "POSCAR_relaxed")))
                frac, cell, species = read_poscar_frac_species(poscar_path)
                a_t, b_t, c_t = ds._template_axes(row.cnid)
                gb_dx_val = (float(cell[0, 0]) - a_t) / 2.0
                membership = ds.membership[(row.cnid, row.k, row.seed)]
                layout = build_canonical_layout(
                    frac=frac, cell=cell, membership=membership, manifest=None,
                    rng=np.random.default_rng(), species_frac=species,
                )
                fa_idx = layout.source_fixed_a_local_idx
                m_idx = layout.source_master_local_idx
                rec["m"] = int(len(m_idx))
                rec["n"] = int(len(fa_idx) + len(layout.source_fixed_b_local_idx) + len(m_idx)
                               + len(layout.source_slave_local_idx))
                geom_ok = True
            except Exception as e:  # noqa: BLE001 -- record and continue (not discarded)
                geom_ok = False
                msg = f"geometry/layout: {type(e).__name__}: {e}"
                rec["fixed_b_ok"] = rec["slave_ok"] = False
                rec["fixed_b_error"] = rec["slave_error"] = msg

            if geom_ok:
                try:
                    _, diag = _derive_fixed_b_correspondence(
                        frac=frac, cell=cell, fixed_a_order=fa_idx,
                        true_fixed_b_idx=layout.source_fixed_b_local_idx,
                        gb_t_yz=layout.effective_gb_t_yz, gb_dx=gb_dx_val,
                        a_template=a_t, b_template=b_t, c_template=c_t, context=ctx,
                    )
                    rec["fixed_b_ok"] = True
                    rec["fixed_b_nn_resid_A"] = diag["max_nn_residual_A"]
                except Exception as e:  # noqa: BLE001
                    rec["fixed_b_ok"] = False
                    rec["fixed_b_error"] = f"{type(e).__name__}: {e}"
                try:
                    _, diag = _derive_slave_correspondence(
                        frac=frac, cell=cell, master_order=m_idx,
                        true_slave_idx=layout.source_slave_local_idx,
                        gb_t_yz=layout.effective_gb_t_yz, context=ctx,
                    )
                    rec["slave_ok"] = True
                    rec["slave_nn_resid_A"] = diag["max_nn_residual_A"]
                except Exception as e:  # noqa: BLE001
                    rec["slave_ok"] = False
                    rec["slave_error"] = f"{type(e).__name__}: {e}"

            # --- production route: ds[i] and the reconstruction residual with the production projector ---
            try:
                g = ds[i]
                if g.pos.shape[0] != row.natoms:
                    raise AssertionError(f"pos rows {g.pos.shape[0]} != CSV natoms {row.natoms}")
                if g.pos.dtype != torch.float64:
                    raise AssertionError(f"graph dtype {g.pos.dtype} != float64")
                out = projector(collate([g]))
                cell_t = out.cell[0]
                dist = mic_cart_dist(out.pos, g.pos, cell_t)
                block = g.gb_block
                per_block = {}
                for bid, name in ((GB_BLOCK_FIXED_A, "fixed_a"), (GB_BLOCK_FIXED_B, "fixed_b"),
                                  (GB_BLOCK_MASTER, "master"), (GB_BLOCK_SLAVE, "slave")):
                    mask = block == bid
                    per_block[name] = float(dist[mask].max()) if bool(mask.any()) else float("nan")
                resid_cell = float((out.cell[0] - g.cell[0]).abs().max())
                rec["resid_fixed_a_A"] = per_block["fixed_a"]
                rec["resid_fixed_b_A"] = per_block["fixed_b"]
                rec["resid_slave_A"] = per_block["slave"]
                rec["resid_cell_A"] = resid_cell
                rec["proj_resid_max_A"] = max(per_block["fixed_a"], per_block["fixed_b"],
                                              per_block["master"], per_block["slave"], resid_cell)
                rec["getitem_ok"] = True
                if rec["n"] == "":
                    rec["n"] = int(g.pos.shape[0])
            except Exception as e:  # noqa: BLE001
                rec["getitem_ok"] = False
                rec["error_type"] = type(e).__name__
                rec["error"] = str(e).replace("\n", " ")[:2000]
            records.append(rec)
        print(f"split={split}: {len(ds)} rows done, elapsed {time.time() - t0:.1f} s", flush=True)

    # --- CSV ---
    cols = list(records[0].keys())
    with open(out_rows, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            r2 = dict(r)
            for k in ("fixed_b_error", "slave_error"):
                r2[k] = str(r2[k]).replace("\n", " ")[:2000]
            w.writerow(r2)

    # --- summary ---
    by_split = {}
    for split in SPLITS:
        rs = [r for r in records if r["split"] == split]
        failed = [r for r in rs if not r["getitem_ok"]]
        okr = [r for r in rs if r["getitem_ok"]]
        resid = [r["proj_resid_max_A"] for r in okr]
        by_split[split] = {
            "n_rows": len(rs),
            "n_failed": len(failed),
            "failed_by_exception_type": dict(collections.Counter(r["error_type"] for r in failed)),
            "n_fixed_b_probe_failed": sum(1 for r in rs if r["fixed_b_ok"] is False),
            "n_slave_probe_failed": sum(1 for r in rs if r["slave_ok"] is False),
            "proj_resid_max_A_over_successful_rows": max(resid) if resid else None,
            "n_successful_rows_above_1e-9": sum(1 for x in resid if x > CHECK2_THRESHOLD_A),
            "fixed_b_nn_resid_max_A": max([r["fixed_b_nn_resid_A"] for r in rs if r["fixed_b_nn_resid_A"] != ""] or [None]),
            "slave_nn_resid_max_A": max([r["slave_nn_resid_A"] for r in rs if r["slave_nn_resid_A"] != ""] or [None]),
            "n_distribution": dict(sorted(collections.Counter(r["n"] for r in rs).items(), key=lambda kv: str(kv[0]))),
            "m_distribution": dict(sorted(collections.Counter(r["m"] for r in rs).items(), key=lambda kv: str(kv[0]))),
            "n_ne_76_plus_2m": sum(1 for r in rs if r["n"] != "" and r["m"] != "" and r["n"] != 76 + 2 * r["m"]),
        }
    n_total = len(records)
    all_ok_resid = [r["proj_resid_max_A"] for r in records if r["getitem_ok"]]
    summary = {
        "design": "data rebuild stage E-7",
        "n_rows_total": n_total,
        "per_split_counts": {k: v["n_rows"] for k, v in by_split.items()},
        "n_failed_total": sum(v["n_failed"] for v in by_split.values()),
        "check_11_rows": {
            "expect": EXPECT,
            "n_rows_total_eq_expect_total": n_total == EXPECT_TOTAL,
            "split_counts_match": {k: by_split[k]["n_rows"] == EXPECT[k] for k in EXPECT},
            "note": "the number of failures is not required to be 0. Giving the count and the list of rows is the acceptance condition (the handling of failing rows is decided by the parent)",
        },
        "check_11_resid": {
            "threshold_A": CHECK2_THRESHOLD_A,
            "max_proj_resid_A_over_all_successful_rows": max(all_ok_resid) if all_ok_resid else None,
            "n_successful_rows": len(all_ok_resid),
            "n_successful_rows_above_threshold": sum(1 for x in all_ok_resid if x > CHECK2_THRESHOLD_A),
            "measured_with": "dataset dtype=torch.float64, device=cpu, torch.use_deterministic_algorithms(True), "
                             "production SymmetryProjector applied to the output of ds[i], minimum-image distance per atom (index agreement)",
        },
        "by_split": by_split,
        "failed_rows": [
            {k: r[k] for k in ("split", "row_pos", "cnid", "k", "seed", "fixed_b_ok", "slave_ok", "error_type", "error")}
            for r in records if not r["getitem_ok"]
        ],
        "successful_rows_above_threshold": [
            {k: r[k] for k in ("split", "row_pos", "cnid", "k", "seed", "proj_resid_max_A")}
            for r in records if r["getitem_ok"] and r["proj_resid_max_A"] > CHECK2_THRESHOLD_A
        ],
        "env": {
            "SCM_HOST_Z": SCM_HOST_Z, "SCM_SOLUTE_ZS": sorted(SCM_SOLUTE_ZS),
            "dtype": "torch.float64", "device": "cpu", "torch": torch.__version__, "numpy": np.__version__,
            "python": sys.version.split()[0],
            "dataset_py": str(Path(dsmod.__file__)), "dataset_py_sha256": sha256_of(Path(dsmod.__file__)),
            "split_json": str(indices_json), "split_json_sha256": sha256_of(indices_json),
            "gamma_csv": str(GAMMA_CSV), "gamma_csv_sha256": sha256_of(GAMMA_CSV),
            "bulk_membership_csv": str(BULK_MEMBERSHIP), "manifest_dir": str(MANIFEST_DIR),
            "code_git_head": subprocess.run(
                ["git", "-C", str(MN), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "host": os.uname().nodename,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out_rows}\nwrote {out_summary}")
    print(f"n_rows_total={n_total} n_failed_total={summary['n_failed_total']} "
          f"max_proj_resid_A={summary['check_11_resid']['max_proj_resid_A_over_all_successful_rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
