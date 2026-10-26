#!/usr/bin/env python3
"""Data rebuild, stage B (1 to 5): atom removal, bulk_membership and the input list of the relaxation.

  python stage_b_removal.py [--cnids cnid_0_0,...]

Precondition: stage A is done (results/template_bulk/{cnid}_{dx_tag}.json and structures/.../k0_s*/POSCAR).
  An exception if the template_bulk json of (cnid,dx) is absent (check 1 did not pass).

Implementation note (difference from the plan; stated in the final report):
  The plan says that duplicates are judged by "the set of info["removed_sites"] rearranged", but info["removed_sites"] of removal_builder
  is the number (int) of removed atoms and not a set. Therefore the atom positions before and after the removal are matched with the order preserved
  to obtain the set of indices of the removed atoms, and it is used for the duplicate judgment (removal_builder is not changed).

Exit code: 0 = passed / 2 = check 2 FAIL (the total shortfall > floor(0.01 * expected total). 65 for all 31 templates).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read as ase_read, write as ase_write

import builder_common as C
import dx_lib
import removal_builder as RB

BM_COLS = ["cnid", "k", "seed", "natoms", "n_bulkA", "n_bulkB", "n_frozen", "max_pos_diff_A", "status",
           "frozen_indices", "bulkA_indices", "bulkB_indices", "op_residual_A", "reason"]
_NUM = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def norm_reason(msg: str) -> str:
    return _NUM.sub("#", msg)


def removed_indices(atoms_old, atoms_new) -> list[int]:
    """atoms_new is atoms_old.copy() with atoms deleted (order preserved). The removed indices are found by exact agreement of positions."""
    P, Q = atoms_old.get_positions(), atoms_new.get_positions()
    j, out = 0, []
    for i in range(len(P)):
        if j < len(Q) and np.array_equal(P[i], Q[j]):
            j += 1
        else:
            out.append(i)
    if j != len(Q) or len(out) != len(P) - len(Q):
        raise RuntimeError("removed_indices: could not align atoms before/after removal")
    return out


def load_template_bulk(cnid: str, dx: float) -> dict:
    p = C.RESULTS / "template_bulk" / f"{cnid}_{C.dx_tag(dx)}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing (stage A / check 1 did not pass)")
    with open(p) as f:
        d = json.load(f)
    d["bulkA_frac"] = np.array(d["bulkA_frac"])
    d["bulkB_frac"] = np.array(d["bulkB_frac"])
    d["R"] = np.array(d["R"])
    d["t"] = np.array(d["t"])
    return d


def bm_row(cnid, k, seed, natoms, op_res):
    return {"cnid": cnid, "k": k, "seed": seed, "natoms": natoms, "n_bulkA": C.N_BULK, "n_bulkB": C.N_BULK,
            "n_frozen": 0, "max_pos_diff_A": "", "status": "", "frozen_indices": "", "bulkA_indices": "",
            "bulkB_indices": "", "op_residual_A": f"{op_res:.6e}", "reason": ""}


def frozen_check(poscar: Path, tb: dict, row: dict):
    """Stage B-3. If ok, returns frozen (ascending) and fills row. Otherwise returns None and sets skip and the reason in row."""
    try:
        loc = dx_lib.locate_frozen_indices_path(poscar, tb)
    except Exception as e:  # noqa: BLE001
        row["status"], row["reason"] = "skip", f"locate_error:{e}"
        return None
    if loc["natoms"] != row["natoms"]:
        row["status"], row["reason"] = "skip", f"natoms_mismatch(expected={row['natoms']},poscar={loc['natoms']})"
        return None
    row["max_pos_diff_A"] = f"{loc['max_pos_diff_A']:.6e}"
    if loc["duplicate"]:
        row["status"], row["reason"] = "skip", "duplicate_frozen_index_match"
        return None
    if not (loc["max_pos_diff_A"] < 1e-6):
        row["status"], row["reason"] = "skip", f"membership_residual_too_large({loc['max_pos_diff_A']:.3e}A)"
        return None
    matched = loc["matched_idx"]
    frozen = sorted(int(i) for i in matched)
    if len(frozen) != C.N_FROZEN:
        row["status"], row["reason"] = "skip", f"n_frozen={len(frozen)}"
        return None
    row.update({
        "n_frozen": len(frozen), "status": "ok",
        "frozen_indices": ";".join(str(i) for i in frozen),
        "bulkA_indices": ";".join(str(int(i)) for i in matched[:C.N_BULK]),
        "bulkB_indices": ";".join(str(int(i)) for i in matched[C.N_BULK:]),
        "reason": "",
    })
    return frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnids", default=os.environ.get("DATA_CNIDS", ""))
    args = ap.parse_args()
    t_start = time.time()

    all_cnids = [r["cnid"] for r in C.read_template_map()]
    sel = C.select_cnids(all_cnids, args.cnids)
    (C.RESULTS / "removal_log").mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "relax_inputs").mkdir(parents=True, exist_ok=True)

    bm_rows = []
    per_cdk = []           # number of successes and shortfall per (cnid,dx,k)
    skip_reasons = Counter()
    n_structures = 0
    n_frozen_fail = 0
    deficit_total = 0

    for cnid in sel:
        relax_inputs = {}
        for dx_idx, dx in enumerate(C.DX10):
            tag = C.dx_tag(dx)
            tb = load_template_bulk(cnid, dx)
            op_res = tb["op_residual_A"]
            seed0 = C.seed_of(dx_idx, 0, 0)
            poscar0 = C.struct_dir(cnid, dx, 0, seed0) / "POSCAR"
            atoms_dx = ase_read(str(poscar0), format="vasp")
            log = []
            entries = []   # (k, seed, poscar, natoms)

            # k=0: the structure of stage A as it is
            entries.append((0, seed0, poscar0, len(atoms_dx)))
            log.append({"cnid": cnid, "dx_A": dx, "dx_idx": dx_idx, "k": 0, "a": 0, "seed": seed0, "ok": True,
                        "reason": "", "removed_sites": [], "op_residual_A": op_res, "asbuilt_residual_A": None,
                        "natoms": len(atoms_dx)})
            per_cdk.append({"cnid": cnid, "dx_A": dx, "dx_idx": dx_idx, "k": 0, "success": 1, "deficit": 0})
            n_structures += 1

            for k in (1, 2, 3, 4):
                seen = set()
                n_ok = 0
                for a in range(C.MAX_A):
                    if n_ok >= C.K_SUCCESS[k]:
                        break
                    seed = C.seed_of(dx_idx, k, a)
                    rec = {"cnid": cnid, "dx_A": dx, "dx_idx": dx_idx, "k": k, "a": a, "seed": seed,
                           "ok": False, "reason": "", "removed_sites": None, "op_residual_A": op_res,
                           "asbuilt_residual_A": None, "natoms": None}
                    try:
                        new_atoms, info = RB.remove_symmetric_pairs(atoms_dx, k, seed, R=tb["R"], t=tb["t"])
                    except RB.RemovalSkip as e:
                        rec["reason"] = e.reason
                        skip_reasons[norm_reason(e.reason)] += 1
                        log.append(rec)
                        continue
                    removed = removed_indices(atoms_dx, new_atoms)
                    if len(removed) != 2 * k or len(new_atoms) != C.N_ATOMS - 2 * k:
                        raise RuntimeError(f"{cnid} {tag} k={k} a={a}: removed {len(removed)} atoms")
                    key = frozenset(removed)
                    if key in seen:
                        rec["reason"] = "duplicate_removal_set"
                        rec["removed_sites"] = sorted(removed)
                        skip_reasons["duplicate_removal_set"] += 1
                        log.append(rec)
                        continue
                    seen.add(key)
                    sd = C.struct_dir(cnid, dx, k, seed)
                    sd.mkdir(parents=True, exist_ok=True)
                    ase_write(str(sd / "POSCAR"), new_atoms, format="vasp")
                    rec.update({"ok": True, "removed_sites": sorted(removed), "natoms": len(new_atoms),
                                "asbuilt_residual_A": info["asbuilt_residual_A"]})
                    log.append(rec)
                    entries.append((k, seed, sd / "POSCAR", len(new_atoms)))
                    n_ok += 1
                    n_structures += 1
                deficit = C.K_SUCCESS[k] - n_ok
                deficit_total += deficit
                per_cdk.append({"cnid": cnid, "dx_A": dx, "dx_idx": dx_idx, "k": k, "success": n_ok,
                                "deficit": deficit})

            # stage B-2: record of the removal (a separate file per (cnid,dx))
            logp = C.RESULTS / "removal_log" / f"{cnid}_{tag}.jsonl"
            tmpl = logp.with_suffix(".jsonl.tmp")
            with open(tmpl, "w") as f:
                for r in log:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmpl, logp)

            # stage B-3/4: fixed indices and the row of bulk_membership
            for (k, seed, poscar, natoms) in entries:
                row = bm_row(cnid, k, seed, natoms, op_res)
                frozen = frozen_check(poscar, tb, row)
                bm_rows.append(row)
                if frozen is None:
                    n_frozen_fail += 1
                    continue
                key = f"{cnid}|{tag}|k{k}|s{seed}"
                relax_inputs[key] = {"poscar_path": str(poscar), "frozen_indices": frozen, "natoms": natoms,
                                     "cnid": cnid, "k": k, "seed": seed, "dx_A": dx, "dx_idx": dx_idx}
        out = C.RESULTS / "relax_inputs" / f"{cnid}.json"
        tmpo = out.with_suffix(".json.tmp")
        with open(tmpo, "w") as f:
            json.dump(relax_inputs, f)
        os.replace(tmpo, out)
        print(f"done {cnid} structures_total={n_structures} deficit_total={deficit_total} "
              f"frozen_fail={n_frozen_fail} elapsed={time.time() - t_start:.1f}s", flush=True)

    tmpc = C.RESULTS / "bulk_membership.csv.tmp"
    with open(tmpc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BM_COLS)
        w.writeheader()
        w.writerows(bm_rows)
    os.replace(tmpc, C.RESULTS / "bulk_membership.csv")

    expected = len(sel) * len(C.DX10) * C.STRUCT_PER_TEMPLATE_DX
    cap = int(np.floor(0.01 * expected))
    by_k_success, by_k_deficit = Counter(), Counter()
    for r in per_cdk:
        by_k_success[str(r["k"])] += r["success"]
        by_k_deficit[str(r["k"])] += r["deficit"]
    summary = {
        "cnids": sel, "expected_total": expected, "n_structures": n_structures, "deficit_total": deficit_total,
        "deficit_cap": cap, "n_frozen_check_failed": n_frozen_fail,
        "success_by_k": dict(by_k_success), "deficit_by_k": dict(by_k_deficit),
        "removal_skip_reason_counts": dict(skip_reasons),
        "per_cnid_dx_k": per_cdk, "elapsed_s": time.time() - t_start,
    }
    with open(C.RESULTS / "stage_b_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"C2: structures={n_structures} deficit={deficit_total} (cap {cap}) expected={expected} "
          f"frozen_fail={n_frozen_fail}", flush=True)
    print(f"success_by_k={dict(by_k_success)} deficit_by_k={dict(by_k_deficit)}", flush=True)
    print(f"RemovalSkip reasons={dict(skip_reasons)}", flush=True)
    raise SystemExit(2 if deficit_total > cap else 0)


if __name__ == "__main__":
    main()
