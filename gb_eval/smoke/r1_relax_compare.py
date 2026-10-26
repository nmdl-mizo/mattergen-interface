#!/usr/bin/env python3
"""R1 smoke test (evaluation chain).

Relaxes a few generated structures of the proposed (symmetry-constrained) model with MACE-MH-1 (head omat_pbe) using the same
functions as the production relaxation script (gb_eval/experiments/relax_novelty/scripts/relax_chunk.py), and
compares the relaxed gamma with the published value in `$GBREL_DATA/relaxation/main/proposed_w2.0/chunk<NN>/relax_rows.csv`.
It only prints and writes the differences; it does not decide pass/fail.

Environment (no defaults; a missing variable raises KeyError): GBREL_DATA, GBREL_CODE, GBREL_WORK, MACE_MODEL_PATH.
Run `python gb_eval/make_compat_tree.py` first. Needs a GPU (MLIP is never run on CPU here).

usage: python gb_eval/smoke/r1_relax_compare.py --chunk 2 --indices 0,1,2 --out <output directory>
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import paths

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, required=True, help="chunk of the w=2.0 main generation (target_id = chunk // 2)")
ap.add_argument("--indices", required=True, help="comma separated cif_index values within the chunk")
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, str(paths.CODE_ROOT / "gb_eval/experiments/relax_novelty/scripts"))
import relax_chunk as R  # noqa: E402

K = R.K
idx = [int(x) for x in args.indices.split(",")]
chunk = args.chunk
pub_csv = paths.DATA_ROOT / "relaxation/main/proposed_w2.0" / f"chunk{chunk:02d}" / "relax_rows.csv"
pub = {int(r["cif_index"]): r for r in K.read_csv_rows(pub_csv)}

cfg = R.get_cfg()
calc = R.make_calc("mace", False, 10.0)
atoms_list = K.load_chunk_atoms("main", "proposed", 2.0, chunk)
fixed = list(range(K.N_FIXED_PROPOSED))

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
rows = []
for i in idx:
    atoms = atoms_list[i][0]
    t0 = time.time()
    res, _relaxed = R.relax_one(atoms, fixed, calc, cfg, proposed_check_idx=fixed)
    p = pub[i]
    g_new = res["gamma_relaxed"]
    g_pub = float(p["gamma_relaxed"])
    rows.append({
        "proposed_w2.0_chunk": chunk, "cif_index": i, "target_id": p["target_id"], "gamma_target": p["gamma_target"],
        "gamma_relaxed_published": repr(g_pub), "gamma_relaxed_rerun": repr(g_new), "abs_diff_J_m2": repr(abs(g_new - g_pub)),
        "status_published": p["status"], "converged_rerun": res["converged"], "n_steps_published": p["n_steps"],
        "n_steps_rerun": res["n_steps"], "fmax_final_rerun": repr(res["fmax_final"]), "max_disp_fixed_A_rerun": res["max_disp_fixed_A"],
        "elapsed_s": f"{time.time() - t0:.1f}",
    })
    print(rows[-1], flush=True)
with open(out / "compare.tsv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
    wr.writeheader()
    wr.writerows(rows)
print("wrote", out / "compare.tsv")
