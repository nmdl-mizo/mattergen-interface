#!/usr/bin/env python3
"""dft_gamma.py -- aggregation of the DFT stage: computes gamma_DFT from 39 VASP OUTCAR files (26 grain boundaries + 13 mu) and makes the input of the sign test (check C2). Standard library only.

Plan: see the experiment description.
The decision formula is not reimplemented: criteria.criterion_2_from_pairs (internally criterion_2) is imported and used.
Input : VRUN/results/dft_selection/selection.json (13 pairs, 26 structures; gamma_relaxed = gamma_MACE)
       VRUN/dft_inputs/index.json (correspondence table written by dft_write_inputs.py: name, natoms=N, area_A2=A, mu_name. A uses the output of the same script's
       "in-plane t x p area = |cross(cell[t], cell[p])|" as it is, and this script does not set another definition. Recomputation from the POSCAR is only a consistency check)
       VRUN/dft_results/<name>/OUTCAR (replaceable with --results-dir)
gamma_DFT,i = (E_TOTEN,i - N_i*mu_i)/(2*A_i)*16.0218 [J/m^2]. mu_i = E_TOTEN(mu cell)/120. For proposed one common mu (mu_proposed_common), for baseline one mu per structure, one to one.
Convergence = (number of final electronic steps < NELM=200) and (OUTCAR has "aborting loop because EDIFF is reached") and a TOTEN line exists. A missing OUTCAR or a truncated one (no TOTEN) is unconverged.
Exit codes: 0 = all 39 converged (PASS/FAIL/FLAG of the decision is in verdict.json). 2 = one or more unconverged or missing OUTCAR (the output is written. The check becomes FLAG depending on n_used). 1 = inconsistency of the inputs (stops immediately).
By default the output does not overwrite existing files (overwrite with --overwrite).
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import criteria as G  # noqa: E402

VRUN = HERE.parent
DESIGN = _gbrel.WORK + '/an internal design note that is not included'
EV_PER_A2_TO_J_PER_M2 = 16.0218     # quantity of the decision
NELM = 200                          # NELM of INCAR (identical to the reference calculation)
N_MU = 120                          # number of atoms of the mu cell
EXPECT_PAIRS = {"proposed": 7, "baseline": 6}
EXPECT_STRUCTURES = 26

RE_TOTEN = re.compile(r"free  energy   TOTEN\s*=\s*([-+0-9.EeDd]+)\s+eV")
RE_ITER = re.compile(r"Iteration\s+\d+\(\s*(\d+)\)")
RE_NIONS = re.compile(r"NIONS\s*=\s*(\d+)")
EDIFF_LINE = "aborting loop because EDIFF is reached"


class InputError(Exception):
    """Inconsistency of the inputs. exit 1."""


# ---------------------------------------------------------------------------
# reading of OUTCAR
# ---------------------------------------------------------------------------
def parse_outcar_text(text):
    toten = None
    for m in RE_TOTEN.finditer(text):
        toten = float(m.group(1))
    steps = None
    for m in RE_ITER.finditer(text):
        steps = int(m.group(1))
    ediff = EDIFF_LINE in text
    nm = RE_NIONS.search(text)
    nions = int(nm.group(1)) if nm else None
    if toten is None:
        status = "no_toten(truncated)"
    elif steps is None:
        status = "no_iteration_line"
    elif steps >= NELM:
        status = "nelm_reached"
    elif not ediff:
        status = "no_ediff_line"
    else:
        status = "ok"
    return {"status": status, "e_toten": toten, "n_elec_steps": steps, "ediff_reached": ediff, "nions": nions, "converged": status == "ok"}


def parse_outcar(path):
    p = Path(path)
    if not p.is_file():
        return {"status": "missing", "e_toten": None, "n_elec_steps": None, "ediff_reached": False, "nions": None, "converged": False}
    return parse_outcar_text(p.read_text(encoding="ascii", errors="replace"))


def gamma_j_m2(e_toten, n, mu, area):
    """gamma_DFT = (E_TOTEN - N*mu)/(2A) x 16.0218 [J/m^2] (quantity of the decision)."""
    return (e_toten - n * mu) / (2.0 * area) * EV_PER_A2_TO_J_PER_M2


# ---------------------------------------------------------------------------
# statistics for reporting only
# ---------------------------------------------------------------------------
def _avg_ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def report_only(rows):
    """rows: dict of structures for which gamma_dft was obtained. Spearman rho for all structures and per arm, median offset per arm, MAE after offset removal."""
    out = {"spearman_rho": {}, "offset_median_dft_minus_mace_J_m2": {}, "offset_removed_mae_J_m2": {}, "n_structures": {}}
    groups = {"all": rows, "proposed": [r for r in rows if r["arm"] == "proposed"], "baseline": [r for r in rows if r["arm"] == "baseline"]}
    for g, rs in groups.items():
        out["n_structures"][g] = len(rs)
        out["spearman_rho"][g] = spearman([r["gamma_dft"] for r in rs], [r["gamma_mace"] for r in rs])
        if g == "all":
            continue
        d = [r["gamma_dft"] - r["gamma_mace"] for r in rs]
        med = median(d)
        out["offset_median_dft_minus_mace_J_m2"][g] = med
        out["offset_removed_mae_J_m2"][g] = (sum(abs(x - med) for x in d) / len(d)) if d else None
    return out


# ---------------------------------------------------------------------------
# reading and checking of the inputs
# ---------------------------------------------------------------------------
def load_inputs(sel, index):
    """Cross-checks selection.json and dft_inputs/index.json and returns the list of structure rows and mu rows. InputError if different from the expectation."""
    pairs = sel["pairs"]
    cnt = {}
    for pr in pairs:
        cnt[pr["arm"]] = cnt.get(pr["arm"], 0) + 1
    n_struct = sum(2 for _ in pairs)
    if cnt != EXPECT_PAIRS or n_struct != EXPECT_STRUCTURES:
        raise InputError(f"selection pairs per arm {cnt} / structures {n_struct} != expected {EXPECT_PAIRS} / {EXPECT_STRUCTURES}")
    by_key = {}
    for st in index["structures"]:
        k = (st["arm"], st["target_id"], st["role"])
        if k in by_key:
            raise InputError(f"index.json duplicate structure key {k}")
        by_key[k] = st
    mus = {m["name"]: m for m in index["mu"]}
    structs = []
    for pr in pairs:
        pid = f"{pr['arm']}_t{pr['target_id']}"
        for role in ("lo", "hi"):
            s = pr[role]
            k = (s["arm"], s["target_id"], role)
            if k not in by_key:
                raise InputError(f"index.json lacks structure {k}")
            st = by_key[k]
            if (st["chunk"], st["cif_index"], st["natoms"]) != (s["chunk"], s["cif_index"], s["natoms"]):
                raise InputError(f"selection/index mismatch for {st['name']}: ({st['chunk']},{st['cif_index']},{st['natoms']}) vs ({s['chunk']},{s['cif_index']},{s['natoms']})")
            if abs(st["gamma_mace_J_m2"] - s["gamma_relaxed"]) > 1e-12:
                raise InputError(f"gamma_mace mismatch for {st['name']}")
            if st["mu_name"] not in mus:
                raise InputError(f"mu '{st['mu_name']}' of {st['name']} not in index.json mu list")
            structs.append({"pair_id": pid, "arm": s["arm"], "target": s["target_id"], "role": role, "chunk": s["chunk"], "cif_index": s["cif_index"],
                            "name": st["name"], "N": st["natoms"], "A": st["area_A2"], "mu_name": st["mu_name"], "gamma_mace": s["gamma_relaxed"]})
    if len(by_key) != EXPECT_STRUCTURES:
        raise InputError(f"index.json has {len(by_key)} structures != {EXPECT_STRUCTURES}")
    used_mu = {s["mu_name"] for s in structs}
    mu_rows = []
    for name in sorted(mus):
        m = mus[name]
        if m["natoms"] != N_MU:
            raise InputError(f"mu cell {name} natoms {m['natoms']} != {N_MU}")
        mu_rows.append({"name": name, "arm": m["arm"], "N": m["natoms"]})
    if {m["name"] for m in mu_rows} != used_mu:
        raise InputError("mu set in index.json != mu names referenced by structures")
    proposed_mu = {s["mu_name"] for s in structs if s["arm"] == "proposed"}
    if proposed_mu != {"mu_proposed_common"}:
        raise InputError(f"proposed mu names {proposed_mu} != {{'mu_proposed_common'}}")
    return structs, mu_rows


def check_area_against_poscar(structs, inputs_dir):
    """Consistency check only: whether A of index.json agrees with |cross(cell[t],cell[p])| of the POSCAR (the axes are the axes string of the index). InputError if not."""
    ax_index = {}
    for st in json.load(open(Path(inputs_dir) / "index.json"))["structures"]:
        ax_index[st["name"]] = st["axes"]
    bad = []
    for s in structs:
        poscar = Path(inputs_dir) / s["name"] / "POSCAR"
        if not poscar.is_file():
            continue
        lines = poscar.read_text(encoding="ascii").splitlines()
        sc = float(lines[1].strip())
        cell = [[float(x) * sc for x in lines[2 + i].split()] for i in range(3)]
        m = re.fullmatch(r"n=([abc]),t=([abc]),p=([abc])", ax_index[s["name"]])
        t, p = "abc".index(m.group(2)), "abc".index(m.group(3))
        u, v = cell[t], cell[p]
        cr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        a = math.sqrt(sum(c * c for c in cr))
        if abs(a - s["A"]) > 1e-6:
            bad.append((s["name"], s["A"], a))
        nat = int(lines[6].split()[0])
        if nat != s["N"]:
            bad.append((s["name"], "N", s["N"], nat))
    if bad:
        raise InputError(f"index.json A/N inconsistent with POSCAR: {bad[:3]}")


# ---------------------------------------------------------------------------
# main body of the aggregation (from the OUTCAR reading results calc = {name: parse dict})
# ---------------------------------------------------------------------------
def compute(structs, mu_rows, calc):
    # (1) μ
    mu_info = {}
    for m in mu_rows:
        c = calc[m["name"]]
        mu = c["e_toten"] / m["N"] if (c["converged"] and c["e_toten"] is not None) else None
        mu_info[m["name"]] = {**m, **c, "mu": mu}
    # (2) per structure
    rows = []
    for s in structs:
        c = calc[s["name"]]
        mi = mu_info[s["mu_name"]]
        mu = mi["mu"]
        gd = gamma_j_m2(c["e_toten"], s["N"], mu, s["A"]) if (c["converged"] and mu is not None) else None
        rows.append({**s, "e_toten": c["e_toten"], "n_elec_steps": c["n_elec_steps"], "ediff_reached": c["ediff_reached"], "converged": c["converged"],
                     "outcar_status": c["status"], "mu": mu, "mu_converged": mi["converged"], "gamma_dft": gd})
    by_name = {r["name"]: r for r in rows}
    # (3) pairs
    pair_ids = []
    for r in rows:
        if r["pair_id"] not in pair_ids:
            pair_ids.append(r["pair_id"])
    pair_table, gate_pairs = [], []
    for pid in pair_ids:
        lo = by_name[f"gb_{pid}_lo"]
        hi = by_name[f"gb_{pid}_hi"]
        reasons = []
        for r in (lo, hi):
            if not r["converged"]:
                reasons.append(f"{r['name']} unconverged ({r['outcar_status']})")
            if not r["mu_converged"]:
                reasons.append(f"{r['mu_name']} unconverged ({mu_info[r['mu_name']]['status']})")
        dm = hi["gamma_mace"] - lo["gamma_mace"]
        ent = {"pair_id": pid, "arm": lo["arm"], "target": lo["target"], "lo": lo["name"], "hi": hi["name"],
               "gamma_mace_lo": lo["gamma_mace"], "gamma_mace_hi": hi["gamma_mace"], "delta_gamma_mace": dm,
               "gamma_dft_lo": lo["gamma_dft"], "gamma_dft_hi": hi["gamma_dft"], "delta_gamma_dft": None, "agree": None}
        if reasons:
            ent["status"] = "excluded_unconverged"
            ent["reasons"] = sorted(set(reasons))
        else:
            dd = hi["gamma_dft"] - lo["gamma_dft"]
            ent["delta_gamma_dft"] = dd
            if abs(dm) < G.G2_DELTA_MIN:
                ent["status"] = "excluded_small_delta_mace"
                ent["reasons"] = [f"|Δγ_MACE|={abs(dm):.4f} < {G.G2_DELTA_MIN}"]
            else:
                ent["status"] = "used"
                ent["agree"] = bool((dm > 0) == (dd > 0) and dd != 0)
            gate_pairs.append({"pair_id": pid, "gamma_mace_lo": lo["gamma_mace"], "gamma_mace_hi": hi["gamma_mace"],
                               "gamma_dft_lo": lo["gamma_dft"], "gamma_dft_hi": hi["gamma_dft"]})
        pair_table.append(ent)
    # (4) decision (the return value of the function itself)
    gate_out = G.criterion_2_from_pairs(gate_pairs)
    # internal consistency of the table and the return value of the function (a discrepancy is a bug on the aggregation side)
    k_tab = sum(1 for e in pair_table if e["status"] == "used" and e["agree"])
    n_tab = sum(1 for e in pair_table if e["status"] == "used")
    if (gate_out["n_used"], gate_out["k"]) != (n_tab, k_tab):
        raise RuntimeError(f"table/gate mismatch: gate (n_used,k)=({gate_out['n_used']},{gate_out['k']}) vs table ({n_tab},{k_tab})")
    rep = report_only([r for r in rows if r["gamma_dft"] is not None])
    unconv = [{"name": n, "kind": "mu" if n in mu_info else "gb", "status": calc[n]["status"]} for n in calc if not calc[n]["converged"]]
    return {"rows": rows, "mu_info": mu_info, "pair_table": pair_table, "gate_out": gate_out, "report_only": rep, "unconverged": unconv,
            "n_pairs_excluded_unconverged": sum(1 for e in pair_table if e["status"] == "excluded_unconverged")}


CSV_COLS = ["arm", "target", "pair_id", "role", "chunk", "cif_index", "name", "N", "A", "E_toten", "n_elec_steps", "ediff_reached", "converged", "mu_name", "mu", "gamma_dft", "gamma_mace"]


def write_outputs(res, out_dir, results_dir, selection_path, overwrite):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f_csv, f_json = out / "gamma_vasp.csv", out / "verdict.json"
    for p in (f_csv, f_json):
        if p.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {p} (use --overwrite)")

    def fmt(v):
        return "" if v is None else repr(v) if isinstance(v, float) else v
    with open(f_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in res["rows"]:
            w.writerow([fmt(x) for x in (r["arm"], r["target"], r["pair_id"], r["role"], r["chunk"], r["cif_index"], r["name"], r["N"], r["A"], r["e_toten"],
                                          r["n_elec_steps"], r["ediff_reached"], r["converged"], r["mu_name"], r["mu"], r["gamma_dft"], r["gamma_mace"])])
    doc = {"design": DESIGN, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "script": Path(__file__).name, "selection": str(selection_path),
           "results_dir": str(results_dir), "results_dir_exists": Path(results_dir).is_dir(),
           "n_calcs": len(res["mu_info"]) + len(res["rows"]), "n_unconverged": len(res["unconverged"]), "unconverged": res["unconverged"],
           "mu_calcs": list(res["mu_info"].values()),
           "pairs": res["pair_table"], "n_pairs": len(res["pair_table"]),
           "excluded_pairs": [{"pair_id": e["pair_id"], "status": e["status"], "reasons": e["reasons"]} for e in res["pair_table"] if e["status"] != "used"],
           "n_used": res["gate_out"]["n_used"], "k": res["gate_out"]["k"],
           "gate_function_output": res["gate_out"], "report_only": res["report_only"]}
    with open(f_json, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return f_csv, f_json


def summary_text(res, results_dir):
    g = res["gate_out"]
    n_calc = len(res["mu_info"]) + len(res["rows"])
    L = [f"[dft_gamma] results_dir={results_dir} (exists={Path(results_dir).is_dir()})"]
    nu = len(res["unconverged"])
    if nu == n_calc:
        L.append(f"!! OUTCAR missing or all unconverged: {nu}/{n_calc} unconverged (0 converged). The test is impossible (n_used={g['n_used']}) -> C2 is {g['verdict']}. It is not PASS.")
    elif nu:
        L.append(f"!! Unconverged {nu}/{n_calc}: " + ", ".join(f"{u['name']}[{u['status']}]" for u in res["unconverged"]))
    else:
        L.append(f"All {n_calc} converged")
    L.append(f"Pairs {len(res['pair_table'])}: excluded because of unconverged {res['n_pairs_excluded_unconverged']}, excluded because of |d gamma_MACE|<{G.G2_DELTA_MIN} {g['n_excluded_small_delta']}")
    L.append(f"C2: n_used={g['n_used']} k={g['k']} p={g['p']} verdict={g['verdict']} ({g['reason']})")
    for e in res["pair_table"]:
        if e["status"] == "used":
            L.append(f"  {e['pair_id']:<12} d gamma_MACE={e['delta_gamma_mace']:+.3f} d gamma_DFT={e['delta_gamma_dft']:+.3f} {'agree' if e['agree'] else 'disagree'}")
        else:
            L.append(f"  {e['pair_id']:<12} excluded: {'; '.join(e['reasons'][:2])}")
    ro = res["report_only"]
    L.append(f"Report only: Spearman rho={ro['spearman_rho']} / median offset={ro['offset_median_dft_minus_mace_J_m2']} / MAE after removal={ro['offset_removed_mae_J_m2']}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", default=str(VRUN / "results/dft/selection.json"))
    ap.add_argument("--inputs-dir", default=str(VRUN / "dft_inputs"), help="location of index.json and <name>/POSCAR")
    ap.add_argument("--results-dir", default=str(VRUN / "dft_results"), help="location of <name>/OUTCAR")
    ap.add_argument("--out-dir", default=str(VRUN / "results/dft"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    try:
        sel = json.load(open(args.selection))
        index = json.load(open(Path(args.inputs_dir) / "index.json"))
        structs, mu_rows = load_inputs(sel, index)
        check_area_against_poscar(structs, args.inputs_dir)
        calc = {}
        for name in [s["name"] for s in structs] + [m["name"] for m in mu_rows]:
            c = parse_outcar(Path(args.results_dir) / name / "OUTCAR")
            n_exp = N_MU if name.startswith("mu_") else next(s["N"] for s in structs if s["name"] == name)
            if c["nions"] is not None and c["nions"] != n_exp:
                raise InputError(f"{name}: OUTCAR NIONS={c['nions']} != expected N={n_exp}")
            calc[name] = c
    except InputError as e:
        print(f"FATAL(C1): {e}", file=sys.stderr)
        return 1
    res = compute(structs, mu_rows, calc)
    try:
        f_csv, f_json = write_outputs(res, args.out_dir, args.results_dir, args.selection, args.overwrite)
    except FileExistsError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
    print(summary_text(res, args.results_dir))
    print(f"wrote {f_csv}\nwrote {f_json}")
    return 0 if not res["unconverged"] else 2


if __name__ == "__main__":
    sys.exit(main())
