#!/usr/bin/env python3
"""aggregate_novelty_control.py -- aggregation and the decision function criterion_143 (standard library only).

Plan: see the experiment description.
  python aggregate_novelty_control.py [--out-dir DIR]   : checks the 20 files rows_base<kk>.csv + DONE_base<kk> and writes cells.csv, criterion.json and summary.md
  python aggregate_novelty_control.py --self-check         : calls the decision function with the 7 validation cases and exits non-zero unless the expected verdict is obtained
Standard library only. --self-check can be run with /usr/bin/python3 -I on the login node (pymatgen and ase are not used).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "novelty_control"

N_BASES = 20
ROWS_PER_BASE = 15
N_ROWS_CELL = 20        # 1 cell = 20 base structures x 1 run. The denominator is fixed to n_rows = 20 (not applied is counted as a mismatch)
PATH_H = "H"
PATH_V = "V"
TRANSFORMS_H = ["P0", "P1", "P1u", "P2", "P3", "P4", "P5", "P6", "P7", "N1"]
TRANSFORMS_V = ["P0", "P1", "P2", "P6", "P7"]
ALL_CELL_KEYS = [(PATH_H, t) for t in TRANSFORMS_H] + [(PATH_V, t) for t in TRANSFORMS_V]
# 14 positive cells, 11 core cells, 3 peripheral cells
POSITIVE_CELLS = [(PATH_H, t) for t in ["P0", "P1", "P1u", "P2", "P3", "P4", "P5", "P6", "P7"]] + [(PATH_V, t) for t in TRANSFORMS_V]
CORE_CELLS = [(PATH_H, t) for t in ["P0", "P1", "P1u", "P2", "P6", "P7"]] + [(PATH_V, t) for t in TRANSFORMS_V]
PERIPHERAL_CELLS = [(PATH_H, t) for t in ["P3", "P4", "P5"]]
NEG_CELL = (PATH_H, "N1")
ROW_COLS = ["base_id", "row_index", "natoms", "transform", "path", "applied", "skip_reason", "novel", "match_row_index",
            "match_is_source", "n_compared", "rms_to_source", "max_dist_to_source", "max_disp_applied_A", "elapsed_s"]
CELL_COLS = ["path", "transform", "n_rows", "n_applied", "n_matched", "n_novel"]
RMS_DEGEN_TOL = 1e-6
DEGEN_CELLS = [(PATH_H, t) for t in ["P0", "P1", "P1u", "P2", "P6"]] + [(PATH_V, t) for t in ["P0", "P1", "P2", "P6"]]


class AggregateError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# decision function (the order of decisions is single-sourced)
# ---------------------------------------------------------------------------
def _cell(cells, key):
    if key not in cells:
        raise AggregateError(f"control_check: required cell missing: {key}")
    c = cells[key]
    if int(c["n_rows"]) != N_ROWS_CELL:
        raise AggregateError(f"control_check: cell {key} n_rows={c['n_rows']} != {N_ROWS_CELL}")
    return c


def control_check(cells):
    """cells: {(path, transform): {"n_rows","n_applied","n_matched","n_novel"}}。
    Returns: {"C-2": "PASS"|"FAIL", "C-1": "PASS"|"FAIL"|"FLAG"|"HOLD", "reason": <the first rule that applied>, ...}.
    C-2: PASS if n_applied = 20 and n_novel = 20 for H x N1, otherwise FAIL.
    Order of decisions of C-1 (evaluated from the top, the first one that applies is taken):
      1. C-2 is FAIL -> HOLD
      2. n_matched < 20 in any core cell -> FAIL
      3. n_applied < 20 in any positive cell -> FLAG
      4. n_matched < 20 in any peripheral cell -> FLAG
      5. otherwise -> PASS"""
    neg = _cell(cells, NEG_CELL)
    g2 = "PASS" if (int(neg["n_applied"]) == N_ROWS_CELL and int(neg["n_novel"]) == N_ROWS_CELL) else "FAIL"
    pos = {k: _cell(cells, k) for k in POSITIVE_CELLS}
    core_short = [k for k in CORE_CELLS if int(pos[k]["n_matched"]) < N_ROWS_CELL]
    unapplied = [k for k in POSITIVE_CELLS if int(pos[k]["n_applied"]) < N_ROWS_CELL]
    periph_short = [k for k in PERIPHERAL_CELLS if int(pos[k]["n_matched"]) < N_ROWS_CELL]
    if g2 == "FAIL":
        g1, rule = "HOLD", "1: C2 FAIL"
    elif core_short:
        g1, rule = "FAIL", "2: core cell n_matched < 20"
    elif unapplied:
        g1, rule = "FLAG", "3: positive cell n_applied < 20"
    elif periph_short:
        g1, rule = "FLAG", "4: peripheral cell n_matched < 20"
    else:
        g1, rule = "PASS", "5: all 14 positive cells n_matched = 20"
    fmt = lambda ks: [f"{p}x{t}" for p, t in ks]  # noqa: E731
    return {"C2": g2, "C1": g1, "rule": rule,
            "core_cells_short": fmt(core_short), "unapplied_positive_cells": fmt(unapplied),
            "peripheral_cells_short": fmt(periph_short)}


# ---------------------------------------------------------------------------
# --self-check (7 validation cases. Only the synthetic cell table. Standard library only)
# ---------------------------------------------------------------------------
def _base_cells():
    cells = {}
    for k in POSITIVE_CELLS:
        cells[k] = {"n_rows": 20, "n_applied": 20, "n_matched": 20, "n_novel": 0}
    cells[NEG_CELL] = {"n_rows": 20, "n_applied": 20, "n_matched": 0, "n_novel": 20}
    return cells


def _w1():
    return _base_cells()


def _w2():
    c = _base_cells()
    c[(PATH_H, "P1")] = {"n_rows": 20, "n_applied": 20, "n_matched": 19, "n_novel": 1}
    return c


def _w3():
    c = _base_cells()
    c[(PATH_H, "P5")] = {"n_rows": 20, "n_applied": 20, "n_matched": 15, "n_novel": 5}
    return c


def _w4():
    c = _base_cells()
    c[NEG_CELL] = {"n_rows": 20, "n_applied": 20, "n_matched": 1, "n_novel": 19}
    return c


def _w5():
    c = _base_cells()
    c[(PATH_H, "P1")] = {"n_rows": 20, "n_applied": 20, "n_matched": 19, "n_novel": 1}
    c[(PATH_H, "P5")] = {"n_rows": 20, "n_applied": 20, "n_matched": 15, "n_novel": 5}
    return c


def _w6():
    c = _base_cells()
    c[(PATH_H, "P4")] = {"n_rows": 20, "n_applied": 0, "n_matched": 0, "n_novel": 0}
    return c


def _w7():
    c = _base_cells()
    c[(PATH_V, "P6")] = {"n_rows": 20, "n_applied": 18, "n_matched": 18, "n_novel": 0}
    return c


VALIDATIONES = [
    (1, "all positive cells match 20/20, N1 20/20 novel", _w1, "PASS", "PASS"),
    (2, "H x P1 19/20, others as in 1", _w2, "PASS", "FAIL"),
    (3, "H x P5 15/20, others as in 1", _w3, "PASS", "FLAG"),
    (4, "n_novel of N1 = 19, others as in 1", _w4, "FAIL", "HOLD"),
    (5, "H x P1 19/20 and H x P5 15/20 (confirmation of the order)", _w5, "PASS", "FAIL"),
    (6, "n_applied = 0 and n_matched = 0 of H x P4 (empty cell)", _w6, "PASS", "FLAG"),
    (7, "n_applied = 18 and n_matched = 18 of V x P6 (not applied in a core cell)", _w7, "PASS", "FAIL"),
]


def run_self_check() -> int:
    n_ok = 0
    for no, desc, build, exp2, exp1 in VALIDATIONES:
        res = control_check(build())
        ok = (res["C2"] == exp2 and res["C1"] == exp1)
        n_ok += int(ok)
        print(f"validation {no}: {desc}")
        print(f"  expected C2/C1 = {exp2}/{exp1}")
        print(f"  got      C2/C1 = {res['C2']}/{res['C1']}  (rule {res['rule']})  -> {'MATCH' if ok else 'MISMATCH'}")
    print(f"SELF_CHECK: {n_ok}/{len(VALIDATIONES)} validationes matched expected verdicts")
    if n_ok != len(VALIDATIONES):
        print("SELF_CHECK FAIL", file=sys.stderr)
        return 1
    print("SELF_CHECK PASS")
    return 0


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def load_rows(out_dir: Path):
    """All of the 20 files rows_base<kk>.csv and DONE_base<kk> exist, each has 15 rows and (path, transform) is exactly the expected 15 combinations once each."""
    all_rows = []
    problems = []
    for b in range(N_BASES):
        tag = f"{b:02d}"
        rf, df = out_dir / f"rows_base{tag}.csv", out_dir / f"DONE_base{tag}"
        if not rf.is_file():
            problems.append(f"missing {rf.name}")
            continue
        if not df.is_file():
            problems.append(f"missing {df.name}")
            continue
        with open(rf, newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != ROWS_PER_BASE:
            problems.append(f"{rf.name} has {len(rows)} rows (expected {ROWS_PER_BASE})")
            continue
        keys = sorted((r["path"], r["transform"]) for r in rows)
        if keys != sorted(ALL_CELL_KEYS):
            problems.append(f"{rf.name}: (path, transform) set differs from the expected 15")
            continue
        if any(int(r["base_id"]) != b for r in rows):
            problems.append(f"{rf.name}: base_id column != {b}")
            continue
        all_rows.extend(rows)
    if problems:
        raise AggregateError("input check failed (the acceptance condition is not satisfied. The decision is not made): " + "; ".join(problems))
    if len(all_rows) != N_BASES * ROWS_PER_BASE:
        raise AggregateError(f"total rows {len(all_rows)} != {N_BASES * ROWS_PER_BASE}")
    return all_rows


def build_cells(rows):
    cells = {k: {"n_rows": 0, "n_applied": 0, "n_matched": 0, "n_novel": 0} for k in ALL_CELL_KEYS}
    for r in rows:
        c = cells[(r["path"], r["transform"])]
        c["n_rows"] += 1
        if r["applied"] == "True":
            c["n_applied"] += 1
            if r["novel"] == "False":
                c["n_matched"] += 1
            elif r["novel"] == "True":
                c["n_novel"] += 1
            else:
                raise AggregateError(f"applied row with novel={r['novel']!r}: base {r['base_id']} {r['path']}x{r['transform']}")
    return cells


def write_outputs(out_dir: Path, rows, cells, gate):
    targets = [out_dir / n for n in ("cells.csv", "control_check.json", "summary.md")]
    for t in targets:
        if t.exists():
            raise AggregateError(f"{t} exists; not overwriting")
    with open(targets[0], "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CELL_COLS)
        for (p, t) in ALL_CELL_KEYS:
            c = cells[(p, t)]
            w.writerow([p, t, c["n_rows"], c["n_applied"], c["n_matched"], c["n_novel"]])

    # auxiliary aggregation (not used for decisions)
    def sel(path, tr):
        return [r for r in rows if r["path"] == path and r["transform"] == tr]

    p0h = sel(PATH_H, "P0")
    src_true = sum(1 for r in p0h if r["match_is_source"] == "True")
    rms_max = {PATH_H: 0.0, PATH_V: 0.0}
    for (p, t) in DEGEN_CELLS:
        for r in sel(p, t):
            if r["rms_to_source"] != "":
                rms_max[p] = max(rms_max[p], float(r["rms_to_source"]))
    nat = sorted({int(r["natoms"]) for r in rows})
    misses = []
    for r in rows:
        key = (r["path"], r["transform"])
        if key in POSITIVE_CELLS and not (r["applied"] == "True" and r["novel"] == "False"):
            misses.append(f"positive miss: base {r['base_id']} (natoms {r['natoms']}, train row {r['row_index']}) {r['path']}x{r['transform']} applied={r['applied']} novel={r['novel']} skip={r['skip_reason']}")
        if key == NEG_CELL and not (r["applied"] == "True" and r["novel"] == "True"):
            misses.append(f"negative miss: base {r['base_id']} (natoms {r['natoms']}, train row {r['row_index']}) N1 novel={r['novel']} match_row_index={r['match_row_index']}")
    gate_out = dict(gate)
    gate_out["cells"] = {f"{p}x{t}": cells[(p, t)] for (p, t) in ALL_CELL_KEYS}
    gate_out["aux"] = {
        "check-2_total_rows": len(rows),
        "check-3_H_P0_novel_false": cells[(PATH_H, "P0")]["n_matched"],
        "check-3_H_P0_match_is_source_true": src_true,
        "check-4_H_N1_novel_true": cells[NEG_CELL]["n_novel"],
        "check-5_V_applied": {t: cells[(PATH_V, t)]["n_applied"] for t in TRANSFORMS_V},
        "check-6_max_rms_to_source_P0_P1_P1u_P2_P6": rms_max,
        "misses": misses,
    }
    targets[1].write_text(json.dumps(gate_out, indent=2, ensure_ascii=False) + "\n")

    L = []
    L.append("# Aggregation of the positive and negative controls of the novelty instrument\n")
    L.append(f"- C-2 (discrimination power of the instrument): **{gate['C2']}**")
    L.append(f"- C-1 (detection of over-counting): **{gate['C1']}** (rule {gate['rule']})")
    L.append(f"- Shortfall in core cells: {gate['core_cells_short'] or 'none'} / unapplied positive cells: {gate['unapplied_positive_cells'] or 'none'} / shortfall in peripheral cells: {gate['peripheral_cells_short'] or 'none'}\n")
    L.append("## Cell table (the denominator is n_rows = 20)\n")
    L.append("| path | transform | n_rows | n_applied | n_matched | n_novel |")
    L.append("|---|---|---|---|---|---|")
    for (p, t) in ALL_CELL_KEYS:
        c = cells[(p, t)]
        L.append(f"| {p} | {t} | {c['n_rows']} | {c['n_applied']} | {c['n_matched']} | {c['n_novel']} |")
    L.append("\n## Number of matches per natoms (positive cells n_matched / negative N1 n_novel. 4 base structures for each natoms)\n")
    L.append("| path | transform | " + " | ".join(str(n) for n in nat) + " |")
    L.append("|---|---|" + "---|" * len(nat))
    for (p, t) in ALL_CELL_KEYS:
        parts = []
        for n in nat:
            rr = [r for r in sel(p, t) if int(r["natoms"]) == n]
            if (p, t) == NEG_CELL:
                v = sum(1 for r in rr if r["applied"] == "True" and r["novel"] == "True")
            else:
                v = sum(1 for r in rr if r["applied"] == "True" and r["novel"] == "False")
            parts.append(f"{v}/{len(rr)}")
        L.append(f"| {p} | {t} | " + " | ".join(parts) + " |")
    L.append("\n## Auxiliary values of the acceptance conditions\n")
    L.append(f"- check D-2: total number of rows {len(rows)} (expected 300)")
    L.append(f"- check-3: H x P0 novel=False {cells[(PATH_H, 'P0')]['n_matched']}/20、match_is_source=True {src_true}/20")
    L.append(f"- check-4: H x N1 novel=True {cells[NEG_CELL]['n_novel']}/20")
    L.append("- check D-5: applied of V: " + ", ".join(f"{t}={cells[(PATH_V, t)]['n_applied']}" for t in TRANSFORMS_V) + " (expected 20 each)")
    L.append(f"- check D-6: maximum of rms_to_source for P0/P1/P1u/P2/P6 (return value of get_rms_dist = normalized by (V/N)^(1/3)): H={rms_max[PATH_H]:.3e}, V={rms_max[PATH_V]:.3e} (upper limit {RMS_DEGEN_TOL:g})")
    L.append("\n## Misses (mismatch or not applied for positive, match for negative)\n")
    L.extend(f"- {m}" for m in misses) if misses else L.append("- none")
    targets[2].write_text("\n".join(L) + "\n")
    return targets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self_check", action="store_true")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()
    if args.self_check:
        return run_self_check()
    out_dir = Path(args.out_dir)
    try:
        rows = load_rows(out_dir)
        cells = build_cells(rows)
        gate = control_check(cells)
        targets = write_outputs(out_dir, rows, cells, gate)
    except AggregateError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    print(f"C2={gate['C2']} C1={gate['C1']} rule={gate['rule']}")
    print("wrote: " + " ".join(str(t) for t in targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
