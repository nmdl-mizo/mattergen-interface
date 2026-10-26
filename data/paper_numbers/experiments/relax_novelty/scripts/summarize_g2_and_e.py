#!/usr/bin/env python3
"""summarize_g2_and_e.py -- aggregates, from the CSV of gb_symfit, the counts of two grain boundaries (G2) and equivalence (E) per (arm,w,target). Standard library only.

Plan: see the experiment description.
Decision formulas (identical to the frozen ones; do not change):
  G2_stage = status ∈ {fitted,no_gb,multi_gb,disordered} ∧ gb0_status == "two_gb"   (aggregate_structure_filtered.py:8,87)
  E_stage  = G2_stage AND gb_symfit_max_A <= THETA_A,  THETA_A = 0.064 A (frozen; do not change)
The denominator is fixed to the requested number n_req (number of manifest rows per (arm,w)). Exclusions and errors are counted as misses. No conditional rates and no rates summed over w are given.

Usage:
  python summarize_g2_and_e.py --layer {main,additional} --csv PATH --manifest PATH --out-prefix PATH [--check-known-main]
  python summarize_g2_and_e.py --self_check      (synthetic data; no file input/output; 5 cases)
Output: <out-prefix>.md and <out-prefix>.json (if they exist, FileExistsError -> exit 2).
Exit codes: 0 = success / 2 = failure of inputs or checks / 3 = FAIL of the known-value check (--check-known-main) or of the self_check
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

DESIGN = '<ORIGINAL_ROOT>/an internal design note that is not included'
THETA_A = 0.064      # frozen (same value as in aggregate_structure_filtered.py:33). Do not change
Z95 = 1.959964       # aggregation of the additional layer
STAGES = ("post", "pre")     # post is the main one, pre is shown for reference
GB0_STATUS_OK = ("fitted", "no_gb", "multi_gb", "disordered")
STATUS_ORDER = ["fitted", "no_gb", "multi_gb", "disordered", "axis_mismatch", "non_al", "excluded_pre", "error"]
MANIFEST_REQUIRED = ["layer", "arm", "w", "chunk", "cif_index", "target_id", "gamma_target"]
CSV_REQUIRED = ["layer", "arm", "w", "chunk", "cif_index", "stage", "status", "gb0_status", "gb_symfit_max_A"]
ARM_ORDER = {"proposed": 0, "baseline": 1}


class InputError(RuntimeError):
    """Failure of the inputs or the consistency checks (exit 2)."""


# ---------------------------------------------------------------------------
# decision formulas
# ---------------------------------------------------------------------------
def norm_w(w) -> str:
    return f"{float(w):.1f}"


def to_finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def classify(status, gb0_status, max_a):
    """(G2, E). max_a is a finite float or None."""
    g2 = status in GB0_STATUS_OK and gb0_status == "two_gb"
    e = g2 and max_a is not None and max_a <= THETA_A
    return g2, e


def wilson(k, n, z=Z95):
    if n <= 0:
        return None, None
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / den
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return min(max(0.0, c - h), p), max(min(1.0, c + h), p)


def _wil(k, n):
    lo, hi = wilson(k, n)
    return {"p": (k / n if n > 0 else None), "lo": lo, "hi": hi}


# ---------------------------------------------------------------------------
# join, consistency checks and aggregation (no file input/output; the self_check also goes through this)
# ---------------------------------------------------------------------------
def _need_cols(rows, required, label):
    if not rows:
        raise InputError(f"{label} is empty")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise InputError(f"{label}: missing columns {missing}")


def check_rows(csv_rows):
    """Same as F1 of aggregate_structure_filtered.py:128-135. InputError if violated."""
    bad_status = sorted({r["status"] for r in csv_rows if r["status"] not in STATUS_ORDER})
    if bad_status:
        raise InputError(f"unknown status value(s): {bad_status}")
    n_a = sum(1 for r in csv_rows if r["gb0_status"] == "two_gb" and r["status"] != "fitted")
    if n_a:
        raise InputError(f"integrity (a): {n_a} rows have gb0_status == two_gb but status != fitted")
    n_b = sum(1 for r in csv_rows if r["gb0_status"] == "two_gb" and to_finite(r["gb_symfit_max_A"]) is None)
    if n_b:
        raise InputError(f"integrity (b): {n_b} two_gb rows have no finite gb_symfit_max_A")


def build_records(layer, manifest_rows, csv_rows):
    """Joins the pre and post rows of the CSV to the (arm,w,chunk,cif_index) of the manifest. Rows that cannot be joined, surplus rows and duplicates are InputError."""
    _need_cols(manifest_rows, MANIFEST_REQUIRED, "manifest")
    _need_cols(csv_rows, CSV_REQUIRED, "csv")
    bad_m = sorted({r["layer"] for r in manifest_rows if r["layer"] != layer})
    if bad_m:
        raise InputError(f"manifest layer column mismatch: expected {layer!r}, found {bad_m}")
    bad_c = sorted({r["layer"] for r in csv_rows if r["layer"] != layer})
    if bad_c:
        raise InputError(f"csv layer column mismatch: expected {layer!r}, found {bad_c}")
    bad_stage = sorted({r["stage"] for r in csv_rows if r["stage"] not in STAGES})
    if bad_stage:
        raise InputError(f"unknown stage value(s): {bad_stage}")
    check_rows(csv_rows)

    def key(r):
        return (r["arm"], norm_w(r["w"]), int(r["chunk"]), int(r["cif_index"]))

    man = {}
    for r in manifest_rows:
        k = key(r)
        if k in man:
            raise InputError(f"manifest: duplicated key {k}")
        man[k] = r
    table = {}
    for r in csv_rows:
        k = key(r) + (r["stage"],)
        if k in table:
            raise InputError(f"csv: duplicated key {k}")
        table[k] = r
    expected = {k + (st,) for k in man for st in STAGES}
    unmatched_man = sorted(expected - set(table))
    extra_csv = sorted(set(table) - expected)
    if unmatched_man or extra_csv:
        raise InputError(f"join failed: manifest keys without csv row {len(unmatched_man)} (e.g. {unmatched_man[:3]}) / csv rows not in manifest {len(extra_csv)} (e.g. {extra_csv[:3]})")
    records = []
    for k, m in man.items():
        for st in STAGES:
            c = table[k + (st,)]
            max_a = to_finite(c["gb_symfit_max_A"])
            g2, e = classify(c["status"], c["gb0_status"], max_a)
            records.append({"arm": k[0], "w": k[1], "stage": st, "target_id": int(m["target_id"]), "gamma_target": m["gamma_target"],
                            "status": c["status"], "gb0_status": c["gb0_status"], "max_A": max_a, "g2": g2, "e": e})
    return records


def aggregate(layer, records):
    """The denominator is fixed to the number of manifest rows n_req (misses include exclusions and errors)."""
    def order(k):
        return (ARM_ORDER.get(k[0], 9), float(k[1]), STAGES.index(k[2]))

    groups = {}
    for r in records:
        groups.setdefault((r["arm"], r["w"], r["stage"]), []).append(r)
    rows = []
    for k in sorted(groups, key=order):
        sub = groups[k]
        n = len(sub)
        g2 = sum(1 for r in sub if r["g2"])
        e = sum(1 for r in sub if r["e"])
        sc = {s: sum(1 for r in sub if r["status"] == s) for s in STATUS_ORDER}
        sc["total"] = sum(sc.values())
        rows.append({"arm": k[0], "w": k[1], "stage": k[2], "n_req": n, "g2": g2, "e": e,
                     "g2_wilson": _wil(g2, n), "e_wilson": _wil(e, n), "status_counts": sc})
    tg = {}
    for r in records:
        tg.setdefault((r["arm"], r["w"], r["stage"], r["target_id"]), []).append(r)
    per_target = []
    for k in sorted(tg, key=lambda k: order(k[:3]) + (k[3],)):
        sub = tg[k]
        per_target.append({"arm": k[0], "w": k[1], "stage": k[2], "target_id": k[3], "gamma_target": sub[0]["gamma_target"],
                           "n_req": len(sub), "g2": sum(1 for r in sub if r["g2"]), "e": sum(1 for r in sub if r["e"])})
    return {"layer": layer, "theta": THETA_A, "rows": rows, "per_target": per_target}


def analyze(layer, manifest_rows, csv_rows):
    return aggregate(layer, build_records(layer, manifest_rows, csv_rows))


# ---------------------------------------------------------------------------
# known-value check (layer=main. Source RES/structure_filtered/report.md:20-26,32-38,44-49)
# ---------------------------------------------------------------------------
def check_known_main(result):
    """Returns: [(item, expected, measured, ok)]. Population = proposed and baseline of the main layer, w=2.0, 350 each (targets 0 to 6 x 50); estimator = counts and Wilson interval."""
    if result["layer"] != "main":
        raise InputError("--check-known-main requires --layer main")
    out = []
    post = {(r["arm"], r["w"]): r for r in result["rows"] if r["stage"] == "post"}
    for arm, g2x, ex in (("proposed", 350, 350), ("baseline", 74, 0)):
        r = post.get((arm, "2.0"))
        out.append((f"{arm} w2.0 post n_req", 350, None if r is None else r["n_req"], r is not None and r["n_req"] == 350))
        out.append((f"{arm} w2.0 post G2", g2x, None if r is None else r["g2"], r is not None and r["g2"] == g2x))
        out.append((f"{arm} w2.0 post E", ex, None if r is None else r["e"], r is not None and r["e"] == ex))
    van = {p["target_id"]: p for p in result["per_target"] if p["arm"] == "baseline" and p["w"] == "2.0" and p["stage"] == "post"}
    exp_t = [3, 8, 11, 18, 22, 1, 11]
    act_t = [van[t]["g2"] if t in van else None for t in range(7)]
    out.append(("baseline w2.0 post G2 per target (target 0 to 6)", exp_t, act_t, act_t == exp_t))
    if 0 in van:
        lo, hi = wilson(van[0]["g2"], van[0]["n_req"])
        act_w = [round(lo, 3), round(hi, 3)]
    else:
        act_w = None
    out.append(("Wilson 95% of the G2 rate of baseline target0 (3 decimals)", [0.021, 0.162], act_w, act_w == [0.021, 0.162]))
    return out


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def _fmt_wil(k, n, w):
    if w["p"] is None:
        return f"- ({k}/{n})"
    return f"{w['p']:.3f} [{w['lo']:.3f}, {w['hi']:.3f}] ({k}/{n})"


def render_md(result, args_desc, known):
    L = []
    A = L.append
    A(f"# gb_symfit counts of G2 and E: layer={result['layer']}")
    A("")
    A(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} / plan: {DESIGN}")
    A(f"- Inputs: {args_desc}")
    A(f"- G2 = status in {{fitted,no_gb,multi_gb,disordered}} AND gb0_status == two_gb. E = G2 AND gb_symfit_max_A <= theta. theta = {result['theta']} A (frozen).")
    A("- The denominator is fixed to n_req (number of manifest rows per (arm,w)). Exclusions and errors are counted as misses. No conditional rates and no rates summed over w are given. stage=post is the main one, pre is for reference.")
    A("")
    for st in STAGES:
        A(f"## per (arm,w): stage={st}")
        A("")
        A("| arm | w | n_req | G2 | G2 rate Wilson 95% | E | E rate Wilson 95% | " + " | ".join(STATUS_ORDER) + " | total |")
        A("|---|---|---|---|---|---|---|" + "---|" * (len(STATUS_ORDER) + 1))
        for r in result["rows"]:
            if r["stage"] != st:
                continue
            sc = r["status_counts"]
            A(f"| {r['arm']} | {r['w']} | {r['n_req']} | {r['g2']} | {_fmt_wil(r['g2'], r['n_req'], r['g2_wilson'])} | {r['e']} | {_fmt_wil(r['e'], r['n_req'], r['e_wilson'])} | "
              + " | ".join(str(sc[s]) for s in STATUS_ORDER) + f" | {sc['total']} |")
        A("")
    for st in STAGES:
        A(f"## per (arm,w,target): stage={st} (counts only)")
        A("")
        A("| arm | w | target_id | gamma_target [J/m²] | n_req | G2 | E |")
        A("|---|---|---|---|---|---|---|")
        for p in result["per_target"]:
            if p["stage"] != st:
                continue
            gt = to_finite(p["gamma_target"])
            A(f"| {p['arm']} | {p['w']} | {p['target_id']} | {('%.4f' % gt) if gt is not None else p['gamma_target']} | {p['n_req']} | {p['g2']} | {p['e']} |")
        A("")
    if known is not None:
        A("## Known-value check (--check-known-main. Source structure_filtered/report.md:20-26,32-38,44-49)")
        A("")
        A("| item | expected | measured | decision |")
        A("|---|---|---|---|")
        for name, exp, act, ok in known:
            A(f"| {name} | {exp} | {act} | {'PASS' if ok else 'FAIL'} |")
        A("")
    return "\n".join(L) + "\n"


def write_outputs(prefix, md_text, doc):
    md_path, js_path = Path(f"{prefix}.md"), Path(f"{prefix}.json")
    for p in (md_path, js_path):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite: {p}")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "x") as f:
        f.write(md_text)
    with open(js_path, "x") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return md_path, js_path


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# self_check (synthetic rows only; no file input/output)
# ---------------------------------------------------------------------------
def _man(arm="proposed", w="1.0", chunk=0, i=0, target=0, layer="additional"):
    return {"layer": layer, "arm": arm, "w": w, "chunk": str(chunk), "cif_index": str(i), "target_id": str(target), "gamma_target": "1.0"}


def _row(status, gb0, max_a, stage, arm="proposed", w="1.0", chunk=0, i=0, layer="additional"):
    return {"layer": layer, "arm": arm, "w": w, "chunk": str(chunk), "cif_index": str(i), "stage": stage,
            "status": status, "gb0_status": gb0, "gb_symfit_max_A": ("" if max_a is None else repr(max_a))}


def self_check():
    results = []

    def case(no, desc, ok, detail):
        results.append(ok)
        print(f"[self_check {no}] {'PASS' if ok else 'FAIL'} {desc}: {detail}")

    # 1: inside (the equality at the boundary 0.064 is not used)
    got = classify("fitted", "two_gb", 0.063)
    case(1, "fitted/two_gb/max_A=0.063 → G2=True,E=True", got == (True, True), got)
    # 2: cut by theta
    got = classify("fitted", "two_gb", 0.065)
    case(2, "fitted/two_gb/max_A=0.065 → G2=True,E=False", got == (True, False), got)
    # 3: a single orientation does not enter G2
    got = classify("fitted", "no_gb", None)
    case(3, "fitted/no_gb → G2=False,E=False", got == (False, False), got)
    # 4: exclusions remain in the denominator as misses. It goes through the real route join -> aggregation
    try:
        res = analyze("additional", [_man()], [_row("excluded_pre", "", None, "pre"), _row("excluded_pre", "", None, "post")])
        r = next(x for x in res["rows"] if x["stage"] == "post")
        ok = r["n_req"] == 1 and r["g2"] == 0 and r["e"] == 0 and r["status_counts"]["excluded_pre"] == 1 and r["status_counts"]["total"] == 1
        case(4, "excluded_pre -> G2=False,E=False,n_req unchanged (1)", ok, {"n_req": r["n_req"], "g2": r["g2"], "e": r["e"], "excluded_pre": r["status_counts"]["excluded_pre"]})
    except Exception as ex:  # noqa: BLE001
        case(4, "excluded_pre -> G2=False,E=False,n_req unchanged (1)", False, f"{type(ex).__name__}: {ex}")
    # 5: consistency check 2(a) really fails
    try:
        analyze("additional", [_man()], [_row("error", "two_gb", None, "pre"), _row("error", "two_gb", None, "post")])
        case(5, "gb0_status=two_gb ∧ status=error → InputError", False, "no exception")
    except InputError as ex:
        case(5, "gb0_status=two_gb ∧ status=error → InputError", True, f"InputError: {ex}")
    except Exception as ex:  # noqa: BLE001
        case(5, "gb0_status=two_gb ∧ status=error → InputError", False, f"{type(ex).__name__}: {ex}")
    n_ok = sum(results)
    print(f"[self_check] {n_ok}/5 PASS")
    return 0 if n_ok == 5 and len(results) == 5 else 3


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", choices=["main", "additional"])
    ap.add_argument("--csv")
    ap.add_argument("--manifest")
    ap.add_argument("--out-prefix")
    ap.add_argument("--check-known-main", action="store_true")
    ap.add_argument("--self_check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        sys.exit(self_check())
    for name in ("layer", "csv", "manifest", "out_prefix"):
        if getattr(args, name) is None:
            ap.error(f"--{name.replace('_', '-')} is required (unless --self_check)")
    try:
        md_path, js_path = Path(f"{args.out_prefix}.md"), Path(f"{args.out_prefix}.json")
        for p in (md_path, js_path):
            if p.exists():
                raise FileExistsError(f"refusing to overwrite: {p}")
        manifest_rows = read_csv(args.manifest)
        csv_rows = read_csv(args.csv)
        print(f"[g2_and_e] layer={args.layer} manifest rows={len(manifest_rows)} csv rows={len(csv_rows)}", file=sys.stderr)
        result = analyze(args.layer, manifest_rows, csv_rows)
        known = check_known_main(result) if args.check_known_main else None
        desc = f"csv={args.csv} manifest={args.manifest} (manifest {len(manifest_rows)} rows / csv {len(csv_rows)} rows)"
        doc = dict(result)
        if known is not None:
            doc["known_main_check"] = [{"item": n, "expected": e, "actual": a, "pass": ok} for n, e, a, ok in known]
        write_outputs(args.out_prefix, render_md(result, desc, known), doc)
    except (InputError, FileExistsError) as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"wrote {md_path} and {js_path}", file=sys.stderr)
    if known is not None:
        fails = [n for n, _e, _a, ok in known if not ok]
        for n, e, a, ok in known:
            print(f"[known-main] {'PASS' if ok else 'FAIL'} {n}: expected {e} actual {a}", file=sys.stderr)
        if fails:
            print(f"FAIL: known-value check failed for {len(fails)} item(s)", file=sys.stderr)
            sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
