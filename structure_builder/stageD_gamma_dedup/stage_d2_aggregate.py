#!/usr/bin/env python3
"""Data rebuild, r2, stages D2-3 to 5: aggregation (connect with Union-Find -> keep one representative row).

  stage_d2_aggregate.py [--mode dedup|verify] [--csv NAME] [--match-subdir NAME] [--no-reuse-pilot]
                        [--skip-reference-check] [--results-dir DIR] [--window-ev W] [--pairs-per-task N]

mode=dedup (default): reads gamma_all.csv, match/ and match_trial/ (all pairs of the check group; reused) and
  writes gamma_dedup.csv, dedup_manifest.json and stage_d2_summary.json. The representative is the lexicographic minimum of (seed, k, cnid).
mode=verify: reads gamma_dedup.csv and match_verify/ (--no-reuse-pilot) and writes the number of edges with fit=True to
  stage_d2_reverify_summary.json (0 is expected; if 1 or more, non-zero exit 3).

Exit code: 0 = passed / 2 = missing, exception or unresolved (non-zero if there is even one exception in the matching; it is not treated as "non-duplicate") /
  3 = edges with fit=True remained in verify / 4 = the reference row is not exactly one row (can be skipped with --skip-reference-check).
In the case of 2, gamma_dedup and the manifest are not written (the summary only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "lib"))  # structure_builder/lib
import stage_d_common as D
import stage_d2_match_chunk as M


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def rep_key(r):
    return (r["seed"], r["k"], r["cnid"])     # lexicographic minimum (seed -> k -> cnid)


def brief(r):
    return {"row": r["row"], "cnid": r["cnid"], "k": r["k"], "seed": r["seed"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dedup", "verify"], default="dedup")
    M.add_args(ap)
    ap.add_argument("--skip-reference-check", action="store_true")
    args = ap.parse_args()
    verify = args.mode == "verify"
    if verify:
        if args.csv == D.GAMMA_ALL:
            args.csv = D.GAMMA_DEDUP
        if args.match_subdir == "match":
            args.match_subdir = "match_verify"
        args.no_reuse_pilot = True
    res = D.results_dir(args.results_dir)
    rows = D.read_gamma(res / args.csv)
    byrow = {r["row"]: r for r in rows}
    conv = [r for r in rows if r["converged"]]
    W, CS = D.window_ev(args.window_ev), D.pairs_per_task(args.pairs_per_task)
    pairs, sizes, pilot_all = D.d2_plan(rows, reuse_pilot=not args.no_reuse_pilot, w=W)
    want = Counter((g, i, j) for (g, i, j) in pairs)
    match_dir = res / args.match_subdir
    nt = M.n_tasks(pairs, CS)
    missing_tasks = [t for t in range(nt) if not (match_dir / f"chunk_{t:04d}.csv").exists()]
    got = D.read_chunks(match_dir) if match_dir.exists() else []
    have = Counter((r["group_key"], int(r["row_i"]), int(r["row_j"])) for r in got)
    problems = []
    if missing_tasks:
        problems.append(f"missing_tasks={len(missing_tasks)}")
    if have != want:
        problems.append(f"pair_set_mismatch: missing={sum((want - have).values())} extra_or_dup={sum((have - want).values())}")
    errors = [r for r in got if r["error"]]
    unresolved = [r for r in got if not r["error"] and r["fit"] not in ("0", "1")]
    edges = [(r["group_key"], int(r["row_i"]), int(r["row_j"])) for r in got if r["fit"] == "1"]
    n_from_pilot = 0
    if pilot_all:                                     # reuse the result of the all-pair matching of the check group of D1
        pj_path = res / "stage_d1_match_trial.json"
        if not pj_path.exists():
            problems.append("pilot_json_missing")
        else:
            pj = json.load(open(pj_path))
            if pj.get("gamma_csv_sha256") != D.sha256_file(res / args.csv):
                problems.append("pilot_result_stale (gamma csv sha256 differs from the one pilot was run on)")
            pgot = [r for r in D.read_chunks(res / "match_trial") if r["kind"] == "pair"]
            if Counter((D.pilot_group_key(), int(r["row_i"]), int(r["row_j"])) for r in pgot) != Counter(pilot_all):
                problems.append("pilot_pair_set_mismatch")
            errors += [r for r in pgot if r["error"]]
            edges += [(D.pilot_group_key(), int(r["row_i"]), int(r["row_j"])) for r in pgot if r["fit"] == "1"]
            n_from_pilot = len(pgot)
    if errors:
        problems.append(f"errors={len(errors)}")
    if unresolved:
        problems.append(f"unresolved={len(unresolved)}")

    summary = {
        "mode": args.mode, "csv": args.csv, "match_subdir": args.match_subdir, "window_eV": W, "pairs_per_task": CS,
        "reuse_pilot": not args.no_reuse_pilot, "n_rows_converged_in": len(conv),
        "n_rows_dropped_unconverged": len(rows) - len(conv), "n_groups": len(sizes),
        "n_tasks": nt, "n_pairs_in_tasks": len(pairs), "n_pairs_from_pilot_reuse": len(pilot_all),
        "n_pairs_total_expected": len(pairs) + len(pilot_all), "n_pairs_read": len(got) + n_from_pilot,
        "n_errors": len(errors), "errors_sample": [{k: r.get(k) for k in ("group_key", "row_i", "row_j", "error")}
                                                    for r in errors[:10]],
        "n_unresolved": len(unresolved), "n_fit_true": len(edges), "problems": problems,
    }
    out_name = "stage_d2_reverify_summary.json" if verify else "stage_d2_summary.json"
    if problems:
        summary["status"] = "FAIL"
        json.dump(summary, open(res / out_name, "w"), indent=1)
        print(json.dumps({k: v for k, v in summary.items() if k != "errors_sample"}, indent=1))
        print("FAIL:", "; ".join(problems))
        sys.exit(2)

    if verify:
        summary["n_fit_edges"] = len(edges)
        summary["status"] = "PASS" if not edges else "FAIL"
        json.dump(summary, open(res / out_name, "w"), indent=1)
        print(json.dumps(summary, indent=1))
        sys.exit(0 if not edges else 3)

    uf = UF()
    for r in conv:
        uf.find(r["row"])
    for (_, i, j) in edges:
        uf.union(i, j)
    comps: dict[int, list[int]] = {}
    for r in conv:
        comps.setdefault(uf.find(r["row"]), []).append(r["row"])
    edge_in = Counter(uf.find(i) for (_, i, _j) in edges)
    keep, manifest_groups, multi = set(), {}, []
    non_clique = 0
    for root, members in comps.items():
        mem = [byrow[m] for m in members]
        rep = min(mem, key=rep_key)
        keep.add(rep["row"])
        gk = D.group_key(rep)
        gs = manifest_groups.setdefault(gk, {"n_rows": 0, "n_components": 0, "n_removed": 0})
        gs["n_rows"] += len(mem)
        gs["n_components"] += 1
        gs["n_removed"] += len(mem) - 1
        if len(mem) > 1:
            if edge_in[root] != len(mem) * (len(mem) - 1) // 2:
                non_clique += 1
            multi.append({"group_key": gk, "representative": brief(rep),
                          "members": [brief(m) for m in sorted(mem, key=rep_key)],
                          "n_fit_edges": edge_in[root], "clique": edge_in[root] == len(mem) * (len(mem) - 1) // 2})
    kept_rows = [r for r in conv if r["row"] in keep]
    n_removed = len(conv) - len(kept_rows)
    ref = [r for r in kept_rows if r["cnid"] == "cnid_0_0" and r["k"] == 0 and r["converged"]
           and "/dx_p0.000/k0_s" in r["dir"]]
    de_max = max((abs(byrow[i]["energy_eV"] - byrow[j]["energy_eV"]) for (_, i, j) in edges), default=None)
    summary.update({
        "n_components": len(comps), "n_components_multi": len(multi), "n_removed": n_removed,
        "n_rows_out": len(kept_rows), "n_non_clique_components": non_clique,
        "dE_max_fit_true_eV": de_max, "n_reference_rows": len(ref), "removed_by_group": {
            k: v["n_removed"] for k, v in manifest_groups.items()},
    })
    ref_bad = (len(ref) != 1) and not args.skip_reference_check
    summary["status"] = "FAIL" if ref_bad else "PASS"
    json.dump(summary, open(res / out_name, "w"), indent=1)
    import csv
    tmp = res / (D.GAMMA_DEDUP + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(D.COLS)
        for r in kept_rows:
            w.writerow([r["raw"][c] for c in D.COLS])
    os.replace(tmp, res / D.GAMMA_DEDUP)
    json.dump({"window_eV": W, "groups": manifest_groups, "components": multi},
              open(res / "dedup_manifest.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if ref_bad:
        print(f"FAIL: reference row count = {len(ref)} (expected exactly 1)")
        sys.exit(4)


if __name__ == "__main__":
    main()
