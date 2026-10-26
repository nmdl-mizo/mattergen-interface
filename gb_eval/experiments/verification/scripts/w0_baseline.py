#!/usr/bin/env python3
"""w0_baseline.py -- aggregation that lines up the neighbourhood achievement rate (unconditional baseline) of the unconditional generation (w=0) of proposed with the main experiment (w=2.0) and train. No decision (gate) is made (descriptive).

Plan: see the experiment description.
Input (paths follow the layer convention of common. The layer proposed_w0 has work_root=VRUN):
  - manifest of the w=0 layer (including gamma_pre=gamma_unrelaxed and excluded_pre of the evaluation): VRUN/results/inputs_manifest_<layer>.csv (350 rows)
  - relaxation of the w=0 layer: VRUN/results/relax/<layer>/proposed_w0.0/chunk<cc>/{RELAX_DONE,relax_rows.csv} (14 x 25 rows)
  - per-target CSV of the main experiment: RUN/results/relax_novelty_per_target.csv (7 rows of layer=main, arm=proposed, w=2.0. near_pre_k, near_post_k, n_req)
  - gamma of train: gamma_Jm2 of RUN/results/train_ref_manifest.csv (4,419 rows)
  - (optional) layer aggregation of g2_and_e: if VRUN/results/gb_equivalence/gb_symfit_<layer>_g2_and_e.json exists, the counts of G2 and E go into the summary
Definition (the same as aggregate_relax_novelty.py:328-329 of the main experiment. Decision width K.NEAR_TOL_J=0.10 J/m2):
  near (before relaxation) = not excluded_pre, gamma_pre finite and |gamma_pre - gamma_target| <= 0.10
  near (after relaxation) = status == ok_converged, gamma_relaxed finite and |gamma_relaxed - gamma_target| <= 0.10
  The denominator is fixed to the requested number (pool 350, seed-matched comparison 50, train 4,419). Exclusions, unconverged structures and errors are counted as misses.
Output (VRUN/results/w0_baseline/. exit 2 without overwriting if any exists): base_rate.csv (7 rows), w0_gamma_summary.json, fig_w0_vs_w2.png
Exit code: 0 = success / 2 = failure of the inputs or consistency checks (no silent fallback) / 3 = the train counts disagree with the known values (714/985/427/85/0/0/0) (the output is kept; degeneracy check)
  --self_check: checks only the pure functions of the aggregation with synthetic data and exits (reads and writes no files)
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

LAYER_DEFAULT = "proposed_w0"
ARM = "proposed"
N_POOL = K.N_PER_ARM_W            # 350 (fixed value of the denominator. Not replaced by the number of rows of the real data)
N_PAIR = K.PER_TARGET             # 50
N_TARGETS = K.N_TARGETS           # 7
N_TRAIN = 4419
NEAR = K.NEAR_TOL_J               # 0.10 (definition of the card. Not changed)
TRAIN_KNOWN = [714, 985, 427, 85, 0, 0, 0]   # cross-check table (an internal note that is not included:61)
MAIN_W = "2.0"
RELAX_STATUSES = ("ok_converged", "ok_unconverged", "excluded_pre", "nonfinite", "error")
QUANTS = (("p10", 0.10), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90))


class InputError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# pure functions (the target of self_check)
# ---------------------------------------------------------------------------
def near_pre(rec, g, tol=NEAR):
    return bool(rec["valid_pre"] and rec["gamma_pre"] is not None and abs(rec["gamma_pre"] - g) <= tol)


def near_post(rec, g, tol=NEAR):
    return bool(rec["converged"] and rec["gamma_relaxed"] is not None and abs(rec["gamma_relaxed"] - g) <= tol)


def quantile(sorted_vals, p):
    """The same as the default (linear) of numpy.percentile. None for empty."""
    n = len(sorted_vals)
    if n == 0:
        return None
    idx = (n - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def describe(vals):
    s = sorted(vals)
    n = len(s)
    return {"n": n, "quantiles": {k: quantile(s, p) for k, p in QUANTS}, "min": s[0] if n else None, "max": s[-1] if n else None,
            "mean": (sum(s) / n) if n else None}


def base_rate_rows(recs, targets, main_by_t, train_gammas, tol=NEAR):
    """recs: 350 records (chunk, valid_pre, gamma_pre, converged, gamma_relaxed). targets: gamma_target of length 7.
    main_by_t: {t: {"near_pre_k","near_post_k","n_req"}}. train_gammas: 4,419 values of gamma. The denominator is fixed to N_POOL, N_PAIR and N_TRAIN."""
    rows = []
    for t, g in enumerate(targets):
        pool_pre = sum(1 for r in recs if near_pre(r, g, tol))
        pool_post = sum(1 for r in recs if near_post(r, g, tol))
        pair = [r for r in recs if r["chunk"] // K.CHUNKS_PER_TARGET == t]
        pair_pre = sum(1 for r in pair if near_pre(r, g, tol))
        pair_post = sum(1 for r in pair if near_post(r, g, tol))
        tr = sum(1 for x in train_gammas if abs(x - g) <= tol)
        rows.append({
            "target_id": t, "gamma_target": repr(g),
            "n_pool": N_POOL, "pool_pre_k": pool_pre, "pool_pre_rate": pool_pre / N_POOL,
            "pool_post_k": pool_post, "pool_post_rate": pool_post / N_POOL,
            "pair_chunks": f"{t * K.CHUNKS_PER_TARGET},{t * K.CHUNKS_PER_TARGET + 1}", "n_pair": N_PAIR,
            "w0_pair_pre_k": pair_pre, "w0_pair_post_k": pair_post,
            "main_n_req": main_by_t[t]["n_req"], "main_near_pre_k": main_by_t[t]["near_pre_k"], "main_near_post_k": main_by_t[t]["near_post_k"],
            "train_near_k": tr, "n_train": N_TRAIN, "train_rate": tr / N_TRAIN,
        })
    return rows


BASE_RATE_FIELDS = ["target_id", "gamma_target", "n_pool", "pool_pre_k", "pool_pre_rate", "pool_post_k", "pool_post_rate",
                    "pair_chunks", "n_pair", "w0_pair_pre_k", "w0_pair_post_k", "main_n_req", "main_near_pre_k", "main_near_post_k",
                    "train_near_k", "n_train", "train_rate"]


def self_check():
    """Check with synthetic data (reads and writes no files). AssertionError on failure."""
    tg = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    recs = []
    for c in range(K.N_CHUNKS):
        for i in range(K.PER_CHUNK):
            recs.append({"chunk": c, "valid_pre": True, "gamma_pre": 1.0, "converged": True, "gamma_relaxed": 1.0})
    main = {t: {"near_pre_k": 1, "near_post_k": 2, "n_req": N_PAIR} for t in range(N_TARGETS)}
    train = [1.0] * 10 + [2.0] * 5
    # (1) all 350 coincide exactly with target 0 (gamma=1.0) -> target 0 is 350/350 and 50/50, the other targets are 0
    rows = base_rate_rows(recs, tg, main, train)
    assert rows[0]["pool_pre_k"] == 350 and rows[0]["pool_post_k"] == 350 and rows[0]["w0_pair_pre_k"] == 50 and rows[0]["w0_pair_post_k"] == 50
    assert all(rows[t]["pool_pre_k"] == 0 and rows[t]["w0_pair_post_k"] == 0 for t in range(1, N_TARGETS))
    assert rows[0]["pool_post_rate"] == 1.0 and rows[0]["train_near_k"] == 10 and rows[1]["train_near_k"] == 5 and rows[0]["n_train"] == N_TRAIN
    # (2) degenerate: even if all are excluded or unconverged, the denominator stays 350 (rate 0.0, count 0). Exclusions do not shrink the denominator
    dead = [{"chunk": c, "valid_pre": False, "gamma_pre": None, "converged": False, "gamma_relaxed": None} for c in range(K.N_CHUNKS) for _ in range(K.PER_CHUNK)]
    rows = base_rate_rows(dead, tg, main, train)
    assert all(r["pool_pre_k"] == 0 and r["pool_post_k"] == 0 and r["pool_pre_rate"] == 0.0 and r["n_pool"] == N_POOL for r in rows)
    # (3) even if unconverged (converged=False), a finite gamma_relaxed is a miss. Even if valid_pre=False, a finite gamma_pre is a miss (the same as the definition of the main experiment)
    r1 = {"chunk": 0, "valid_pre": False, "gamma_pre": 1.0, "converged": False, "gamma_relaxed": 1.0}
    assert not near_pre(r1, 1.0) and not near_post(r1, 1.0)
    # (4) boundary of the decision width (with exactly representable values: tol=0.25, difference 0.25 is inside, difference 0.2500001 is outside)
    r2 = {"chunk": 0, "valid_pre": True, "gamma_pre": 1.25, "converged": True, "gamma_relaxed": 1.2500001}
    assert near_pre(r2, 1.0, 0.25) and not near_post(r2, 1.0, 0.25)
    # (5) pairs of chunks for each target: only chunks 2t and 2t+1 form a pair (chunk 5 is target 2)
    only5 = [{"chunk": 5, "valid_pre": True, "gamma_pre": 3.0, "converged": True, "gamma_relaxed": 3.0}]
    rows = base_rate_rows(only5, tg, main, train)
    assert rows[2]["w0_pair_pre_k"] == 1 and rows[2]["pair_chunks"] == "4,5" and rows[1]["w0_pair_pre_k"] == 0 and rows[3]["w0_pair_pre_k"] == 0
    # (6) quantiles: p50=3.0, p10=1.4, p90=4.6 for [1,2,3,4,5]; None for empty
    d = describe([5.0, 1.0, 3.0, 2.0, 4.0])
    assert d["quantiles"]["p50"] == 3.0 and abs(d["quantiles"]["p10"] - 1.4) < 1e-12 and abs(d["quantiles"]["p90"] - 4.6) < 1e-12 and d["n"] == 5 and d["mean"] == 3.0
    e = describe([])
    assert e["n"] == 0 and e["quantiles"]["p50"] is None and e["min"] is None and e["mean"] is None
    print("self_check PASS (6 checks)")


# ---------------------------------------------------------------------------
# reading of the inputs (with checks of counts and consistency)
# ---------------------------------------------------------------------------
def load_w0_records(layer):
    cfg = K.layer_cfg(layer)
    ws = cfg["ws"]
    if cfg["arms"] != [ARM] or len(ws) != 1 or ws[0] != 0.0:
        raise InputError(f"layer {layer!r} must be proposed-only with ws=[0.0] (got arms={cfg['arms']} ws={ws})")
    manifest = K.manifest_path_for(layer)
    mrows = K.read_csv_rows(manifest)
    if len(mrows) != N_POOL:
        raise InputError(f"{manifest}: {len(mrows)} rows != {N_POOL}")
    by_chunk = {}
    for r in mrows:
        if r["layer"] != layer or r["arm"] != ARM or K.to_float(r["w"]) != 0.0:
            raise InputError(f"{manifest}: row with layer/arm/w = {r['layer']}/{r['arm']}/{r['w']} (expected {layer}/{ARM}/0.0)")
        k = (int(r["chunk"]), int(r["cif_index"]))
        d = by_chunk.setdefault(k[0], {})
        if k[1] in d:
            raise InputError(f"{manifest}: duplicated key chunk={k[0]} cif_index={k[1]}")
        d[k[1]] = r
    if sorted(by_chunk) != list(range(K.N_CHUNKS)) or any(sorted(d) != list(range(K.PER_CHUNK)) for d in by_chunk.values()):
        raise InputError(f"{manifest}: chunks/cif_index are not 0..{K.N_CHUNKS - 1} x 0..{K.PER_CHUNK - 1}")
    relax_root = Path(K.relax_root_for(layer)) / K.armw(ARM, ws[0])
    recs = []
    targets = {}
    for c in range(K.N_CHUNKS):
        rd = relax_root / K.chunk_name(c)
        if not (rd / "RELAX_DONE").is_file():
            raise InputError(f"{rd}/RELAX_DONE missing")
        rrows = K.read_csv_rows(rd / "relax_rows.csv")
        if len(rrows) != K.PER_CHUNK:
            raise InputError(f"{rd}/relax_rows.csv has {len(rrows)} rows != {K.PER_CHUNK}")
        rr = {int(x["cif_index"]): x for x in rrows}
        if sorted(rr) != list(range(K.PER_CHUNK)):
            raise InputError(f"{rd}/relax_rows.csv: cif_index is not 0..{K.PER_CHUNK - 1}")
        for i in range(K.PER_CHUNK):
            m, x = by_chunk[c][i], rr[i]
            if (x["excluded_pre"] or "") != (m["excluded_pre"] or ""):
                raise InputError(f"{rd}: excluded_pre differs between manifest and relax_rows at cif_index {i}")
            status = x["status"]
            if status not in RELAX_STATUSES:
                raise InputError(f"{rd}: unknown status {status!r} at cif_index {i}")
            if (status == "excluded_pre") != bool(m["excluded_pre"]):
                raise InputError(f"{rd}: status/excluded_pre mismatch at cif_index {i}")
            t = int(m["target_id"])
            if t != c // K.CHUNKS_PER_TARGET:
                raise InputError(f"{manifest}: chunk {c} has target_id {t} != {c // K.CHUNKS_PER_TARGET}")
            g = float(m["gamma_target"])
            if targets.setdefault(t, g) != g:
                raise InputError(f"{manifest}: target_id {t} has two gamma_target values ({targets[t]} vs {g})")
            valid_pre = not m["excluded_pre"]
            recs.append({"chunk": c, "cif_index": i, "target_id": t, "valid_pre": valid_pre,
                         "gamma_pre": K.to_float(m["gamma_unrelaxed"]) if valid_pre else None,
                         "status": status, "converged": status == "ok_converged", "gamma_relaxed": K.to_float(x.get("gamma_relaxed"))})
    if sorted(targets) != list(range(N_TARGETS)):
        raise InputError(f"{manifest}: target_id set {sorted(targets)} != 0..{N_TARGETS - 1}")
    if len(recs) != N_POOL:
        raise InputError(f"internal: {len(recs)} records != {N_POOL}")
    return recs, [targets[t] for t in range(N_TARGETS)]


def load_main_per_target():
    path = f"{K.RUN}/results/relax_novelty_per_target.csv"
    out = {}
    for r in K.read_csv_rows(path):
        if r["layer"] == "main" and r["arm"] == ARM and f"{float(r['w']):.1f}" == MAIN_W:
            t = int(r["target_id"])
            if t in out:
                raise InputError(f"{path}: duplicated main proposed w={MAIN_W} target_id {t}")
            out[t] = {"near_pre_k": int(float(r["near_pre_k"])), "near_post_k": int(float(r["near_post_k"])), "n_req": int(float(r["n_req"]))}
    if sorted(out) != list(range(N_TARGETS)) or any(v["n_req"] != N_PAIR for v in out.values()):
        raise InputError(f"{path}: main proposed w={MAIN_W} rows are not 7 targets x n_req={N_PAIR} (got {sorted(out)} n_req={[v['n_req'] for v in out.values()]})")
    return out, path


def load_train_gammas():
    path = f"{K.RUN}/results/train_ref_manifest.csv"
    g = []
    for r in K.read_csv_rows(path):
        v = K.to_float(r["gamma_Jm2"])
        if v is None:
            raise InputError(f"{path}: non-finite gamma_Jm2 at row_index {r.get('row_index')}")
        g.append(v)
    if len(g) != N_TRAIN:
        raise InputError(f"{path}: {len(g)} rows != {N_TRAIN}")
    return g, path


def load_g2_and_e(layer):
    p = Path(K.P(f"{K.VRUN}/results/gb_equivalence/gb_symfit_{layer}_g2_and_e.json"))
    if not p.is_file():
        return None, str(p)
    doc = json.load(open(p))
    cells = {}
    for r in doc["rows"]:
        if r["arm"] == ARM and K.to_float(r["w"]) == 0.0:
            cells[r["stage"]] = {"n_req": r["n_req"], "g2": r["g2"], "e": r["e"]}
    return cells, str(p)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def make_figure(path, rows, targets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    xs = list(range(n))
    wd = 0.27
    w0 = [100.0 * r["pool_post_rate"] for r in rows]
    w2 = [100.0 * r["main_near_post_k"] / r["main_n_req"] for r in rows]
    tr = [100.0 * r["train_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar([x - wd for x in xs], w0, wd, label="w = 0 (pooled, /350)")
    ax.bar(xs, w2, wd, label="w = 2.0 (main, per target, /50)")
    ax.bar([x + wd for x in xs], tr, wd, label="train (/4,419)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g:.4f}" for g in targets], rotation=0)
    ax.set_xlabel("target gamma (J/m^2)")
    ax.set_ylabel("share within |gamma - target| <= 0.10 (%)")
    ax.set_title("Near-target share after relaxation: unconditional (w=0) vs conditioned (w=2.0) vs train")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", default=LAYER_DEFAULT)
    ap.add_argument("--out-dir", default=f"{K.VRUN}/results/w0_baseline")
    ap.add_argument("--self_check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        self_check()
        return 0
    out_dir = Path(K.P(a.out_dir))
    outs = {n: out_dir / n for n in ("base_rate.csv", "w0_gamma_summary.json", "fig_w0_vs_w2.png")}
    try:
        for p in outs.values():
            if p.exists():
                raise FileExistsError(f"refusing to overwrite: {p}")
        recs, targets = load_w0_records(a.layer)
        main_by_t, main_path = load_main_per_target()
        train, train_path = load_train_gammas()
        g2_and_e, g2_and_e_path = load_g2_and_e(a.layer)
    except (InputError, K.InputError, FileExistsError, FileNotFoundError, KeyError, ValueError) as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    rows = base_rate_rows(recs, targets, main_by_t, train)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(outs["base_rate.csv"], "x", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=BASE_RATE_FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    pre_vals = [r["gamma_pre"] for r in recs if r["valid_pre"] and r["gamma_pre"] is not None]
    post_vals = [r["gamma_relaxed"] for r in recs if r["converged"] and r["gamma_relaxed"] is not None]
    st = {s: sum(1 for r in recs if r["status"] == s) for s in RELAX_STATUSES}
    train_counts = [r["train_near_k"] for r in rows]
    summary = {
        "layer": a.layer, "arm": ARM, "w": 0.0, "n_req": N_POOL, "near_tol_J_per_m2": NEAR, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "targets": targets,
        "w0_pre": {**describe(pre_vals), "n_excluded_pre": sum(1 for r in recs if not r["valid_pre"])},
        "w0_post": {**describe(post_vals), "n_unconverged": st["ok_unconverged"], "n_nonfinite": st["nonfinite"], "n_error": st["error"], "n_excluded_pre": st["excluded_pre"]},
        "w0_status_counts": st,
        "train": {**describe(train), "n_req": N_TRAIN},
        "g2_and_e_layer": g2_and_e,
        "train_known_check": {"expected": TRAIN_KNOWN, "actual": train_counts, "match": train_counts == TRAIN_KNOWN},
        "inputs": {"manifest": K.manifest_path_for(a.layer), "relax_root": K.relax_root_for(a.layer), "main_per_target": main_path,
                   "train_ref_manifest": train_path, "g2_and_e_json": g2_and_e_path},
    }
    with open(outs["w0_gamma_summary.json"], "x") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
        f.write("\n")
    make_figure(str(outs["fig_w0_vs_w2.png"]), rows, targets)
    for p in outs.values():
        if not p.is_file() or p.stat().st_size == 0:
            print(f"FATAL: output missing or empty: {p}", file=sys.stderr)
            return 2
    print(f"[w0_baseline] layer={a.layer} rows={len(rows)} pool_post_k={[r['pool_post_k'] for r in rows]} pair_post_k={[r['w0_pair_post_k'] for r in rows]} "
          f"main_post_k={[r['main_near_post_k'] for r in rows]} train_k={train_counts} g2_and_e={'present' if g2_and_e else 'absent'}")
    if train_counts != TRAIN_KNOWN:
        print(f"FAIL: train counts {train_counts} != known {TRAIN_KNOWN} (degeneracy check. The output was kept)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
