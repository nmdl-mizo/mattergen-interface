#!/usr/bin/env python3
# Copy of the range-target version. Source: ${GBREL_CODE}/gb_eval/experiments/gen_eval/scripts/aggregate_tracking.py.
# Only RUN_DIR (the value of the RUN constant), TARGETS_JSON, CARD_HASH12 and the range of k were changed (this py has only the RUN constant and TARGETS_JSON).
"""aggregate_tracking.py -- aggregation and figures of the tracking of the proposed model and the baseline (all are reference values; no threshold of acceptance is set).

Input: <eval-root>/<model>_w<W>/gamma_pre.csv and validation.json for all (arm, w) combinations of stage E (of the 2 arms x w=1.0 to 4.0, determined by --ws or by the existence in eval-root; 6 files if only w=1 to 3).
  If any is absent, or the verdict of validation.json is not PASS, it exits non-zero without aggregating.
Classification of rows (350 rows per (arm, w); mutually disjoint and decided from the top):
  1 excluded_error  : error is not empty (angle_fail, nonorthogonal, nonfinite, calculation failure)
  2 excluded_non_al : non_al_count > 0
  3 valid           : neither of the above
Main aggregation of the 5 inner targets (target_id 0..4, requested 250): beta_ts and its 95% CI (scipy theilslopes(gamma_pre, gamma_target, alpha=0.95), valid only, same formula as in the earlier analysis),
  hit_rate_valid (denominator = number of valid; comparison metric (2) of the model specification), hit_rate_req (denominator = 250; auxiliary), mae, mean_dev.
Per target (7 targets): requested 50, valid, excluded, exclusion rate, median/IQR of gamma_pre, median of the error, median of |error|, hit_rate_req (denominator 50), hit_rate_valid.
Source of the statistical formulas: ${GBREL_CODE}/gb_eval/generation_hooks/aggregate_tracking.py
  (arm_stats_from_arrays -> the theilslopes call of aggregate_beta.core_beta_stats, and wilson_ci and band_rate of the earlier aggregation script are copied as functions).
  The decision function (PASS/FAIL/FLAG) of the earlier check is not copied (ruling: no acceptance is set for the tracking).

Output: <out-dir>/tracking_summary.json, tracking_per_target.csv, tracking_report.md, figs/tracking_<model>_w<W>.{png,pdf}, figs/tracking_grid.{png,pdf}
  (not overwritten if they exist. The grid figure has 2 rows x (number of w) columns. If only some are given with --pairs, it is for validation and the grid figure is not made)
C-S: if the number of excluded_non_al per (arm, w) is 18 or more (exceeds 5% of 350 = 17.5), FLAG (report only; the aggregation continues), 17 or less is PASS.

Execution environment: the evaluation environment (numpy, scipy, pandas, matplotlib. Not executed on the login node). PYTHONDONTWRITEBYTECODE=1.
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

RUN_REF = Path(_gbrel.WORK + '/workspace/experiments/range_targets')
TARGETS_JSON = Path(_gbrel.WORK + '/workspace/experiments/range_targets/results/eval_targets_range.json')
ARMS = ["proposed", "baseline"]
WS_ALL = ["1.0", "2.0", "3.0", "4.0"]   # allowed w (w=4.0 was added). The w actually aggregated is decided by main() (--ws or the existence in eval-root)
N_PER_PAIR = 350
N_PER_TARGET = 50
N_INNER_TARGETS = 5
N_INNER_REQ = N_INNER_TARGETS * N_PER_TARGET   # 250
BAND = 0.10                  # J/m^2 (reference band of the evaluation criterion of the model specification)
GB_DX_LIMIT_A = 3.0
S_FLAG_MIN = 18          # 350 x 0.05 = 17.5, exceeding it = 18 or more
REQUIRED_COLS = ["arm", "gamma_target", "gamma_unrelaxed", "natoms", "error", "gb_dx_realized_A", "w", "chunk", "target_id",
                 "is_inner", "ckpt_sha256", "non_al_count", "min_dist_A"]
LABELS = ["inner0", "inner1", "inner2", "inner3", "inner4", "extrap_low", "extrap_high"]


class AggError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# statistics (formulas of the source)
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    """Copy of the corresponding function of an earlier aggregation script."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)) / denom
    return float(center - half), float(center + half)


def ts_beta(t: np.ndarray, y: np.ndarray) -> dict:
    """Same formula as in the earlier analysis: scipy.stats.theilslopes(y, t, alpha=0.95) (same call as aggregate_beta.core_beta_stats:152)."""
    from scipy.stats import theilslopes

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2 or np.unique(t).size < 2:
        return {"n": int(t.size), "beta_ts": None, "ci_lo": None, "ci_hi": None, "intercept": None}
    beta, inter, lo, hi = theilslopes(y, t, alpha=0.95)
    return {"n": int(t.size), "beta_ts": float(beta), "ci_lo": float(lo), "ci_hi": float(hi), "intercept": float(inter)}


def rate_block(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n > 0 else None, "wilson_lo": lo, "wilson_hi": hi}


def _f(x):
    """For JSON: NaN/inf -> None, numpy -> float."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# ---------------------------------------------------------------------------
# reading and classification of the inputs
# ---------------------------------------------------------------------------
def load_targets() -> dict:
    j = json.loads(TARGETS_JSON.read_text())
    if len(j["targets"]) != 7:
        raise AggError(f"{TARGETS_JSON}: targets length {len(j['targets'])} != 7")
    return j


def require_g63v_pass(path: Path) -> dict:
    """validation.json is absent or not PASS -> exception (no aggregation)."""
    if not path.is_file():
        raise AggError(f"validation.json missing: {path} (no aggregation without the verdict of the previous stage)")
    payload = json.loads(path.read_text())
    if payload.get("verdict") != "PASS":
        raise AggError(f"g63v verdict={payload.get('verdict')!r} (not PASS): {path}")
    return payload


def load_pair(eval_root: Path, arm: str, w: str, targets: list) -> pd.DataFrame:
    d = eval_root / f"{arm}_w{w}"
    g63v = require_g63v_pass(d / "g63v.json")
    csvp = d / "gamma_pre.csv"
    if not csvp.is_file():
        raise AggError(f"gamma_pre.csv missing: {csvp}")
    df = pd.read_csv(csvp, dtype={"error": str}, float_precision="round_trip")   # the default parser can be off by the last 1 ulp (for the exact-agreement check of the target values)
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise AggError(f"{csvp}: missing columns {miss}")
    if len(df) != N_PER_PAIR:
        raise AggError(f"{csvp}: {len(df)} rows != {N_PER_PAIR}")
    df["error"] = df["error"].fillna("").astype(str).str.strip()
    df["gamma_target"] = df["gamma_target"].astype(float)
    df["gamma_unrelaxed"] = pd.to_numeric(df["gamma_unrelaxed"], errors="coerce")
    df["non_al_count"] = df["non_al_count"].astype(int)
    df["target_id"] = df["target_id"].astype(int)
    df["is_inner"] = df["is_inner"].astype(int)
    # classification (mutually disjoint, from the top)
    status = np.where(df["error"] != "", "excluded_error", np.where(df["non_al_count"] > 0, "excluded_non_al", "valid"))
    df["status"] = status
    # invariant: correspondence of the targets (is_inner = target_id <= 4, gamma_target = targets[target_id], 50 rows per target)
    if not ((df["is_inner"] == (df["target_id"] <= 4).astype(int)).all()):
        raise AggError(f"{csvp}: is_inner != (target_id <= 4)")
    for tid in range(7):
        sub = df[df["target_id"] == tid]
        if len(sub) != N_PER_TARGET:
            raise AggError(f"{csvp}: target_id={tid} has {len(sub)} rows != {N_PER_TARGET}")
        if not (sub["gamma_target"] == float(targets[tid])).all():
            raise AggError(f"{csvp}: gamma_target of target_id={tid} != targets[{tid}]={targets[tid]!r}")
    v = df[df["status"] == "valid"]
    if not np.isfinite(v["gamma_unrelaxed"].to_numpy(dtype=float)).all():
        raise AggError(f"{csvp}: a valid row has non-finite gamma_unrelaxed")
    df.attrs["g63v"] = g63v
    return df


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def basic_stats(sub: pd.DataFrame, n_required: int) -> dict:
    """sub = a set of rows (all rows, with the status column). Statistics only from valid rows, and the denominator of hit_rate_req is n_required."""
    v = sub[sub["status"] == "valid"]
    t = v["gamma_target"].to_numpy(dtype=float)
    y = v["gamma_unrelaxed"].to_numpy(dtype=float)
    k = int(np.sum(np.abs(t - y) <= BAND)) if t.size else 0
    return {
        "n_required": int(n_required),
        "n_valid": int(len(v)),
        "n_excluded_error": int((sub["status"] == "excluded_error").sum()),
        "n_excluded_non_al": int((sub["status"] == "excluded_non_al").sum()),
        "beta_ts": ts_beta(t, y),
        "hit_rate_valid": rate_block(k, len(v)),
        "hit_rate_req": rate_block(k, n_required),
        "mae": _f(np.mean(np.abs(y - t))) if t.size else None,
        "mean_dev": _f(np.mean(y - t)) if t.size else None,
    }


def per_target_rows(df: pd.DataFrame, arm: str, w: str, targets: list) -> list[dict]:
    rows = []
    for tid in range(7):
        sub = df[df["target_id"] == tid]
        v = sub[sub["status"] == "valid"]
        y = v["gamma_unrelaxed"].to_numpy(dtype=float)
        t0 = float(targets[tid])
        n_ee = int((sub["status"] == "excluded_error").sum())
        n_en = int((sub["status"] == "excluded_non_al").sum())
        k = int(np.sum(np.abs(y - t0) <= BAND)) if y.size else 0
        q1, med, q3 = (np.percentile(y, [25, 50, 75]) if y.size else (np.nan, np.nan, np.nan))
        rows.append({
            "arm": arm, "w": float(w), "target_id": tid, "label": LABELS[tid], "gamma_target": t0, "is_inner": int(tid <= 4),
            "n_required": len(sub), "n_valid": int(len(v)), "n_excluded_error": n_ee, "n_excluded_non_al": n_en,
            "excluded_rate": (n_ee + n_en) / len(sub),
            "gamma_pre_median": _f(med), "gamma_pre_q1": _f(q1), "gamma_pre_q3": _f(q3),
            "err_median": _f(np.median(y - t0)) if y.size else None,
            "abs_err_median": _f(np.median(np.abs(y - t0))) if y.size else None,
            "hit_k": k,
            "hit_rate_req": k / len(sub),          # denominator 50 (requested number)
            "hit_rate_valid": (k / len(v)) if len(v) else None,
        })
    return rows


def summarize_pair(df: pd.DataFrame, arm: str, w: str, targets: list) -> dict:
    inner = df[df["is_inner"] == 1]
    s = {
        "arm": arm, "w": float(w),
        "denominators": {"inner_main": "beta_ts: valid inner rows; hit_rate_valid: valid inner rows; hit_rate_req: 250 (5 targets x 50, exclusions counted as misses;)",
                         "per_target": "hit_rate_req: 50; hit_rate_valid: valid rows of that target"},
        "n_rows": int(len(df)),
        "classification": {"valid": int((df["status"] == "valid").sum()),
                           "excluded_error": int((df["status"] == "excluded_error").sum()),
                           "excluded_non_al": int((df["status"] == "excluded_non_al").sum())},
        "inner_main": basic_stats(inner, N_INNER_REQ),
        "all7_beta_ts_reference_only": ts_beta(df[df["status"] == "valid"]["gamma_target"].to_numpy(dtype=float),
                                               df[df["status"] == "valid"]["gamma_unrelaxed"].to_numpy(dtype=float)),
        "natoms_distribution": {str(int(k)): int(c) for k, c in df["natoms"].value_counts().sort_index().items()},
    }
    # breakdown of exclusions
    err_counts = df[df["status"] == "excluded_error"]["error"].str.split(":").str[0].value_counts().to_dict()
    s["excluded_error_breakdown"] = {k: int(v) for k, v in err_counts.items()}
    s["n_angle_fail"] = int((df["error"] == "angle_fail").sum())
    s["n_nonorthogonal"] = int((df["error"] == "nonorthogonal").sum())
    s["n_nonfinite"] = int((df["error"] == "nonfinite").sum())
    # CS
    n_non_al_cls = s["classification"]["excluded_non_al"]
    s["CS"] = {"n_excluded_non_al": n_non_al_cls, "rate": n_non_al_cls / len(df),
                    "n_rows_with_non_al_any_status": int((df["non_al_count"] > 0).sum()),
                    "verdict": "FLAG" if n_non_al_cls >= S_FLAG_MIN else "PASS",
                    "rule": f"FLAG if excluded_non_al >= {S_FLAG_MIN} (> 5% of {len(df)}); report only"}
    # min_dist_A
    md = pd.to_numeric(df["min_dist_A"], errors="coerce").dropna().to_numpy(dtype=float)
    s["min_dist_A"] = ({"n": int(md.size), "min": _f(md.min()), "p5": _f(np.percentile(md, 5)), "median": _f(np.median(md))}
                       if md.size else {"n": 0})
    # proposed: distribution of gb_dx_realized_A (valid rows)
    if arm == "proposed":
        gv = pd.to_numeric(df[df["status"] == "valid"]["gb_dx_realized_A"], errors="coerce").dropna().to_numpy(dtype=float)
        s["gb_dx_realized_A"] = ({"n": int(gv.size), "min": _f(gv.min()), "median": _f(np.median(gv)), "max": _f(gv.max()),
                                  "n_abs_gt_3A": int((np.abs(gv) > GB_DX_LIMIT_A).sum())} if gv.size else {"n": 0})
    return s


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def draw_panel(ax, df: pd.DataFrame, targets: list, summ: dict, xlim, ylim, train_range, title: str, small: bool = False):
    from paperfig import OKABE_ITO

    rng = np.random.default_rng(20301000)   # random horizontal offset (only for the appearance of the figure)
    v = df[df["status"] == "valid"]
    lo, hi = xlim
    ax.axvspan(train_range[0], train_range[1], color="0.92", zorder=0, lw=0)
    xs = np.linspace(lo, hi, 50)
    ax.fill_between(xs, xs - BAND, xs + BAND, color=OKABE_ITO[2], alpha=0.18, lw=0, zorder=1)
    ax.plot(xs, xs, color="k", lw=0.6, zorder=2)
    data_for_box, positions = [], []
    for tid in range(7):
        sub = v[v["target_id"] == tid]
        if sub.empty:
            continue
        t0 = float(targets[tid])
        yy = sub["gamma_unrelaxed"].to_numpy(dtype=float)
        jit = rng.uniform(-0.045, 0.045, size=yy.size)
        inner = tid <= 4
        ax.scatter(t0 + jit, yy, s=3.0, marker="o" if inner else "^",
                   color=OKABE_ITO[5] if inner else OKABE_ITO[6], alpha=0.55, lw=0, zorder=3, rasterized=True)
        data_for_box.append(yy)
        positions.append(t0)
    if data_for_box:
        ax.boxplot(data_for_box, positions=positions, widths=0.10, showfliers=False, manage_ticks=False,
                   medianprops=dict(color="k", lw=0.7), boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                   patch_artist=False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=7 if small else 9)
    im = summ["inner_main"]
    bt = im["beta_ts"]
    btxt = "n/a" if bt["beta_ts"] is None else f"{bt['beta_ts']:.3f} [{bt['ci_lo']:.3f}, {bt['ci_hi']:.3f}]"
    hv, hr = im["hit_rate_valid"]["rate"], im["hit_rate_req"]["rate"]
    hv_s = "n/a" if hv is None else format(hv, ".2f")
    hr_s = "n/a" if hr is None else format(hr, ".2f")
    mae_s = "n/a" if im["mae"] is None else format(im["mae"], ".3f")
    n_excl = summ["classification"]["excluded_error"] + summ["classification"]["excluded_non_al"]
    txt = f"inner5 (ref. only)\nbeta_TS {btxt}\nhit(valid) {hv_s}  hit(req) {hr_s}\nMAE {mae_s}  excl {n_excl}/350"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=4.6 if small else 5.5, linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.4, alpha=0.9), zorder=5)


def make_figs(per_pair: dict, dfs: dict, targets: list, train_range, out_figs: Path, grid_ws) -> list[str]:
    """grid_ws: list of w that become the columns of the grid figure (no grid if None). The grid has 2 rows (ARMS) x len(grid_ws) columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paperfig import COL_2, setup_style, save_fig

    setup_style()
    allv = np.concatenate([d[d["status"] == "valid"]["gamma_unrelaxed"].to_numpy(dtype=float) for d in dfs.values()])
    ymin = float(min(allv.min() if allv.size else 0.0, min(targets)) - 0.15)
    ymax = float(max(allv.max() if allv.size else 3.0, max(targets)) + 0.15)
    xlim = (min(targets) - 0.15, max(targets) + 0.15)
    ylim = (ymin, ymax)
    made = []
    for (arm, w), df in dfs.items():
        fig, ax = plt.subplots(figsize=(COL_2 * 0.55, COL_2 * 0.5))
        draw_panel(ax, df, targets, per_pair[(arm, w)], xlim, ylim, train_range, f"{arm}, w={w}")
        ax.set_xlabel(r"target $\gamma$ [J/m$^2$]")
        ax.set_ylabel(r"unrelaxed $\gamma_{\mathrm{pre}}$ [J/m$^2$]")
        save_fig(fig, f"tracking_{arm}_w{w}", str(out_figs), pdf=True)
        plt.close(fig)
        made += [f"tracking_{arm}_w{w}.png", f"tracking_{arm}_w{w}.pdf"]
    if grid_ws:
        ncol = len(grid_ws)
        # for 3 columns the dimensions are as before. For 4 or more columns the width of the figure is widened in proportion to the number of columns to keep the width of one panel
        fig, axes = plt.subplots(2, ncol, figsize=(COL_2 * max(ncol, 3) / 3, COL_2 * 0.68), sharex=True, sharey=True, squeeze=False)
        for i, arm in enumerate(ARMS):
            for j, w in enumerate(grid_ws):
                draw_panel(axes[i][j], dfs[(arm, w)], targets, per_pair[(arm, w)], xlim, ylim, train_range, f"{arm}, w={w}", small=True)
                if i == 1:
                    axes[i][j].set_xlabel(r"target $\gamma$ [J/m$^2$]")
                if j == 0:
                    axes[i][j].set_ylabel(r"$\gamma_{\mathrm{pre}}$ [J/m$^2$]")
        fig.tight_layout()
        save_fig(fig, "tracking_grid", str(out_figs), pdf=True)
        plt.close(fig)
        made += ["tracking_grid.png", "tracking_grid.pdf"]
    return made


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def fmt_ci(b: dict) -> str:
    return "n/a" if b["beta_ts"] is None else f"{b['beta_ts']:.3f} [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]"


def fmt_rate(r: dict) -> str:
    return "n/a" if r["rate"] is None else f"{r['rate']:.3f} ({r['k']}/{r['n']}) [{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}]"


def write_report(path: Path, summary: dict, per_target: list[dict], figs: list[str], pairs: list, targets_j: dict) -> None:
    L = ["# Aggregation of the tracking of proposed / baseline", "",
         "No acceptance threshold is set (evaluation criterion of the model specification). The choice of w is made by looking at the figures. All numbers below are reference values.", "",
         "## Figures", ""]
    L += [f"- `figs/{f}`" for f in figs if f.endswith(".png")]
    L += ["", "## Reference values per (arm, w) (5 inner targets, target_id 0 to 4, requested 250)", "",
          "| arm | w | valid | excl_error | excl_non_al | beta_TS [95% CI] | hit_rate_valid [Wilson] | hit_rate_req (/250) [Wilson] | MAE | mean_dev |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, w in pairs:
        s = summary[f"{arm}_w{w}"]
        im = s["inner_main"]
        L.append(f"| {arm} | {w} | {im['n_valid']} | {im['n_excluded_error']} | {im['n_excluded_non_al']} | {fmt_ci(im['beta_ts'])} | "
                 f"{fmt_rate(im['hit_rate_valid'])} | {fmt_rate(im['hit_rate_req'])} | "
                 f"{'n/a' if im['mae'] is None else format(im['mae'], '.4f')} | {'n/a' if im['mean_dev'] is None else format(im['mean_dev'], '+.4f')} |")
    L += ["", "- hit_rate_valid = comparison metric (2) of the model specification (denominator = number of valid). hit_rate_req = auxiliary reference value (denominator = requested number 250, exclusions are counted as misses)."
          "Which of the two is used in the main table of the paper is decided by the author.",
          "- Reference: beta_TS over the valid rows of all 7 targets (including the 2 extrapolation points. Not the main aggregation):"]
    for arm, w in pairs:
        L.append(f"  - {arm} w={w}: {fmt_ci(summary[f'{arm}_w{w}']['all7_beta_ts_reference_only'])}")
    L += ["", "## Per target (7 targets. Including the 2 extrapolation points)", "",
          "| arm | w | target_id | target gamma | valid | excl_err | excl_non_al | exclusion rate | median of gamma_pre [Q1, Q3] | median of the error | median of |error| | hit_rate_req (/50) | hit_rate_valid |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in per_target:
        def g(x, p=4):
            return "n/a" if x is None else format(x, f".{p}f")
        L.append(f"| {r['arm']} | {r['w']} | {r['target_id']} ({r['label']}) | {r['gamma_target']:.4f} | {r['n_valid']} | {r['n_excluded_error']} | "
                 f"{r['n_excluded_non_al']} | {r['excluded_rate']:.2f} | {g(r['gamma_pre_median'])} [{g(r['gamma_pre_q1'])}, {g(r['gamma_pre_q3'])}] | "
                 f"{g(r['err_median'])} | {g(r['abs_err_median'])} | {r['hit_rate_req']:.2f} | {g(r['hit_rate_valid'], 2)} |")
    L += ["", "## Breakdown of exclusions and C-S", "",
          "| arm | w | valid | excluded_error (breakdown) | excluded_non_al | C-S | angle_fail | nonorthogonal | nonfinite | min_dist_A (min / p5 / median) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, w in pairs:
        s = summary[f"{arm}_w{w}"]
        md = s["min_dist_A"]
        mds = "n/a" if md.get("n", 0) == 0 else f"{md['min']:.3f} / {md['p5']:.3f} / {md['median']:.3f}"
        c = s["classification"]
        L.append(f"| {arm} | {w} | {c['valid']} | {c['excluded_error']} {s['excluded_error_breakdown']} | {c['excluded_non_al']} | "
                 f"{s['CS']['verdict']} ({s['CS']['n_excluded_non_al']}/350; threshold: FLAG for >= {S_FLAG_MIN}) | "
                 f"{s['n_angle_fail']} | {s['n_nonorthogonal']} | {s['n_nonfinite']} | {mds} |")
    flags = [f"{a} w={w}" for a, w in pairs if summary[f"{a}_w{w}"]["CS"]["verdict"] == "FLAG"]
    nonfin = [f"{a} w={w}: {summary[f'{a}_w{w}']['n_nonfinite']} rows" for a, w in pairs if summary[f"{a}_w{w}"]["n_nonfinite"] > 0]
    L += ["", f"- C-S FLAG: {flags if flags else 'none'} (for a FLAG the fraction is reported to the author. The aggregation has been continued with the exclusion)",
          f"- nonfinite (NaN etc.): {nonfin if nonfin else 'none'} (even for one row, the count and (arm, w, chunk) are reported to the author)", ""]
    L += ["## Distribution of the number of atoms (350 structures, count per n)", ""]
    for arm, w in pairs:
        L.append(f"- {arm} w={w}: {summary[f'{arm}_w{w}']['natoms_distribution']}")
    L += ["", "## gb_dx_realized_A of proposed (valid rows)", ""]
    for arm, w in pairs:
        if arm == "proposed":
            L.append(f"- w={w}: {summary[f'proposed_w{w}']['gb_dx_realized_A']}")
    L += ["", "## Reference row (earlier data; not for the paper)", "",
          ('- Reference value of a model trained on earlier data (not used in the paper; w=2.0, 500 structures at 10 levels, a different number of targets): beta_TS 0.934 [0.874, 0.993], hit rate 0.294 (147/500).'), "",
          f"Range of the training data (min and max of `eval_targets.json`): {targets_j['min']:.4f} to {targets_j['max']:.4f} J/m^2 (gray vertical band in the figure).", ""]
    path.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", type=Path, default=RUN_REF / "results" / "eval")
    ap.add_argument("--out-dir", type=Path, default=RUN_REF / "results")
    ap.add_argument("--pairs", default="", help="for validation: give only some, like 'proposed:1.0,baseline:2.0' (takes precedence over --ws. The grid figure is made only when all combinations of 2 arms x the given w are present)")
    ap.add_argument("--ws", default="", help="comma-separated w to aggregate (example '1.0,2.0,3.0,4.0'). A given w is required for both arms."
                    "If not given, the w for which <model>_w<W>/ exists in <eval-root> are taken automatically (FATAL if there is a w for which not both arms are present)")
    args = ap.parse_args()

    if args.pairs:
        pairs = [tuple(x.split(":")) for x in args.pairs.split(",")]
        for a, w in pairs:
            if a not in ARMS or w not in WS_ALL:
                print(f"FATAL: bad --pairs entry {a}:{w}", file=sys.stderr)
                return 2
    else:
        if args.ws:
            ws = [x.strip() for x in args.ws.split(",") if x.strip()]
            bad = [w for w in ws if w not in WS_ALL]
            if bad or not ws or len(set(ws)) != len(ws):
                print(f"FATAL: bad --ws '{args.ws}' (allowed: {WS_ALL}, no duplicates)", file=sys.stderr)
                return 2
        else:
            ws = []
            for w in WS_ALL:
                have = [(args.eval_root / f"{a}_w{w}").is_dir() for a in ARMS]
                if all(have):
                    ws.append(w)
                elif any(have):
                    print(f"FATAL: w={w}: only one arm has an eval dir under {args.eval_root} (partial; pass --ws or --pairs explicitly)", file=sys.stderr)
                    return 2
            if not ws:
                print(f"FATAL: no <model>_w<W>/ dirs found under {args.eval_root}", file=sys.stderr)
                return 2
        pairs = [(a, w) for a in ARMS for w in ws]
    ws_in_pairs = sorted({w for _, w in pairs}, key=float)
    grid_ws = ws_in_pairs if set(pairs) == {(a, w) for a in ARMS for w in ws_in_pairs} else None
    print(f"pairs: {pairs} grid_ws={grid_ws}")

    out_dir = args.out_dir
    paths = {"summary": out_dir / "tracking_summary.json", "per_target": out_dir / "tracking_per_target.csv", "report": out_dir / "tracking_report.md"}
    for p in paths.values():
        if p.exists():
            print(f"FATAL: output exists (no overwrite): {p}", file=sys.stderr)
            return 2
    figs_dir = out_dir / "figs"
    try:
        tj = load_targets()
        targets = [float(x) for x in tj["targets"]]
        # first check validation.json of all 6 (or the given combinations). If even one is missing or not PASS, exit without writing anything
        for a, w in pairs:
            require_g63v_pass(args.eval_root / f"{a}_w{w}" / "g63v.json")
        dfs = {(a, w): load_pair(args.eval_root, a, w, targets) for a, w in pairs}
    except AggError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
    if any(figs_dir.glob("tracking_*")):
        print(f"FATAL: figure outputs already exist in {figs_dir} (no overwrite)", file=sys.stderr)
        return 2

    per_pair = {(a, w): summarize_pair(dfs[(a, w)], a, w, targets) for a, w in pairs}
    per_target = []
    for a, w in pairs:
        per_target += per_target_rows(dfs[(a, w)], a, w, targets)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {f"{a}_w{w}": per_pair[(a, w)] for a, w in pairs}
    with open(paths["summary"], "x") as f:
        json.dump(summary, f, indent=2)
    cols = list(per_target[0].keys())
    with open(paths["per_target"], "x", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(per_target)
    made = make_figs(per_pair, dfs, targets, (float(tj["min"]), float(tj["max"])), figs_dir, grid_ws)
    write_report(paths["report"], summary, per_target, made, pairs, tj)

    for a, w in pairs:
        s = per_pair[(a, w)]
        im = s["inner_main"]
        c = s["classification"]
        print(f"{a} w={w}: valid={c['valid']} excl_error={c['excluded_error']} excl_non_al={c['excluded_non_al']} "
              f"beta_ts={fmt_ci(im['beta_ts'])} hit_valid={fmt_rate(im['hit_rate_valid'])} hit_req={fmt_rate(im['hit_rate_req'])} "
              f"mae={im['mae']} CS={s['CS']['verdict']}")
    print(f"per_target rows={len(per_target)} figs={len(made)} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
