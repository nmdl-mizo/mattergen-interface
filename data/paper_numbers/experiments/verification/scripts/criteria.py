#!/usr/bin/env python3
"""criteria.py -- decision functions (C1 / P1 / C3 / P3 / C4 / P4 / C2 and the composition of the 10/20 decision).

Plan: see the experiment description (frozen before the analysis).
Standard library only. Takes counts and aggregated values as input and returns a verdict (PASS/FAIL/FLAG) and the basis of the decision. Thresholds and rules are as frozen and are not changed after seeing the results.
  - Wilson 95%: z = 1.959964 (same value as summarize_g2_and_e.py). Comparison of point estimates is exact with Fraction (0.30, 0.05 and 0.95 are rational numbers). The comparison of the Wilson lower limit is in float.
  - One-sided binomial (C2): exact with fractions.Fraction.
The decisions are checked mechanically by a separate check. This module neither re-runs nor changes thresholds.
"""
from __future__ import annotations

import math
from fractions import Fraction

Z95 = 1.959964
N_REQ = 350

# C1
G1_P_MAX = Fraction(3, 10)          # 0.30
G1_BASELINE_REPS = ("baseline_regen1", "baseline_regen2", "baseline_regen3")
G1_PROPOSED_REPS = ("proposed_regen1", "proposed_regen2")
# C3
G3_Q_MAX = Fraction(3, 10)          # q ≤ 0.30
G3_QLOSS_MIN = Fraction(1, 2)       # q_loss ≥ 0.50
G3_A_MAX = 17                       # a ≤ 17
G3_S3_MAX = 37                      # s3 ≤ 37
G3_P3_MIN = Fraction(95, 100)       # P3: proposed #two_gb_cna/350 ≥ 0.95
G3_SINGLE_ORIENT_FRAC = Fraction(9, 10)   # max(frac_A, frac_B) >= 0.9 is regarded as "single orientation at stage 0" (ruling by the user; the reviewer's value, not calibrated)
# C4
G4_E_MAX = Fraction(1, 20)          # e_r ≤ 0.05
G4_JUDGE_THETAS = (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.032, 0.064, 0.1)
G4_THETA_TOP = 0.1
# C2
G2_ALPHA_PASS = Fraction(1, 100)    # p ≤ 0.01 → PASS
G2_ALPHA_FLAG = Fraction(5, 100)    # 0.01 < p ≤ 0.05 → FLAG、p > 0.05 → FAIL
G2_N_MIN = 7                        # n_used < 7 is FLAG
G2_DELTA_MIN = 0.04                 # pairs with |d gamma_MACE| < 0.04 J/m2 are excluded from the sign test


def wilson(k, n, z=Z95):
    """Wilson 95% interval (lo, hi). Same formula as wilson in summarize_g2_and_e."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / den
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return min(max(0.0, c - h), p), max(min(1.0, c + h), p)


def binom_sf_half(n, k):
    """Returns P(X >= k | n, 1/2) exactly as a Fraction."""
    if not (0 <= k <= n):
        raise ValueError(f"need 0 <= k <= n (k={k}, n={n})")
    return Fraction(sum(math.comb(n, i) for i in range(k, n + 1)), 2 ** n)


def _check_reps(got, expected, label):
    if tuple(sorted(got)) != tuple(sorted(expected)):
        raise ValueError(f"{label}: replicate set {sorted(got)} != expected {sorted(expected)} (no silent substitution by another layer)")


def _check_count(k, n, label):
    if not (isinstance(k, int) and 0 <= k <= n):
        raise ValueError(f"{label}: count {k!r} not an int in 0..{n}")


# ---------------------------------------------------------------------------
# C1 / P1
# ---------------------------------------------------------------------------
def criterion_1(baseline_k_by_rep, proposed_k_by_rep, n=N_REQ, baseline_reps=G1_BASELINE_REPS, proposed_reps=G1_PROPOSED_REPS):
    """baseline_k_by_rep: {baseline_regen1..3: k_g2_post}, proposed_k_by_rep: {proposed_regen1..2: k_g2_post}. The denominator n is fixed (exclusions are misses).
    P1: k == n (350/350) for each new proposed replicate. If it does not hold, C1 is FLAG.
    PASS: p_r <= 0.30 for all r. FAIL: the Wilson95 lower limit > 0.30 for some r. FLAG: otherwise.
    (When P1 does not hold, it is FLAG even if the FAIL condition is satisfied.)"""
    _check_reps(baseline_k_by_rep, baseline_reps, "C1 baseline")
    _check_reps(proposed_k_by_rep, proposed_reps, "P1 proposed")
    for r, k in {**baseline_k_by_rep, **proposed_k_by_rep}.items():
        _check_count(k, n, r)
    p1 = all(proposed_k_by_rep[r] == n for r in proposed_reps)
    per = {}
    for r in baseline_reps:
        k = baseline_k_by_rep[r]
        lo, hi = wilson(k, n)
        per[r] = {"k": k, "n": n, "p": k / n, "wilson_lo": lo, "wilson_hi": hi,
                  "p_le_0.30": Fraction(k, n) <= G1_P_MAX, "lo_gt_0.30": lo > float(G1_P_MAX)}
    all_pass = all(v["p_le_0.30"] for v in per.values())
    any_fail = any(v["lo_gt_0.30"] for v in per.values())
    if not p1:
        verdict = "FLAG"
    elif any_fail:
        verdict = "FAIL"
    elif all_pass:
        verdict = "PASS"
    else:
        verdict = "FLAG"
    return {"gate": "C1", "verdict": verdict, "P1": p1, "per_rep": per,
            "proposed_k": dict(proposed_k_by_rep), "reason": _g1_reason(verdict, p1, per)}


def _g1_reason(verdict, p1, per):
    if not p1:
        return "P1 does not hold (the new replicates of proposed are not 350/350) -> FLAG"
    return {"PASS": "p_r <= 0.30 for all r", "FAIL": "the Wilson95 lower limit > 0.30 for some r", "FLAG": "some r has p_r > 0.30 but the Wilson lower limit is 0.30 or below"}[verdict]


# ---------------------------------------------------------------------------
# C3 / P3
# ---------------------------------------------------------------------------
def criterion_3(q_k, qloss_k, a, s3, proposed_k, u_total, u_single, n=N_REQ, n_proposed=N_REQ):
    """q_k = #two_gb_cna(baseline main), qloss_k = #loss_cna, a = #phase_ambiguous(baseline main), s3 = among the 74 baseline structures with two_gb at stage 0, the number whose CNA n_slab >= 3,
    proposed_k = #two_gb_cna(proposed main), u_total = |U| (U = two_gb_cna AND gb0_status=no_gb at stage 0 in baseline main),
    u_single = number of U with max(frac_A, frac_B) >= 0.9.
    FAIL candidate: P3 holds AND Wilson95 lower limit(q) > 0.30. For a candidate: if |U|=0 then FAIL. If u_single > |U|/2 then FLAG (with diagnosis), otherwise FAIL.
    PASS: q <= 0.30 AND q_loss >= 0.50 AND a <= 17 AND s3 <= 37 AND P3 holds. FLAG: otherwise."""
    for lab, k, nn in (("q_k", q_k, n), ("qloss_k", qloss_k, n), ("proposed_k", proposed_k, n_proposed), ("a", a, n), ("s3", s3, n)):
        _check_count(k, nn, lab)
    if not (isinstance(u_total, int) and isinstance(u_single, int) and 0 <= u_single <= u_total):
        raise ValueError(f"U counts invalid: u_total={u_total!r} u_single={u_single!r}")
    p3 = Fraction(proposed_k, n_proposed) >= G3_P3_MIN
    lo, hi = wilson(q_k, n)
    q_le = Fraction(q_k, n) <= G3_Q_MAX
    qloss_ge = Fraction(qloss_k, n) >= G3_QLOSS_MIN
    a_ok, s3_ok = a <= G3_A_MAX, s3 <= G3_S3_MAX
    fail_candidate = bool(p3 and lo > float(G3_Q_MAX))
    detail = {"q": q_k / n, "q_k": q_k, "q_wilson_lo": lo, "q_wilson_hi": hi, "q_le_0.30": q_le,
              "q_loss": qloss_k / n, "qloss_k": qloss_k, "q_loss_ge_0.50": qloss_ge,
              "a": a, "a_le_17": a_ok, "s3": s3, "s3_le_37": s3_ok,
              "P3": p3, "proposed_k": proposed_k, "proposed_rate": proposed_k / n_proposed,
              "fail_candidate": fail_candidate, "U_total": u_total, "U_single_orient_ge_0.9": u_single}
    if fail_candidate:
        if u_total == 0:
            verdict, reason = "FAIL", "FAIL candidate AND |U|=0 -> FAIL (no explanation other than an oversight at stage 0)"
        elif u_single * 2 > u_total:
            verdict, reason = "FLAG", (f"FAIL candidate, but the single orientation of U (max(frac_A,frac_B)>=0.9) is the majority ({u_single}/{u_total}) -> "
                                      "suspicion that CNA counted a planar defect of a single-orientation crystal as a grain boundary (FLAG with diagnosis)")
        else:
            verdict, reason = "FAIL", f"FAIL candidate AND the single orientation of U is not the majority ({u_single}/{u_total}) -> suspicion that stage 0 overlooked the second orientation"
    elif q_le and qloss_ge and a_ok and s3_ok and p3:
        verdict, reason = "PASS", "q <= 0.30 AND q_loss >= 0.50 AND a <= 17 AND s3 <= 37 AND P3 holds"
    else:
        bad = [name for name, ok in (("q≤0.30", q_le), ("q_loss≥0.50", qloss_ge), ("a≤17", a_ok), ("s3≤37", s3_ok), ("P3", p3)) if not ok]
        verdict, reason = "FLAG", "PASS condition not satisfied: " + ",".join(bad) + " (not a FAIL candidate)"
    return {"gate": "C3", "verdict": verdict, "reason": reason, **detail}


def split_u(rows):
    """Input for the distribution of U in case of a FAIL candidate (per structure). rows: list of dict (keys: arm, two_gb_cna(bool), gb0_status, frac_A, frac_B).
    Returns: (u_total, u_single). U = two_gb_cna AND gb0_status == 'no_gb' in baseline. Single orientation = max(frac_A, frac_B) >= 0.9 (compared with Fraction)."""
    u = [r for r in rows if r["arm"] == "baseline" and r["two_gb_cna"] and r["gb0_status"] == "no_gb"]
    single = sum(1 for r in u if Fraction(repr(max(float(r["frac_A"]), float(r["frac_B"])))) >= G3_SINGLE_ORIENT_FRAC)
    return len(u), single


# ---------------------------------------------------------------------------
# C4 / P4
# ---------------------------------------------------------------------------
def criterion_4(baseline_e_top_by_rep, proposed_by_rep, n=N_REQ, baseline_reps=G1_BASELINE_REPS, proposed_reps=G1_PROPOSED_REPS, thetas=G4_JUDGE_THETAS):
    """baseline_e_top_by_rep: {baseline_regen1..3: E_r(0.1)}(stage=post)。
    proposed_by_rep: {proposed_regen1..2: {"n_g2": number of G2, "E": {theta: E(theta)}}} (including E for all theta of the decision grid).
    P4: for each proposed rep, E(theta) = number of G2 for all theta of the decision grid (1e-4 to 0.1). If it does not hold, C4 is FLAG.
    PASS: e_r = E_r(0.1)/350 <= 0.05 for all r AND P4 holds. FAIL: Wilson95 lower limit(e_r) > 0.05 for some r. FLAG: otherwise.
    (When P does not hold, FLAG even if the FAIL condition is satisfied. No criterion for review is set; ruling by the user.)"""
    _check_reps(baseline_e_top_by_rep, baseline_reps, "C4 baseline")
    _check_reps(proposed_by_rep, proposed_reps, "P4 proposed")
    per = {}
    for r in baseline_reps:
        k = baseline_e_top_by_rep[r]
        _check_count(k, n, r)
        lo, hi = wilson(k, n)
        per[r] = {"E_0.1": k, "n": n, "e": k / n, "wilson_lo": lo, "wilson_hi": hi,
                  "e_le_0.05": Fraction(k, n) <= G4_E_MAX, "lo_gt_0.05": lo > float(G4_E_MAX)}
    p4_detail, p4 = {}, True
    for r in proposed_reps:
        h = proposed_by_rep[r]
        Ed = h["E"]
        missing = [t for t in thetas if t not in Ed]
        if missing:
            raise ValueError(f"P4 {r}: E(θ) lacks judge-grid θ {missing}")
        short = {t: h["n_g2"] - Ed[t] for t in thetas if Ed[t] != h["n_g2"]}
        p4_detail[r] = {"n_g2": h["n_g2"], "theta_with_E_ne_G2": {repr(t): d for t, d in short.items()}, "ok": not short}
        p4 = p4 and not short
    all_pass = all(v["e_le_0.05"] for v in per.values())
    any_fail = any(v["lo_gt_0.05"] for v in per.values())
    if not p4:
        verdict, reason = "FLAG", "P4 does not hold (there are theta for which E(theta) != number of G2 on the constrained side) -> FLAG (consult the user with the excess count and the residual)"
    elif any_fail:
        verdict, reason = "FAIL", "Wilson95 lower limit(e_r) > 0.05 for some r"
    elif all_pass:
        verdict, reason = "PASS", "e_r <= 0.05 for all r AND P4 holds"
    else:
        verdict, reason = "FLAG", "some r has e_r > 0.05 but the Wilson lower limit is 0.05 or below"
    return {"gate": "C4", "verdict": verdict, "reason": reason, "P4": p4, "P4_detail": p4_detail, "per_rep": per}


# ---------------------------------------------------------------------------
# C2
# ---------------------------------------------------------------------------
def criterion_2(n_used, k):
    """n_used = number of pairs with |d gamma_MACE| >= 0.04, k = number of agreements. p = P(X >= k | n_used, 1/2) (exact).
    PASS: p <= 0.01. FLAG: 0.01 < p <= 0.05, or n_used < 7 (n_used < 7 is FLAG regardless of p). FAIL: p > 0.05."""
    if not (isinstance(n_used, int) and isinstance(k, int) and 0 <= k <= n_used):
        raise ValueError(f"need ints 0 <= k <= n_used (n_used={n_used!r}, k={k!r})")
    if n_used == 0:
        return {"gate": "C2", "verdict": "FLAG", "n_used": 0, "k": 0, "p": None, "p_float": None, "reason": "n_used=0 -> FLAG (the test cannot be done)"}
    p = binom_sf_half(n_used, k)
    if n_used < G2_N_MIN:
        verdict, reason = "FLAG", f"n_used={n_used} < {G2_N_MIN} → FLAG"
    elif p <= G2_ALPHA_PASS:
        verdict, reason = "PASS", "p ≤ 0.01"
    elif p <= G2_ALPHA_FLAG:
        verdict, reason = "FLAG", "0.01 < p ≤ 0.05"
    else:
        verdict, reason = "FAIL", "p > 0.05"
    return {"gate": "C2", "verdict": verdict, "n_used": n_used, "k": k, "p": f"{p.numerator}/{p.denominator}",
            "p_float": float(p), "reason": reason}


def criterion_2_from_pairs(pairs):
    """pairs: [{"gamma_mace_lo","gamma_mace_hi","gamma_dft_lo","gamma_dft_hi", ...}]. Pairs with |d gamma_MACE| < 0.04 are excluded and reported.
    Agreement = sign(gamma_DFT,hi - gamma_DFT,lo) == sign(gamma_MACE,hi - gamma_MACE,lo)."""
    used, excluded = [], []
    for p in pairs:
        dm = p["gamma_mace_hi"] - p["gamma_mace_lo"]
        dd = p["gamma_dft_hi"] - p["gamma_dft_lo"]
        (excluded if abs(dm) < G2_DELTA_MIN else used).append((p, dm, dd))
    k = sum(1 for _p, dm, dd in used if (dm > 0) == (dd > 0) and dd != 0)
    out = criterion_2(len(used), k)
    out["n_pairs"] = len(pairs)
    out["n_excluded_small_delta"] = len(excluded)
    return out


# ---------------------------------------------------------------------------
# composition of the 10/20 decision (single master)
# ---------------------------------------------------------------------------
def compose_1020(v1, v3, v4):
    """The three of C1, C3 and C4 (C2 is not included). All PASS -> "passed"; any FAIL -> "did not pass"; no FAIL but a FLAG included -> user decision."""
    vs = [v1, v3, v4]
    if any(v not in ("PASS", "FAIL", "FLAG") for v in vs):
        raise ValueError(f"invalid verdict in {vs}")
    if "FAIL" in vs:
        return "did not pass (switch to the fallback model is recommended; the decision is made by the user)"
    if all(v == "PASS" for v in vs):
        return "passed"
    return "user decision (no FAIL; includes a FLAG)"
