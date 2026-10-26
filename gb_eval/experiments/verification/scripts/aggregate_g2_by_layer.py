#!/usr/bin/env python
"""aggregate_g2_by_layer.py -- makes results/regeneration/g2_by_layer.csv from the per-layer g2_and_e JSON.

Standard library only. 1 row = (layer, arm). The set and order of layers and arms are defined only by layers.json.
Columns: layer, arm, seed_base, n_req, k_g2_post, k_g2_pre, k_e_post, wilson_lo, wilson_hi,
    k_g2_by_target_0 .. k_g2_by_target_6
- k_g2_post / k_e_post / wilson_lo / wilson_hi: stage=post of rows[]
- k_g2_pre: g2 of stage=pre of rows[]
- k_g2_by_target_t: g2 of stage=post, target_id=t of per_target[]
All checks are explicit exceptions, not asserts (they do not disappear with python -O). No decision is made.
If the output CSV already exists it stops without overwriting (open mode 'x').
"""
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)
import csv
import io
import json
import os
import sys

VRUN = _gbrel.WORK + '/workspace/experiments/verification'
LAYERS_JSON = VRUN + "/config/layers.json"
G2_AND_E_DIR = VRUN + "/results/gb_equivalence"
OUT_CSV = VRUN + "/results/regeneration/g2_by_layer.csv"

# layer name -> g2_and_e JSON file name (the set of layer names must agree exactly with layers.json)
G2_AND_E_FILES = {
    "main": "gb_symfit_main_g2_and_e.json",
    "baseline_regen1": "gb_symfit_baseline_regen1_g2_and_e.json",
    "baseline_regen2": "gb_symfit_baseline_regen2_g2_and_e.json",
    "baseline_regen3": "gb_symfit_baseline_regen3_g2_and_e.json",
    "proposed_regen1": "gb_symfit_proposed_regen1_g2_and_e.json",
    "proposed_regen2": "gb_symfit_proposed_regen2_g2_and_e.json",
}

N_REQ = 350
N_REQ_PER_TARGET = 50
N_TARGETS = 7
W_ONLY = "2.0"
THETA = 0.064

COLUMNS = (
    ["layer", "arm", "seed_base", "n_req", "k_g2_post", "k_g2_pre", "k_e_post",
     "wilson_lo", "wilson_hi"]
    + ["k_g2_by_target_%d" % t for t in range(N_TARGETS)]
)


def fail(msg):
    raise SystemExit("ERROR: " + msg)


def need(d, key, where):
    if not isinstance(d, dict) or key not in d:
        fail("required key '%s' is missing (%s)" % (key, where))
    return d[key]


def pick_row(rows, arm, stage, where):
    hit = [r for r in rows if r.get("arm") == arm and r.get("stage") == stage]
    if len(hit) != 1:
        fail("rows[] has %d entries with arm=%s stage=%s (expected 1) (%s)" % (len(hit), arm, stage, where))
    return hit[0]


def build_rows(layer, spec, g2_and_e_path):
    with open(g2_and_e_path) as fh:
        d = json.load(fh)
    where = "%s [%s]" % (layer, g2_and_e_path)
    if need(d, "layer", where) != layer:
        fail("layer=%r of the JSON does not agree with the layer name %r (%s)" % (d["layer"], layer, where))
    if float(need(d, "theta", where)) != THETA:
        fail("theta=%r (expected %r) (%s)" % (d["theta"], THETA, where))
    rows = need(d, "rows", where)
    per_target = need(d, "per_target", where)
    arms = list(need(spec, "arms", "layers.json:" + layer))
    seed_base = need(spec, "seed_base", "layers.json:" + layer)

    # check of arm / w: all rows in the JSON belong to the arms of the layer, and w is only 2.0
    for r in rows + per_target:
        if need(r, "arm", where) not in arms:
            fail("arm=%r is not in arms=%s of layer %s (%s)" % (r["arm"], layer, arms, where))
        if str(need(r, "w", where)) != W_ONLY:
            fail("w=%r is other than %s (%s)" % (r["w"], W_ONLY, where))

    out = []
    for arm in arms:
        post = pick_row(rows, arm, "post", where)
        pre = pick_row(rows, arm, "pre", where)
        for stage_name, r in (("post", post), ("pre", pre)):
            if need(r, "n_req", where) != N_REQ:
                fail("n_req=%r != %d (%s arm=%s stage=%s)" % (r["n_req"], N_REQ, where, arm, stage_name))
            sc = need(r, "status_counts", where)
            if need(sc, "total", where) != N_REQ:
                fail("status_counts.total=%r != %d (%s arm=%s stage=%s)" % (sc["total"], N_REQ, where, arm, stage_name))
        wil = need(post, "g2_wilson", where)
        by_t = []
        for t in range(N_TARGETS):
            hit = [p for p in per_target
                   if p["arm"] == arm and p["stage"] == "post" and p["target_id"] == t]
            if len(hit) != 1:
                fail("per_target has %d entries with arm=%s stage=post target_id=%d (expected 1) (%s)" % (len(hit), arm, t, where))
            if need(hit[0], "n_req", where) != N_REQ_PER_TARGET:
                fail("per_target n_req=%r != %d (%s arm=%s t=%d)" % (hit[0]["n_req"], N_REQ_PER_TARGET, where, arm, t))
            by_t.append(need(hit[0], "g2", where))
        if sum(by_t) != post["g2"]:
            fail("sum(k_g2_by_target)=%d != k_g2_post=%d (%s arm=%s)" % (sum(by_t), post["g2"], where, arm))
        row = {
            "layer": layer, "arm": arm, "seed_base": seed_base, "n_req": post["n_req"],
            "k_g2_post": post["g2"], "k_g2_pre": pre["g2"], "k_e_post": post["e"],
            "wilson_lo": need(wil, "lo", where), "wilson_hi": need(wil, "hi", where),
        }
        for t in range(N_TARGETS):
            row["k_g2_by_target_%d" % t] = by_t[t]
        out.append(row)
    return out


def main():
    if os.path.exists(OUT_CSV):
        fail("the output already exists (not overwritten): " + OUT_CSV)
    with open(LAYERS_JSON) as fh:
        layers = need(json.load(fh), "layers", LAYERS_JSON)
    if set(layers) != set(G2_AND_E_FILES):
        fail("the set of layers %s of layers.json does not agree with G2_AND_E_FILES %s" % (sorted(layers), sorted(G2_AND_E_FILES)))
    all_rows = []
    for layer in layers:  # the order of the description in layers.json
        all_rows.extend(build_rows(layer, layers[layer], os.path.join(G2_AND_E_DIR, G2_AND_E_FILES[layer])))
    if len(all_rows) != 7:
        fail("number of rows %d (expected 7)" % len(all_rows))
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="raise", lineterminator="\n")
    wr.writeheader()
    for r in all_rows:
        wr.writerow(r)
    with open(OUT_CSV, "x") as fh:  # 'x': FileExistsError if it exists
        fh.write(buf.getvalue())
    print("wrote %s (%d rows)" % (OUT_CSV, len(all_rows)))
    sys.stdout.write(buf.getvalue())


if __name__ == "__main__":
    main()
