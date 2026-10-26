#!/usr/bin/env python3
"""S5: rerun stage E-1 (template split) on the released gamma_dedup.csv and compare with the released split.json.

  SB_WORK=<new work dir> python s5_split_compare.py --released $GBREL_DATA/training --out s5_compare.txt
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SB_WORK = Path(os.environ["SB_WORK"])
SB = Path(__file__).resolve().parents[1]
KEYS = ["train_indices", "val_indices", "test_indices", "val_dx_indices", "n_train", "n_val", "n_test", "n_val_dx", "templates_train",
        "templates_val", "templates_test", "seed", "n_templates", "template_counts", "split_rule", "permutation", "cnid_0_0_side", "n_rows"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.released)
    res = SB_WORK / "results"
    res.mkdir(parents=True, exist_ok=True)
    shutil.copy(rel / "gamma_dedup.csv", res / "gamma_dedup.csv")
    names = sorted(p.stem for p in (rel / "manifests").glob("cnid_*.json"))
    (res / "template_map.csv").write_text("cnid\n" + "\n".join(names) + "\n")
    env = dict(os.environ, SB_WORK=str(SB_WORK))
    cmd = [sys.executable, str(SB / "stageE_split" / "stage_e_split.py"), "--results-dir", str(res)]
    print("S5 run:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    print("S5 stage_e_split exit code:", rc)
    new = json.load(open(res / "split.json"))
    old = json.load(open(rel / "split.json"))
    lines = []
    n_same = 0
    for k in KEYS:
        same = (k in new and k in old and new[k] == old[k])
        n_same += int(same)
        lines.append("%s\t%s\t(released has: %s, rebuilt has: %s)" % (k, "identical" if same else "DIFFERENT", k in old, k in new))
    extra = sorted(set(old) - set(KEYS))
    lines.append("keys only in the released file (not compared, path/hash fields): %s" % extra)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("S5 compared keys=%d identical=%d" % (len(KEYS), n_same))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
