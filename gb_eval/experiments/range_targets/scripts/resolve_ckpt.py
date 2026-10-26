#!/usr/bin/env python3
"""resolve_ckpt.py -- resolution and freezing of the checkpoint with the best val_loss.

  python resolve_ckpt.py --train-dir <training dir> --arm {proposed,baseline} --out <ckpt_resolved_<model>.json>

Checks (non-zero exit if even one is not satisfied; no fallback to last.ckpt / epoch=199 / periodic):
  1a. exactly one <train-dir>/csv_logs/<run name>/version_* (run name = al_<model>_train).
  1b. checkpoints/last.ckpt exists and torch.load(...)['epoch'] == 199.
  1c. exactly 10 periodic files named epoch=<E>-periodic.ckpt with E in {19,39,...,199}.
  2.  the best ckpt = the one directly under checkpoints/ that matches ^epoch=(\\d+)-loss_val=([0-9.]+)\\.ckpt$. Exactly one.
  3.  among the rows of metrics.csv with non-empty loss_val, the epoch of the minimum row == E of 2 (if several epochs have the same value, show all and stop).
      Also torch.load(best ckpt)['epoch'] == E.
Output JSON (ckpt_resolved_<model>.json. A separate file per arm. Not overwritten if it exists):
  arm, train_dir, ckpt_path (<CLUSTER_MOUNT>/... form), epoch, loss_val_from_metrics (value of metrics.csv, not rounded),
  loss_val_in_filename, sha256 (of the whole file), size_bytes, card_hash12, resolved_at

Execution environment: needs torch (not executed on the login node). Evaluation environment (compute node).
  Run with PYTHONDONTWRITEBYTECODE=1.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

CARD_HASH12 = "bbd1307e34eb"
MAX_EPOCH_LAST = 199
PERIODIC_EPOCHS = list(range(19, 200, 20))  # 19,39,...,199 (10 files)
BEST_RE = re.compile(r"^epoch=(\d+)-loss_val=([0-9.]+)\.ckpt$")
PERIODIC_RE = re.compile(r"^epoch=(\d+)-periodic\.ckpt$")


class ResolveError(RuntimeError):
    pass


def _norm_mount(p: str) -> str:
    """Normalizes the login-node notation <CLUSTER_MOUNT> to the compute-node notation <CLUSTER_MOUNT>."""
    return re.sub(('^' + '/mnt/CLUSTER_ROOT_A'), '/CLUSTER_ROOT_B', str(p))


def _torch_epoch(path: Path) -> int:
    import torch  # lazy import (not called on the login node)

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if "epoch" not in ck:
        raise ResolveError(f"{path}: 'epoch' key missing in checkpoint")
    return int(ck["epoch"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def resolve(train_dir: Path, arm: str) -> dict:
    run_name = f"al_{arm}_train"
    csv_root = train_dir / "csv_logs" / run_name
    # 1a
    versions = sorted(p for p in csv_root.glob("version_*") if p.is_dir()) if csv_root.is_dir() else []
    if len(versions) != 1:
        raise ResolveError(f"1a: {csv_root}/version_* must be exactly 1 dir (found {len(versions)}: {[v.name for v in versions]})")
    ver = versions[0]
    ckdir = ver / "checkpoints"
    if not ckdir.is_dir():
        raise ResolveError(f"1a: checkpoints dir not found: {ckdir}")
    # 1b
    last = ckdir / "last.ckpt"
    if not last.is_file():
        raise ResolveError(f"1b: last.ckpt not found: {last}")
    last_ep = _torch_epoch(last)
    if last_ep != MAX_EPOCH_LAST:
        raise ResolveError(f"1b: last.ckpt epoch={last_ep} != {MAX_EPOCH_LAST} (training not finished)")
    # 1c
    names = sorted(p.name for p in ckdir.iterdir() if p.is_file())
    periodic = [n for n in names if n.endswith("-periodic.ckpt")]
    pe = []
    for n in periodic:
        m = PERIODIC_RE.match(n)
        if m is None:
            raise ResolveError(f"1c: unexpected periodic file name {n!r}; actual names in {ckdir}: {names}")
        pe.append(int(m.group(1)))
    if sorted(pe) != PERIODIC_EPOCHS:
        raise ResolveError(f"1c: periodic epochs {sorted(pe)} != expected {PERIODIC_EPOCHS}; actual names: {names}")
    # 2
    cands = [n for n in names if BEST_RE.match(n)]
    if len(cands) != 1:
        raise ResolveError(f"2: best-ckpt candidates must be exactly 1 (found {len(cands)}: {cands}); all names: {names}")
    best_name = cands[0]
    m = BEST_RE.match(best_name)
    e_name = int(m.group(1))
    loss_in_name = m.group(2)
    best = ckdir / best_name
    # 3
    metrics = ver / "metrics.csv"
    if not metrics.is_file():
        raise ResolveError(f"3: metrics.csv not found: {metrics}")
    rows = []
    with open(metrics, newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("loss_val") or "").strip()
            if v != "":
                rows.append((float(v), v, int(float(r["epoch"]))))
    if not rows:
        raise ResolveError(f"3: no non-empty loss_val rows in {metrics}")
    vmin = min(r[0] for r in rows)
    min_epochs = sorted({r[2] for r in rows if r[0] == vmin})
    if len(min_epochs) != 1:
        raise ResolveError(f"3: loss_val minimum {vmin!r} is attained at multiple epochs {min_epochs} (tie; refusing to pick)")
    e_metrics = min_epochs[0]
    loss_str = next(r[1] for r in rows if r[0] == vmin)
    if e_metrics != e_name:
        raise ResolveError(f"3: metrics.csv min-loss epoch={e_metrics} (loss_val={vmin!r}) != best ckpt filename epoch={e_name} ({best_name})")
    e_ckpt = _torch_epoch(best)
    if e_ckpt != e_name:
        raise ResolveError(f"3: torch.load(best)['epoch']={e_ckpt} != filename epoch={e_name} ({best_name})")
    if abs(float(loss_in_name) - vmin) > 0.0051:
        print(f"WARN: loss_val in filename ({loss_in_name}) differs from metrics.csv ({vmin!r}) by more than 0.0051", file=sys.stderr)
    return {
        "arm": arm,
        "train_dir": _norm_mount(train_dir),
        "ckpt_path": _norm_mount(best.resolve()),
        "epoch": e_name,
        "loss_val_from_metrics": float(loss_str),
        "loss_val_in_filename": loss_in_name,
        "sha256": _sha256(best),
        "size_bytes": best.stat().st_size,
        "card_hash12": CARD_HASH12,
        "resolved_at": datetime.datetime.now().astimezone().isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True, type=Path)
    ap.add_argument("--arm", required=True, choices=["proposed", "baseline"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        print(f"FATAL: --out exists (no overwrite): {args.out}", file=sys.stderr)
        return 2
    try:
        res = resolve(args.train_dir, args.arm)
    except ResolveError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x") as f:  # exception if it exists
        json.dump(res, f, indent=2)
        f.write("\n")
    print(json.dumps(res, indent=2))
    print(f"RESOLVED arm={args.arm} epoch={res['epoch']} ckpt={res['ckpt_path']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
