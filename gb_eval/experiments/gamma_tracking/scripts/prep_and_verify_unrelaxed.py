#!/usr/bin/env python3
"""Preprocessing of the calculation of gamma before relaxation (unzipping) and post-check (detection of false success).

In the 4 conditions of an earlier analysis (c26_e199 / c38_e199 / c26_ruleS / c38_ruleS) the generated CIF files are left as
`generated_crystals_cif.zip` and `cif/` was not expanded.
On the other hand `compute_gamma_unrelaxed.py` reads `cif/gen_{i}.cif` directly, so running it without
expanding gives a **false success in which every file raises FileNotFoundError but the CSV and the DONE marker alone
are written** (this actually happened with arm N;
n_ok=0/n_err=1650).

This script closes these two holes:

  extract : expands the zip into `cif/` directly under each chunk (idempotent; skipped if it exists)
  verify  : reads the output CSV and checks that **n_ok > 0 and the number of rows agrees with the expected value**
            (`compute_gamma_unrelaxed.py` does not have this guard)

Does not depend on the `unzip` command (it was missing on compute nodes).
Uses only the standard library `zipfile`.

[Source of the method]
  experiments/dx_gamma_unrelaxed/scripts/unzip_arm_n_cif.py
  (a verified implementation for the same problem. ROOT is hard-coded there, so it is generalized in this script.
    The source file is not changed)

[Files not touched]
This script exists only under a new directory and does not change the Python files that waiting jobs
read at run time (p46_*.py / p51_*.py / summarize_gamma_following_arm_w.py /
run_step_d_relax_v2.py / scripts/config/config.py / runner_guards.py etc.) at all.
Existing zip, extxyz and generation_metrics.json files are not changed or deleted either (only `cif/` is newly created).

Usage:
  python prep_and_verify_unrelaxed.py extract --generated-root <root> [--expect-chunks 20] [--expect-per-chunk 25]
  python prep_and_verify_unrelaxed.py verify  --csv <out.csv> --expect-rows 500
"""
from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

ZIP_NAME = "generated_crystals_cif.zip"


def cmd_extract(args: argparse.Namespace) -> int:
    root = Path(args.generated_root)
    if not root.is_dir():
        print(f"FATAL: generated root not found: {root}", file=sys.stderr)
        return 2

    chunks = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("chunk"))
    if not chunks:
        print(f"FATAL: no chunk dirs under {root}", file=sys.stderr)
        return 2
    if args.expect_chunks is not None and len(chunks) != args.expect_chunks:
        print(
            f"FATAL: chunk count {len(chunks)} != expected {args.expect_chunks} under {root}",
            file=sys.stderr,
        )
        return 2

    total = 0
    for cdir in chunks:
        zip_path = cdir / ZIP_NAME
        cif_dir = cdir / "cif"
        if not zip_path.is_file():
            print(f"FATAL: missing zip: {zip_path}", file=sys.stderr)
            return 2

        if cif_dir.exists():
            n_existing = len(list(cif_dir.glob("*.cif")))
            if args.expect_per_chunk is not None and n_existing != args.expect_per_chunk:
                print(
                    f"FATAL: {cdir.name}: existing cif/ has {n_existing} files "
                    f"!= expected {args.expect_per_chunk}. refusing to proceed "
                    "(partial extraction from an earlier interrupted run)",
                    file=sys.stderr,
                )
                return 2
            print(f"{cdir.name}: cif/ already present (n={n_existing}), skip")
            total += n_existing
            continue

        cif_dir.mkdir(parents=False, exist_ok=False)
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".cif")]
            for name in names:
                # the zip is assumed to contain flat gen_i.cif. Entries containing a path separator are rejected (zip-slip protection)
                if "/" in name or "\\" in name or name.startswith(".."):
                    print(f"FATAL: unexpected path in zip: {zip_path}!{name}", file=sys.stderr)
                    return 2
                (cif_dir / name).write_bytes(zf.read(name))
        n = len(names)
        if args.expect_per_chunk is not None and n != args.expect_per_chunk:
            print(
                f"FATAL: {cdir.name}: extracted {n} cif != expected {args.expect_per_chunk}",
                file=sys.stderr,
            )
            return 2
        print(f"{cdir.name}: extracted {n} cif")
        total += n

    print(f"EXTRACT_OK root={root} chunks={len(chunks)} total_cif={total}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Applies from outside the n_ok>0 guard that compute_gamma_unrelaxed.py does not have."""
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"FAIL V0: output csv not found: {csv_path}", file=sys.stderr)
        return 1

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n_rows = len(rows)
    gcol = "gamma_unrelaxed"
    if n_rows and gcol not in rows[0]:
        print(f"FAIL V1: column {gcol!r} missing. columns={list(rows[0])}", file=sys.stderr)
        return 1

    def _is_ok(r: dict) -> bool:
        err = (r.get("error") or "").strip()
        val = (r.get(gcol) or "").strip()
        if err:
            return False
        if not val:
            return False
        try:
            float(val)
        except ValueError:
            return False
        return True

    n_ok = sum(1 for r in rows if _is_ok(r))
    n_err = n_rows - n_ok

    print(f"[V1] rows={n_rows} (expected {args.expect_rows})")
    print(f"[V2] n_ok={n_ok} n_err={n_err}")

    verdict_ok = True
    if n_rows != args.expect_rows:
        print(f"FAIL V1: row count {n_rows} != expected {args.expect_rows}", file=sys.stderr)
        verdict_ok = False
    if n_ok == 0:
        print(
            "FAIL V2: n_ok == 0. This is the same symptom as the false success "
            "(the CSV and the DONE marker are written but all contents are errors).",
            file=sys.stderr,
        )
        verdict_ok = False
    if n_ok != n_rows:
        # it is not fatal if not everything succeeded, but it is always made visible (not silently ignored)
        print(f"WARN V3: {n_err} rows have errors or empty {gcol}", file=sys.stderr)

    print(f"VERDICT: {'PASS' if verdict_ok else 'FAIL'} csv={csv_path}")
    return 0 if verdict_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="expand generated_crystals_cif.zip into cif/")
    p_ex.add_argument("--generated-root", required=True)
    p_ex.add_argument("--expect-chunks", type=int, default=None)
    p_ex.add_argument("--expect-per-chunk", type=int, default=None)
    p_ex.set_defaults(func=cmd_extract)

    p_ve = sub.add_parser("verify", help="check n_ok>0 and the number of rows of the output CSV")
    p_ve.add_argument("--csv", required=True)
    p_ve.add_argument("--expect-rows", type=int, required=True)
    p_ve.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
