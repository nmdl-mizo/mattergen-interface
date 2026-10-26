#!/usr/bin/env python3
"""gen_baseline_seeded.py -- a wrapper that gives a seed to mattergen-generate.

Background: main() of mattergen/scripts/generate.py has no seed argument (0 grep hits). The random numbers of the generation run without a seed,
so the seed of each chunk cannot be recorded and, as far as possible, cannot be reproduced. This wrapper calls seed_everything and then runs main with fire.

Usage:
  python gen_baseline_seeded.py --seed <int> -- <the same arguments as mattergen-generate...>
    Example) python gen_baseline_seeded.py --seed 20292000 -- /out/dir --model_path=/stage --checkpoint_epoch=199 ...

Determinism (whether the same structures come out with the same seed) is not claimed. The purpose is to record the seed and to reproduce as far as possible.
Environment: ${GEN_PY}, PYTHONPATH=${GBREL_CODE} (set by the sbatch script of the baseline generation).
"""
from __future__ import annotations

import argparse
import os
import sys


def _split_argv(argv: list[str]) -> tuple[int, list[str]]:
    """Decomposes argv[1:] as `--seed N -- rest...`. SystemExit(2) if "--" is missing, the seed is missing, or the seed is not an integer."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--seed", type=int, required=True)
    args = argv[1:]
    if "--" not in args:
        print("FATAL: usage: gen_baseline_seeded.py --seed <int> -- <mattergen-generate args...>", file=sys.stderr)
        raise SystemExit(2)
    i = args.index("--")
    own, rest = args[:i], args[i + 1:]
    ns, unknown = ap.parse_known_args(own)
    if unknown:
        print(f"FATAL: unexpected args before '--': {unknown}", file=sys.stderr)
        raise SystemExit(2)
    return ns.seed, rest


def main() -> None:
    seed, rest = _split_argv(sys.argv)
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl  # type: ignore[no-redef]
    pl.seed_everything(seed, workers=True)
    print(f"SEED_SET seed={seed} PL_GLOBAL_SEED={os.environ.get('PL_GLOBAL_SEED')}", flush=True)

    sys.argv = [sys.argv[0]] + rest  # fire reads sys.argv[1:]
    import fire
    from mattergen.scripts.generate import main as generate_main

    fire.Fire(generate_main)


if __name__ == "__main__":
    main()
