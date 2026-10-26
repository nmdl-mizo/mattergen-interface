#!/usr/bin/env python3
"""check_baseline_config.py -- supplement of the generation of the baseline: reads the training config of the baseline (csv_logs/config.yaml) as YAML and checks it.

Called by sbatch_gen_baseline.sh on compute nodes (DRYRUN=0). Only the DRYRUN on the login node substitutes grep (no YAML library there and <CLUSTER_MOUNT> is not visible).
Checks (rc=1 if even one is not satisfied):
  1. Cache name: all string values that contain /datasets/cache/ are directly under /datasets/cache/<--cache-name> (at the end or followed by '/'), and there is at least one.
  2. num_atoms_distribution_key == --num-atoms-key。
  3. property_embeddings_adapt has gb_gamma_J_m2 (a conditional model).
  4. data_module.average_density agrees with 0.05839622542107227 as a float within abs difference 1e-15 (comparison of the arms in the model specification: "the same density for proposed and baseline".
     For the baseline it affects the prior of the lattice at generation. The location was confirmed in conf/data_module/scm_al_baseline.yaml:28 and data_module.average_density of the training config).
Execution environment: the python of the generation environment (${GEN_PY}) (with PyYAML). PYTHONDONTWRITEBYTECODE=1.
"""
from __future__ import annotations

import argparse
import re
import sys

import yaml

sys.dont_write_bytecode = True


AVERAGE_DENSITY = 0.05839622542107227
DENSITY_ABS_TOL = 1e-15


def walk(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, x


def has_gb_gamma(x) -> bool:
    if isinstance(x, dict):
        pe = x.get("property_embeddings_adapt")
        if isinstance(pe, dict) and "gb_gamma_J_m2" in pe:
            return True
        return any(has_gb_gamma(v) for v in x.values())
    if isinstance(x, list):
        return any(has_gb_gamma(v) for v in x)
    return False


def check(cfg, cache_name: str, key: str):
    problems = []
    cache_strs = [(p, v) for p, v in walk(cfg) if isinstance(v, str) and "/datasets/cache/" in v]
    pat = re.compile(r"/datasets/cache/" + re.escape(cache_name) + r"(/|$)")
    if not cache_strs:
        problems.append("no string value contains /datasets/cache/")
    bad = [(p, v) for p, v in cache_strs if not pat.search(v)]
    if bad:
        problems.append(f"cache name != {cache_name} in {len(bad)} value(s): {bad[:3]}")
    if cfg.get("num_atoms_distribution_key") != key:
        problems.append(f"num_atoms_distribution_key={cfg.get('num_atoms_distribution_key')!r} != {key!r}")
    dm = cfg.get("data_module")
    dens = dm.get("average_density") if isinstance(dm, dict) else None
    try:
        dens_ok = dens is not None and abs(float(dens) - AVERAGE_DENSITY) <= DENSITY_ABS_TOL
    except (TypeError, ValueError):
        dens_ok = False
    if not dens_ok:
        problems.append(f"data_module.average_density={dens!r} != {AVERAGE_DENSITY!r} (abs tol {DENSITY_ABS_TOL:g})")
    if not has_gb_gamma(cfg):
        problems.append("property_embeddings_adapt has no gb_gamma_J_m2 (not a conditional model?)")
    return problems, len(cache_strs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-name", required=True)
    ap.add_argument("--num-atoms-key", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    problems, n_cache = check(cfg, args.cache_name, args.num_atoms_key)
    print(f"RESOLVED train : num_atoms_distribution_key={cfg.get('num_atoms_distribution_key')} (from {args.config})")
    if problems:
        for p in problems:
            print(f"CONFIG_CHECK: NG {p}", file=sys.stderr)
        return 1
    print(f"CONFIG_CHECK: OK (YAML) cache name = {args.cache_name} in all {n_cache} cache path value(s), num_atoms_distribution_key={args.num_atoms_key}, gb_gamma_J_m2 present, average_density={cfg['data_module']['average_density']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
