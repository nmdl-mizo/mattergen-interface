"""Prints the effective hyperparameters as JSON from the instantiated objects just after the start of the training.

The entry points of the proposed model (mattergen.scripts.finetune_scm_cond) and of the baseline (mattergen.scripts.finetune) both
call ``trainer.fit(model=..., datamodule=...)`` just after instantiating the trainer / datamodule / lightning_module from cfg.
This wrapper replaces ``pytorch_lightning.Trainer.fit`` and, **just before** calling the real fit, writes out the following and then proceeds to the real fit
(the code of the original entry points is not changed):

  diffusion_module.pre_corruption_fn.p_unconditional / dropout_fields_iid
  class of loss_fn, reduce, d3pm_hybrid_lambda, loss_weights (pos, atomic_numbers, cell, etc.), weights of gb_t/gb_dx, presence of freeze_atomic_numbers
  gradient_clip_val / gradient_clip_algorithm / max_epochs / check_val_every_n_epoch / limit_train_batches / accumulate_grad_batches of the trainer
  every_n_epochs / filename / save_last / save_top_k / monitor / mode of each ModelCheckpoint callback
  classes and arguments of the optimizer and scheduler (from the objects obtained by actually calling lightning_module.configure_optimizers())
  num_atoms_distribution_key (from the resolved cfg held by SaveConfigCallback) and SCM_HOST_Z (for the proposed model, the value resolved by mattergen.symmetry_constrained.constants)
  measurement of the effective seed (the PL_GLOBAL_SEED environment variable and torch.initial_seed())

The hydra config of the baseline does not show p_unconditional, the loss weights and the d3pm lambda (finetune.py takes them from the pretrained checkpoint at run time),
so they are read from the objects after instantiation here and not from the resolved config.

Usage:
  python run_with_hparam_dump.py --entry mattergen.scripts.finetune_scm_cond --kind proposed --out <json> [--seed N] [--dump-only] -- <hydra overrides...>
  --seed N: calls ``pl.seed_everything(N, workers=True)`` before the start in place of the entry point. The entry point of the proposed model reads ``+seed`` and seeds by itself,
            so this is not needed there. The entry point of the baseline (finetune.py) does not read ``+seed`` (measured by grep: finetune.py has no seed handling),
            so for the baseline this flag makes the seed effective (seeded before the instantiation, in the same order as for the proposed model).
  --dump-only: after writing the JSON, exits without proceeding to the real fit (for demonstration; the path up to the instantiation is the same as in production).
Output: the JSON of --out. If the file exists it is not overwritten and an exception is raised (it fails before the training starts).
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import socket
import sys
import time
from pathlib import Path


def _plain(o):
    """default of json.dumps: DictConfig/ListConfig of OmegaConf to plain dict/list, repr for anything else."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(o, (DictConfig, ListConfig)):
            return OmegaConf.to_container(o, resolve=True)
    except Exception:  # noqa: BLE001
        pass
    return repr(o)


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        return repr(x)


def build_dump(trainer, model, args, overrides) -> dict:
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint

    dm = model.diffusion_module
    pcf = getattr(dm, "pre_corruption_fn", None)
    loss_fn = getattr(dm, "loss_fn", None)

    ckpts = []
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ckpts.append(
                {
                    "class": type(cb).__name__,
                    "every_n_epochs": cb._every_n_epochs,
                    "every_n_train_steps": cb._every_n_train_steps,
                    "filename": cb.filename,
                    "save_last": cb.save_last,
                    "save_top_k": cb.save_top_k,
                    "monitor": cb.monitor,
                    "mode": cb.mode,
                }
            )

    opt_info, sched_info = None, []
    ret = model.configure_optimizers()
    if isinstance(ret, tuple):
        opts, scheds = ret
    else:
        opts, scheds = [ret], []
    opt = opts[0]
    opt_info = {
        "class": f"{type(opt).__module__}.{type(opt).__name__}",
        "defaults": {k: _jsonable(v) for k, v in opt.defaults.items()},
        "n_param_groups": len(opt.param_groups),
        "param_group0_lr": opt.param_groups[0]["lr"],
    }
    for sd in scheds:
        s = sd["scheduler"]
        sched_info.append(
            {
                "class": f"{type(s).__module__}.{type(s).__name__}",
                "T_max": getattr(s, "T_max", None),
                "eta_min": getattr(s, "eta_min", None),
                "interval": sd.get("interval"),
                "frequency": sd.get("frequency"),
                "strict": sd.get("strict"),
            }
        )

    cfg_dict = None
    for cb in trainer.callbacks:
        c = getattr(cb, "config", None)
        if isinstance(c, dict) and "lightning_module" in c:
            cfg_dict = c
            break
    nkey = None if cfg_dict is None else cfg_dict.get("num_atoms_distribution_key")

    host_z = None
    if args.kind == "proposed":
        from mattergen.symmetry_constrained.constants import SCM_HOST_Z, SCM_SOLUTE_ZS

        host_z = {"SCM_HOST_Z": int(SCM_HOST_Z), "SCM_SOLUTE_ZS": sorted(SCM_SOLUTE_ZS)}

    return {
        "kind": args.kind,
        "entry": args.entry,
        "overrides": overrides,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "dump_only": bool(args.dump_only),
        "p_unconditional": getattr(pcf, "p_unconditional", None),
        "dropout_fields_iid": getattr(pcf, "dropout_fields_iid", None),
        "pre_corruption_fn_class": None if pcf is None else f"{type(pcf).__module__}.{type(pcf).__name__}",
        "loss_fn": {
            "class": None if loss_fn is None else f"{type(loss_fn).__module__}.{type(loss_fn).__name__}",
            "reduce": getattr(loss_fn, "reduce", None),
            "d3pm_hybrid_lambda": getattr(loss_fn, "d3pm_hybrid_lambda", None),
            "loss_weights": getattr(loss_fn, "loss_weights", None),
            "gb_t_loss_weight": getattr(loss_fn, "gb_t_loss_weight", None),
            "gb_dx_loss_weight": getattr(loss_fn, "gb_dx_loss_weight", None),
            "has_freeze_atomic_numbers_attr": hasattr(loss_fn, "freeze_atomic_numbers"),
        },
        "diffusion_module": {
            "class": f"{type(dm).__module__}.{type(dm).__name__}",
            "has_freeze_atomic_numbers_attr": hasattr(dm, "freeze_atomic_numbers"),
            "has__freeze_atomic_numbers_attr": hasattr(dm, "_freeze_atomic_numbers"),
        },
        "trainer": {
            "gradient_clip_val": trainer.gradient_clip_val,
            "gradient_clip_algorithm": str(trainer.gradient_clip_algorithm),
            "max_epochs": trainer.max_epochs,
            "check_val_every_n_epoch": trainer.check_val_every_n_epoch,
            "limit_train_batches": trainer.limit_train_batches,
            "accumulate_grad_batches": trainer.accumulate_grad_batches,
            "precision": str(getattr(trainer, "precision", None)),
        },
        "model_checkpoint_callbacks": ckpts,
        "optimizer": opt_info,
        "schedulers": sched_info,
        "num_atoms_distribution_key": nkey,
        "host_z": host_z,
        "seed": {
            "PL_GLOBAL_SEED": os.environ.get("PL_GLOBAL_SEED"),
            "torch_initial_seed": torch.initial_seed(),
            "cfg_seed": None if cfg_dict is None else cfg_dict.get("seed"),
            "seed_everything_called_by_wrapper": args.seed,
        },
        "pytorch_lightning_version": pl.__version__,
    }


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: run_with_hparam_dump.py <options> -- <hydra overrides...>")
    split = argv.index("--")
    own, overrides = argv[:split], argv[split + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--kind", required=True, choices=["proposed", "baseline"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dump-only", action="store_true")
    args = ap.parse_args(own)

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"FATAL: {out} exists (overwriting is prohibited; stopping before the training starts)")
    out.parent.mkdir(parents=True, exist_ok=True)

    import pytorch_lightning as pl

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    orig_fit = pl.Trainer.fit

    def fit(self, model, *a, **k):
        d = build_dump(self, model, args, overrides)
        out.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=_plain))
        print(f"[hparam_dump] wrote {out}", flush=True)
        print(json.dumps({k2: d[k2] for k2 in ("p_unconditional", "loss_fn", "trainer", "optimizer", "schedulers",
                                               "num_atoms_distribution_key", "host_z", "seed")},
                         indent=2, ensure_ascii=False, default=_plain), flush=True)
        if args.dump_only:
            print("[hparam_dump] --dump-only: exiting without calling fit", flush=True)
            raise SystemExit(0)
        return orig_fit(self, model, *a, **k)

    pl.Trainer.fit = fit
    # hydra.main reads sys.argv[1:] as overrides.
    sys.argv = [args.entry] + overrides
    runpy.run_module(args.entry, run_name="__main__")


if __name__ == "__main__":
    main()
