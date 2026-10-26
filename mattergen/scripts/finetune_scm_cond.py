"""Fine-tuning entry point of the symmetry-constrained model (proposed).

Starts from the pretrained MatterGen base model (`pretrained_name=mattergen_base`) and adds the
gamma (grain-boundary energy) scalar adapter condition (GemNetTAdapter + property_embeddings_adapt).

Why this script exists instead of reusing finetune.py's pretrained-load path:
  finetune.py's `init_adapter_lightningmodule_from_pretrained` always takes the denoiser config
  (`model:` group) from the PRETRAINED run's saved config.yaml and unconditionally overwrites
  `adapter_cfg.adapter` with it wholesale, which would stomp the symmetry-constrained
  adapter/GemNetTCtrl/property_embeddings_adapt wiring configured in the proposed configuration.

The pretrained state_dict is loaded per key by `apply_partial_load_cond` below: every key present
in the pretrained checkpoint with a matching shape is copied; keys that exist only in the new model
(every property_embeddings_adapt.* parameter) stay at Hydra's fresh initialization; any other
shape mismatch fails loudly with RuntimeError (no silent dropping of weights).

Otherwise it mirrors finetune.py's overall shape (hydra.main -> build trainer/datamodule/
lightning_module -> load pretrained state_dict -> partial-load -> trainer.fit), including the
opt-in `+seed=<int>` injection.
"""

import json
import logging
from pathlib import Path

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.cli import SaveConfigCallback

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT, get_device
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.diffusion.run import AddConfigCallback, SimpleParser, maybe_instantiate

logger = logging.getLogger(__name__)


def load_pretrained_state_dict(
    pretrained_name: str | None, model_path: str | None, load_epoch: int | str = "last"
) -> dict:
    """Resolve the pretrained checkpoint (HF hub name or a local run path) and return its
    state_dict.

    `load_epoch` selects WHICH checkpoint file under `model_path` to load, for run directories
    that contain multiple `epoch=N.ckpt` files in addition to `last.ckpt` (a SPECIFIC epoch, e.g.
    `epoch=179.ckpt`, not whatever `last.ckpt` happens to be in that run directory). Must be
    the literal string "last" (default, backward compatible with every existing caller that does
    not set `load_epoch` in its Hydra config) or an int; any other type is rejected immediately
    rather than silently falling back to "last" (silent-fallback prohibition).
    """
    if model_path is not None:
        if load_epoch != "last" and not isinstance(load_epoch, int):
            raise ValueError(f"load_epoch must be 'last' or an int, got {load_epoch!r}")
        ckpt_info = MatterGenCheckpointInfo(Path(model_path), load_epoch)
        # Resolve eagerly (checkpoint_path is a cached_property) so a mis-specified/missing
        # epoch fails fast here, and so the resolved path can be logged to prevent mix-ups
        # detection before any training state depends on it.
        resolved_path = Path(ckpt_info.checkpoint_path).resolve()
        logger.info("resolved checkpoint: %s", resolved_path)
    else:
        assert pretrained_name is not None, "Either model_path or pretrained_name is required."
        ckpt_info = MatterGenCheckpointInfo.from_hf_hub(pretrained_name)
    ckpt: dict = torch.load(ckpt_info.checkpoint_path, map_location=get_device())
    return ckpt["state_dict"]


def apply_partial_load_cond(lightning_module: DiffusionLightningModule, pretrained_dict: dict) -> None:
    """Load `pretrained_dict` into `lightning_module` (the pretrained denoiser shape PLUS newly
    added property_embeddings_adapt.* parameters).

    Algorithm, per key in `lightning_module.state_dict()`:
      1. Absent from `pretrained_dict` entirely (every property_embeddings_adapt.* parameter)
         -> left at Hydra's fresh-initialization value.
      2. Present with a MATCHING shape -> copied verbatim (plain state_dict-style partial load).
      3. Present with a MISMATCHED shape -> fails loudly with RuntimeError rather than silently
         dropping weights .
    """
    scratch_dict = lightning_module.state_dict()
    copied, skipped_new, mismatch = [], [], []
    update = {}
    for k, v in scratch_dict.items():
        if k not in pretrained_dict:
            skipped_new.append(k)
            continue
        pv = pretrained_dict[k]
        if pv.shape == v.shape:
            update[k] = pv
            copied.append(k)
        else:
            mismatch.append(k)

    scratch_dict.update(update)
    lightning_module.load_state_dict(scratch_dict, strict=True)

    logger.info(
        "partial load: copied %d keys from checkpoint, left %d new (adapter) keys at "
        "fresh init, skipped %d shape-mismatched keys.",
        len(copied),
        len(skipped_new),
        len(mismatch),
    )
    if mismatch:
        raise RuntimeError(
            f"partial load found {len(mismatch)} shape-mismatched keys "
            f"(not simply new adapter params): {mismatch[:10]}... "
            "This usually means the checkpoint does not match the "
            "lightning_module/diffusion_module/model config selected on the CLI."
        )


@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT / "conf"),
    config_name="finetune-scm-al-proposed",
    version_base="1.1",
)
def mattergen_finetune_scm_cond(cfg: omegaconf.DictConfig) -> None:
    torch.set_float32_matmul_precision("high")

    # Seed injection: opt-in via `+seed=<int>` on the CLI (the `seed` key does not exist in the
    # configuration by default, hence `+`). Absent -> no-op.
    seed = cfg.get("seed", None)
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    trainer: pl.Trainer = maybe_instantiate(cfg.trainer, pl.Trainer)
    datamodule: pl.LightningDataModule = maybe_instantiate(cfg.data_module, pl.LightningDataModule)

    pretrained_dict = load_pretrained_state_dict(
        pretrained_name=cfg.get("pretrained_name", None),
        model_path=cfg.get("model_path", None),
        load_epoch=cfg.get("load_epoch", "last"),
    )

    lightning_module: DiffusionLightningModule = hydra.utils.instantiate(cfg.lightning_module)
    apply_partial_load_cond(lightning_module, pretrained_dict)

    if not cfg.get("full_finetuning", True):
        for name, param in lightning_module.named_parameters():
            if name in pretrained_dict:
                param.requires_grad_(False)

    config_as_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(config_as_dict, indent=4))
    trainer.callbacks.append(
        SaveConfigCallback(parser=SimpleParser(), config=config_as_dict, overwrite=True)
    )
    trainer.callbacks.append(AddConfigCallback(config_as_dict))

    resume_ckpt_path = cfg.get("resume_ckpt_path", None)
    trainer.fit(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path=resume_ckpt_path,
    )


if __name__ == "__main__":
    mattergen_finetune_scm_cond()
