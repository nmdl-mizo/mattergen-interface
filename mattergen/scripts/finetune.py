# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.cli import SaveConfigCallback

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT, get_device
from mattergen.diffusion.run import AddConfigCallback, SimpleParser, maybe_instantiate

logger = logging.getLogger(__name__)


def init_adapter_lightningmodule_from_pretrained(
    adapter_cfg: DictConfig, lightning_module_cfg: DictConfig
) -> Tuple[pl.LightningModule, DictConfig]:

    if adapter_cfg.model_path is not None:
        if adapter_cfg.pretrained_name is not None:
            logger.warning(
                "pretrained_name is provided, but will be ignored since model_path is also provided."
            )
        model_path = Path(hydra.utils.to_absolute_path(adapter_cfg.model_path))
        ckpt_info = MatterGenCheckpointInfo(model_path, adapter_cfg.load_epoch)
    elif adapter_cfg.pretrained_name is not None:
        assert (
            adapter_cfg.model_path is None
        ), "model_path must be None when pretrained_name is provided."
        ckpt_info = MatterGenCheckpointInfo.from_hf_hub(adapter_cfg.pretrained_name)

    ckpt_path = ckpt_info.checkpoint_path

    version_root_path = Path(ckpt_path).relative_to(ckpt_info.model_path).parents[1]
    config_path = ckpt_info.model_path / version_root_path

    # load pretrained model config.
    if (config_path / "config.yaml").exists():
        pretrained_cfg_path = config_path
    else:
        pretrained_cfg_path = config_path.parent.parent

    # global hydra already initialized with @hydra.main
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    with hydra.initialize_config_dir(str(pretrained_cfg_path.absolute()), version_base="1.1"):
        pretrained_cfg = hydra.compose(config_name="config")

    # compose adapter lightning_module config.

    ## copy denoiser config from pretrained model to adapter config.
    pretrained_diffusion_cfg = deepcopy(pretrained_cfg.lightning_module.diffusion_module)
    diffusion_module_cfg = (
        deepcopy(lightning_module_cfg.diffusion_module)
        if adapter_cfg.get("use_configured_diffusion_module", False)
        else pretrained_diffusion_cfg
    )
    denoiser_cfg = pretrained_diffusion_cfg.model

    with open_dict(adapter_cfg.adapter):
        for k, v in denoiser_cfg.items():
            # only legacy denoiser configs should contain property_embeddings_adapt
            if (
                k != "_target_"
                and k != "property_embeddings_adapt"
                and k != "site_property_embeddings_adapt"
            ):
                adapter_cfg.adapter[k] = v

            # do not adapt an existing <property_embeddings> field.
            if k == "property_embeddings":
                for field in v:
                    if field in adapter_cfg.adapter.property_embeddings_adapt:
                        adapter_cfg.adapter.property_embeddings_adapt.remove(field)

            if k == "site_property_embeddings":
                for field in v:
                    if field in adapter_cfg.adapter.site_property_embeddings_adapt:
                        adapter_cfg.adapter.site_property_embeddings_adapt.remove(field)

        # replace original GemNetT model with GemNetTCtrl model.
        adapter_cfg.adapter.gemnet["_target_"] = "mattergen.common.gemnet.gemnet_ctrl.GemNetTCtrl"

        # GemNetTCtrl model has additional input parameter condition_on_adapt, which needs to be set via property_embeddings_adapt.
        # Properties promoted to z_per_crystal (v2) skip the layerwise adapter pathway.
        _promote = set(adapter_cfg.adapter.get("promote_to_z", []))
        adapter_cfg.adapter.gemnet.crystal_condition_on_adapt = [
            k for k in adapter_cfg.adapter.property_embeddings_adapt
            if k not in _promote
        ]
        adapter_cfg.adapter.gemnet.site_condition_on_adapt = list(
            adapter_cfg.adapter.site_property_embeddings_adapt
        )

        # Surface fine-tune: override graph-construction PBC (z non-periodic for vacuum slabs).
        # Set AFTER the pretrained-denoiser copy above (which overwrites adapter.gemnet wholesale).
        # No-op for fine-tunes that don't set gemnet_pbc -> fully backward compatible. Baked into
        # the saved config.yaml, so generation inherits it automatically.
        if adapter_cfg.get("gemnet_pbc", None) is not None:
            adapter_cfg.adapter.gemnet.pbc = list(adapter_cfg.gemnet_pbc)

    # copy adapter config back into diffusion module config
    with open_dict(diffusion_module_cfg):
        diffusion_module_cfg.model = adapter_cfg.adapter
    with open_dict(lightning_module_cfg):
        lightning_module_cfg.diffusion_module = diffusion_module_cfg

    lightning_module = hydra.utils.instantiate(lightning_module_cfg)

    ckpt: dict = torch.load(ckpt_path, map_location=get_device())
    pretrained_dict: OrderedDict = ckpt["state_dict"]
    scratch_dict: OrderedDict = lightning_module.state_dict()

    # v2 hook: skip parameters whose shape changed (e.g. wider atom_latent_emb)
    model = lightning_module.diffusion_module.model
    resized_keys = (
        model.get_resized_param_keys()
        if hasattr(model, "get_resized_param_keys")
        else set()
    )
    scratch_dict.update(
        (k, pretrained_dict[k])
        for k in scratch_dict.keys() & pretrained_dict.keys()
        if k not in resized_keys
    )
    lightning_module.load_state_dict(scratch_dict, strict=True)

    # v2 hook: zero-pad pretrained weights into resized parameters
    if hasattr(model, "init_resized_params"):
        model.init_resized_params(pretrained_dict)

    # freeze pretrained weights if not full finetuning.
    if not adapter_cfg.full_finetuning:
        for name, param in lightning_module.named_parameters():
            if name in set(pretrained_dict.keys()):
                param.requires_grad_(False)

    if adapter_cfg.get("unfreeze_last_blocks", False):
        gemnet = lightning_module.diffusion_module.model.gemnet
        for module in (gemnet.int_blocks[-1], gemnet.out_blocks[-1]):
            for param in module.parameters():
                param.requires_grad_(True)

    # v4 M1 (, gb-inverse `internal notes` SS9.2 point 6,
    # opt-in via adapter config `unfreeze_fc_atom: true`, default False -> byte-identical to the
    # pre-flag freeze behaviour above): fc_atom (the atom-type prediction head, denoiser.py:211,
    # `self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM + int(with_mask_type))`) IS present in
    # every pretrained MatterGen checkpoint (mattergen_base carries fc_atom.weight/bias), so the
    # freeze-pretrained-weights rule two blocks above (`full_finetuning=false` -> freeze every
    # parameter name found in pretrained_dict) silently freezes it -- discovered while implementing
    # the acceptance condition "confirm that the atom-type head is not frozen in the adapter/config" (the per-atom D3PM
    # element generation cannot learn the GB-specific element distribution if this head never
    # receives gradient). Mirrors unfreeze_last_blocks's existing pattern exactly: same opt-in
    # config knob shape, same requires_grad_(True) call, just targeting fc_atom instead of GemNet's
    # last blocks. Inert (no-op) for any config that leaves unfreeze_fc_atom unset/False -- v1/v2/v3
    # GB configs and baseline MatterGen fine-tuning are all byte-identical unaffected.
    if adapter_cfg.get("unfreeze_fc_atom", False):
        fc_atom = lightning_module.diffusion_module.model.fc_atom
        for param in fc_atom.parameters():
            param.requires_grad_(True)

    # v2c: differential LR for SplitLinear cond projection
    model = lightning_module.diffusion_module.model
    if hasattr(model, "get_cond_param_names") and hasattr(model, "cond_lr_scale"):
        cond_suffixes = model.get_cond_param_names()
        scale = model.cond_lr_scale
        if cond_suffixes and scale != 1.0:
            base_lr = lightning_module_cfg.optimizer_partial.lr
            cond_params, base_params = [], []
            for name, p in lightning_module.diffusion_module.named_parameters():
                if not p.requires_grad:
                    continue
                if any(name.endswith(s) for s in cond_suffixes):
                    cond_params.append(p)
                else:
                    base_params.append(p)
            logger.info(
                "Differential LR: %d base params (lr=%.1e), %d cond params (lr=%.1e)",
                len(base_params), base_lr, len(cond_params), base_lr * scale,
            )

            def _make_optimizer(params=None, **kw):
                return torch.optim.AdamW(
                    [
                        {"params": base_params, "lr": base_lr},
                        {"params": cond_params, "lr": base_lr * scale},
                    ],
                )

            lightning_module._optimizer_partial = _make_optimizer

    return lightning_module, lightning_module_cfg


@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT / "conf"), config_name="finetune-sf", version_base="1.1"
)
def mattergen_finetune(cfg: omegaconf.DictConfig):
    # Tensor Core acceleration (leads to ~2x speed-up during training)
    torch.set_float32_matmul_precision("high")
    trainer: pl.Trainer = maybe_instantiate(cfg.trainer, pl.Trainer)
    datamodule: pl.LightningDataModule = maybe_instantiate(cfg.data_module, pl.LightningDataModule)

    # establish an adapter model
    pl_module, lightning_module_cfg = init_adapter_lightningmodule_from_pretrained(
        cfg.adapter, cfg.lightning_module
    )

    # replace denoiser config with adapter config.
    with open_dict(cfg):
        cfg.lightning_module = lightning_module_cfg

    config_as_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(config_as_dict, indent=4))
    # This callback will save a config.yaml file.
    trainer.callbacks.append(
        SaveConfigCallback(
            parser=SimpleParser(),
            config=config_as_dict,
            overwrite=True,
        )
    )
    # This callback will add a copy of the config to each checkpoint.
    trainer.callbacks.append(AddConfigCallback(config_as_dict))

    # Optional training resume from a Lightning checkpoint (epoch/optimizer/step continuation).
    # Absent by default -> ckpt_path=None -> identical to the original non-resume behavior
    # (fully backward compatible). Add via CLI with `+resume_ckpt_path=<abs/path/to/last.ckpt>`.
    #
    # NOTE ON LAYERING: adapter.model_path (above) is a *warm-start* that instantiates the adapter
    # module and copies pretrained weights before .fit(). resume_ckpt_path is Lightning's *native*
    # training-resume hook: Trainer.fit(ckpt_path=...) restores model weights, optimizer state,
    # LR-scheduler state, current_epoch and global_step from the checkpoint. When both are given,
    # the ckpt_path state is loaded last and is what training actually continues from. Since
    # trainer.max_epochs (= data_module.max_epochs) is unchanged, a run interrupted at epoch N
    # resumes at epoch N and continues to max_epochs.
    resume_ckpt_path = cfg.get("resume_ckpt_path", None)
    trainer.fit(
        model=pl_module,
        datamodule=datamodule,
        ckpt_path=resume_ckpt_path,
    )


if __name__ == "__main__":
    mattergen_finetune()
