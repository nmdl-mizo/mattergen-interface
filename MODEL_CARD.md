# Model card: symmetry-constrained MatterGen for Al Sigma5(210)[001] tilt grain boundaries

This card describes the two checkpoints released with this repository (in the Zenodo record `10.5281/zenodo.23667782`):
`proposed_epoch159.ckpt` (the *symmetry-constrained model*, proposed) and `baseline_epoch129.ckpt` (the *unconstrained baseline*).
It replaces the model card of upstream MatterGen, which is kept in the upstream repository (https://github.com/microsoft/mattergen) and does not describe these checkpoints.

## Model details

- Both checkpoints are fine-tuned from the pretrained checkpoint `mattergen_base` of [microsoft/mattergen](https://github.com/microsoft/mattergen) (MIT License), with the full model fine-tuned (`full_finetuning: true`).
- **Proposed (symmetry-constrained) model**: one grain boundary is generated and the second one is reconstructed from it by a 2_1 screw operation at every denoising step; the bulk regions are fixed. It is conditioned on the grain-boundary energy gamma (J/m^2) with classifier-free guidance (probability of dropping the condition during training: 0.2). See `README.md` section 2.
- **Unconstrained baseline**: MatterGen fine-tuned on the same rows with the standard data representation and without the constraint; all atoms and the cell diffuse.
- Training: AdamW, learning rate 5.0e-06 with cosine annealing, batch size 4, 200 epochs, precision 32, gradient clipping 0.5 (value), seed 20290000, on 1 GPU. The resolved configurations are released next to the checkpoints (`proposed_resolved_config.yaml`, `baseline_resolved_config.yaml`; see `docs/reproduce.md`). Training hardware: one NVIDIA RTX PRO 5000 Blackwell GPU (48 GB) per model; 200 epochs took 9.1 h (proposed) and 8.4 h (baseline).
- Code state at training time: see `NOTICE`.

## Training data

- 5,509 relaxed Al Sigma5(210)[001] symmetric tilt grain-boundary structures (31 CNID templates, 10 rigid-shift values dx, structures with 0 to 4 symmetrically removed atom pairs; 112 to 120 atoms), built from 6,510 candidate structures after removal of duplicates.
- Split by template: train 4,419, validation 538, test 552 structures (25 / 3 / 3 templates), the same rows for both models.
- Label: the grain-boundary energy gamma after structural relaxation with the machine-learning interatomic potential MACE-MH-1 (head `omat_pbe`; FIRE, fmax 0.02 eV/A, at most 500 steps; the two bulk-side blocks of 76 atoms fixed). The labels are MLIP values, not DFT values.
- Files: `training/` in the Zenodo record.

## Checkpoint selection

The checkpoint with the lowest validation loss was released for each model.

| model | file | epoch | validation loss | SHA-256 |
|---|---|---|---|---|
| proposed (symmetry-constrained) | `proposed_epoch159.ckpt` | 159 | 0.6371 | `0e236655bd2454e701c2bde95db9a3f80f3aed85e217f9bdf3f9b663201c0e87` |
| unconstrained baseline | `baseline_epoch129.ckpt` | 129 | 2.6200 | `2b8b4bf24b2ba8f5d35c0c15e1068683dd983882728412fc4c43770219eee252` |

The two validation losses are defined differently (the proposed model's loss covers only the variables that diffuse), so they must not be compared with each other.

## Intended use

Generation of Al Sigma5(210)[001] symmetric tilt grain boundaries at a target grain-boundary energy within the range covered by the training data (about 0.53 to 2.54 J/m^2), as described in the paper, and reproduction of the paper's results (`docs/reproduce.md`).
The models are not intended for other grain boundaries, other elements, or other applications.

## Limitations

- A single grain-boundary geometry (Sigma5(210)[001] symmetric tilt) and a single element (Al).
- The labels and the evaluation of generated structures use a machine-learning potential (MACE-MH-1); DFT checks cover only a subset of the generated structures.
- Targets outside the range of the training energies are extrapolations and are tracked less well.
- The variation between independent trainings was not measured (one training per model).
- Structures should be relaxed and checked before any further use.

## License

MIT License (see `LICENSE`; the same license as the upstream MatterGen weights). The accompanying data are licensed under CC BY 4.0.

## Citation

Please cite this repository as given in `CITATION.cff`. The accompanying paper, "Generating grain boundary structures at a specified energy with a symmetry-constrained diffusion model", has been submitted to APL Machine Learning.
Please also cite MatterGen: Zeni et al., Nature 639, 624-632 (2025), doi:10.1038/s41586-025-08628-5. See `CITATION.cff`.
