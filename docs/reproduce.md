# Reproduction guide

Four levels, from cheap to expensive. All levels except R3 were run on the authors' cluster with the files of this release: R0 reproduces the 10 figure PNG files of the paper byte for byte (SHA-256 identical) from the released files alone; R1 re-ran the relaxation of 3 structures and the re-run relaxed gamma agrees with the published value to within 5.0e-14 J/m^2; R2 generated 5 structures of the proposed model and 5 of the baseline with the commands of this guide from the released checkpoints (SHA-256 verified; the symmetry residual of the generated structures of the proposed model was 0.0); the batch script of the baseline experiment generated one chunk (25 structures); R3 was not run. For a fixed input (deterministic algorithms enabled) the outputs of both networks and one reverse-diffusion step are bit-identical to those of the previous release candidate, which used the checkpoints before they were re-saved (see `NOTICE`, section 3). Note that diffusion sampling is not bit-reproducible on a GPU even with a fixed seed: repeated runs of the same code with the same seed give different structures with the same atom counts.
All commands use environment variables instead of paths; there are no defaults (an unset variable raises an error).

## Environment variables

| variable | meaning |
|---|---|
| `GBREL_CODE` | root of this repository |
| `GBREL_DATA` | the extracted data record (Zenodo `<ZENODO_DOI>`): `training/ generation/ relaxation/ evaluation/ dft/ weights/` |
| `GBREL_WORK` | writable work directory (empty or new). `gb_eval/make_compat_tree.py` builds `$GBREL_WORK/workspace/...`, a symlink view that mirrors the directory layout the analysis scripts were written for; new outputs are written there |
| `MACE_MODEL_PATH` | MACE-MH-1 model file (not distributed; see the top-level README) |
| `MACE_PY` | python of the evaluation environment (`environment-eval.yml`) |
| `GEN_PY` | python of the generation environment (`environment-gen.txt`); required by every `gb_eval` script because the shared constants module reads it, even where it is not run |

```bash
export GBREL_CODE=/path/to/mattergen-interface
export GBREL_DATA=/path/to/extracted/data-record
export GBREL_WORK=/path/to/empty/work
export MACE_MODEL_PATH=/path/to/mace-mh-1.model
export MACE_PY=/path/to/eval-env/bin/python
export GEN_PY=/path/to/gen-venv/bin/python
(cd /path/to/downloaded/files && sha256sum -c SHA256SUMS)   # the files as downloaded from the data record (before extracting them), checked before anything else
```

The two checkpoints must have these SHA-256 values (also in the data record):

- `weights/proposed_epoch159.ckpt` (proposed, symmetry-constrained model, epoch 159): `0e236655bd2454e701c2bde95db9a3f80f3aed85e217f9bdf3f9b663201c0e87`
- `weights/baseline_epoch129.ckpt` (unconstrained baseline, epoch 129): `2b8b4bf24b2ba8f5d35c0c15e1068683dd983882728412fc4c43770219eee252`

## R0: figures and numbers (CPU, minutes)

Regenerates the 10 figure PNG files of the paper from the files in `data/paper_numbers/` and the data record.

```bash
export PYTHONPATH=$GBREL_CODE/gb_eval:$GBREL_CODE/figures/scripts
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
$MACE_PY $GBREL_CODE/gb_eval/make_compat_tree.py   # builds the symlink view; published figure PNGs are never linked
for s in fig_F1_schematic fig_F2_validity fig_F3_tracking fig_F4_novelty fig_F5_si_w_sweep \
         fig_F6_si_dft_vs_mlip fig_F7_si_representative_structures fig_F8_si_dx_distribution fig_F9_si_dmax_ecdf fig_F10_si_main_w0_tracking; do
  $MACE_PY $GBREL_WORK/workspace/paper/figures/scripts/$s.py   # via the symlink view (the F1-F4 scripts find their inputs relative to their own path)
done
# PNG files: $GBREL_WORK/workspace/paper/figures/png/
```

As a batch job: `sbatch gb_eval/slurm/sbatch_r0_figures.sh` (edit the partition lines for your cluster).
The figure scripts of F1-F4 convert a vector drawing with `rsvg-convert` (librsvg); the scripts refuse to overwrite an existing PNG.
The SI figure scripts F6-F9 assert numbers against the paper's numbers table; they read the extract `data/paper_numbers/ledger_subset.yaml`
(only the entries and fields these four scripts read; the full ledger is not distributed).

## R1: evaluation chain smoke test (1 GPU, minutes)

Relaxes 3 generated structures of the proposed model (w = 2.0, target 1) with MACE-MH-1 and compares the relaxed gamma with the published rows.

```bash
export PYTHONPATH=$GBREL_CODE/gb_eval
$MACE_PY $GBREL_CODE/gb_eval/make_compat_tree.py
$MACE_PY $GBREL_CODE/gb_eval/smoke/r1_relax_compare.py --chunk 2 --indices 0,1,2 --out <output directory>
```

Batch version: `R1_OUT=<output directory> sbatch gb_eval/slurm/sbatch_r1_smoke.sh`. The output `compare.tsv` lists the published and re-run values and their difference.
MLIP calculations are never run on CPU by these scripts (a GPU is required).

## R2: generation from the released checkpoint (1 GPU, tens of minutes)

Generates 5 structures of target 1 with the released checkpoint of the proposed model. `mattergen` must be imported from this repository (the batch script checks this).

```bash
export PYTHONPATH=$GBREL_CODE
export SCM_HOST_Z=13 SCM_SOLUTE_ZS=
sha256sum $GBREL_DATA/weights/proposed_epoch159.ckpt          # must equal the value above
export R2_OUT=/path/to/new/output/directory
mkdir -p $R2_OUT
envsubst '${GBREL_DATA}' < $GBREL_DATA/training/gamma_dedup.csv > $R2_OUT/gamma_dedup.expanded.csv
$GEN_PY $GBREL_CODE/scripts/generate_symmetry_constrained.py \
  --checkpoint $GBREL_DATA/weights/proposed_epoch159.ckpt --config $GBREL_DATA/weights/proposed_resolved_config.yaml \
  --guidance_scale 2.0 --seed 20301002 --gamma_target 1.1321329025690032 --num_samples 5 --output_path $R2_OUT/gen \
  --gamma_csv_path $R2_OUT/gamma_dedup.expanded.csv \
  --bulk_membership_csv_path $GBREL_DATA/training/bulk_membership.csv --manifest_dir $GBREL_DATA/training/manifests \
  --num-atoms-distribution-key AL_S5_210_TRAIN --host-z 13
```

Batch version: `R2_OUT=<new directory> sbatch gb_eval/slurm/sbatch_r2_smoke.sh`. The full experiment of the paper is 7 targets x 50 structures
(targets and seeds: chunk c = 0..13 uses target c // 2 of the 7 gamma levels and seed 20301000 + c; the 7 gamma levels are in the data record,
`generation/proposed_main/`). The `dir` column of the training CSV files contains `${GBREL_DATA}` and must be expanded with `envsubst` before a program reads it.

## R2 (baseline): generation of the unconstrained baseline (1 GPU, tens of minutes)

Generates 5 structures of target 1 with the released checkpoint of the unconstrained baseline through the standard MatterGen path (`mattergen.scripts.generate`; the wrapper `gen_baseline_seeded.py` only sets the random seed and then calls it).
`mattergen.scripts.generate` reads a model directory: a `config.yaml` (the complete model configuration, `weights/baseline_model_config.yaml`) next to `checkpoints/epoch=<N>.ckpt`. The data record holds the configuration and the checkpoint separately, so build the directory first.
The baseline needs about 48 GB of GPU memory for 5 structures (measured peak 47,904 MiB for a batch of 5 at guidance factor 2.0): use a GPU with 80 GB or more.

```bash
export PYTHONPATH=$GBREL_CODE
export SCM_HOST_Z=13 SCM_SOLUTE_ZS=
sha256sum $GBREL_DATA/weights/baseline_epoch129.ckpt       # must equal the value above
export R2B_OUT=/path/to/new/output/directory
mkdir -p $R2B_OUT/model_path/checkpoints
cp $GBREL_DATA/weights/baseline_model_config.yaml $R2B_OUT/model_path/config.yaml
ln -s $GBREL_DATA/weights/baseline_epoch129.ckpt $R2B_OUT/model_path/checkpoints/epoch=129.ckpt
$GEN_PY $GBREL_CODE/gb_eval/baseline_generation/gen_baseline_seeded.py --seed 20301002 -- $R2B_OUT/gen \
  --model_path=$R2B_OUT/model_path --checkpoint_epoch=129 --num_atoms_distribution=AL_S5_210_TRAIN \
  --batch_size=5 --num_batches=1 "--properties_to_condition_on={'gb_gamma_J_m2':1.1321329025690032}" \
  --diffusion_guidance_factor=2.0 --record_trajectories=False
```

One chunk (25 structures) of the paper's baseline experiment is the same command with `--num_batches=5`, seed 20301000 + c and target c // 2 of the 7 gamma levels (c = 0..13); the batch script is `gb_eval/experiments/range_targets/scripts/sbatch_gen_baseline.sh` (it builds the model directory in the same way from `weights/baseline_model_config.yaml`).
The seed fixes the random number generators but diffusion sampling on a GPU is still not bit-reproducible (see above).

## The batch script of the baseline experiment (one chunk)

`gb_eval/experiments/range_targets/scripts/sbatch_gen_baseline.sh` is the batch script that generated the paper's baseline structures (array task k = 14..27 is chunk c = k - 14 of the main experiment, w = 2.0: 25 structures, 5 batches of 5; seed 20301000 + c). Before it generates anything it checks the training configuration (`check_baseline_config.py`), the checkpoint (sha256 and epoch), the free GPU memory (71 GiB) and builds the model directory; afterwards it checks the loaded checkpoint, the number of structures and writes `CHUNK_DONE`.
It reads its inputs from `$GBREL_WORK/workspace/experiments/range_targets/`, so prepare that directory in a new work directory. Do not use `gb_eval/make_compat_tree.py` for this: it links `results/gen/baseline` to the published chunks in the data record, which the script refuses to overwrite.

```bash
export GBREL_WORK=/path/to/new/work
P=$GBREL_WORK/workspace/experiments/range_targets
mkdir -p $P/results $P/scripts $GBREL_WORK/logs
cp $GBREL_CODE/gb_eval/experiments/range_targets/scripts/check_baseline_config.py $P/scripts/
cp $GBREL_CODE/data/paper_numbers/experiments/range_targets/results/eval_targets_range.json $P/results/
envsubst '${GBREL_DATA} ${GBREL_WORK}' < $GBREL_CODE/data/paper_numbers/experiments/range_targets/results/ckpt_resolved_baseline.json > $P/results/ckpt_resolved_baseline.json
cd $GBREL_WORK && sbatch --array=14 $GBREL_CODE/gb_eval/experiments/range_targets/scripts/sbatch_gen_baseline.sh   # the #SBATCH partition/gres lines are examples: override them for your cluster; GBREL_CODE, GBREL_DATA, GBREL_WORK and GEN_PY must be exported
```

The script requires 71 GiB of free GPU memory by default (1.5 times the measured peak of a batch of 5). On a GPU with 48 GB, `export VRAM_MIN_GIB_W4=46` lowers the limit to 46 GiB; the script accepts this for w = 2.0 and w = 4.0 (the measured peak of a chunk of 25 structures was 44,586 MiB on a 96 GB GPU and 45,222 MiB on a 48 GB GPU, so it fits in 48 GB).

`gb_eval/experiments/verification/scripts/sbatch_gen_baseline.sh` (the three regenerations of the baseline, `LAYER=baseline_regen1|2|3`, array k = 0..13) has the same checks and reads the same files. Its output directory is the layer's `input_run_root` in `gb_eval/experiments/verification/config/layers.json`, which points into the published data (`$GBREL_DATA/generation/baseline_repN`); it was only dry-run for the release (`DRYRUN=1`).

## R3: training (not run before release)

The resolved training configurations are `weights/proposed_resolved_config.yaml` and `weights/baseline_resolved_config.yaml` (the configuration stored inside each checkpoint agrees with them, up to the entry `MODELS_PROJECT_ROOT`, which the entry script sets).
The training commands are recorded in `gb_eval/experiments/data_rebuild/scripts/sbatch_train_al_proposed.sh` and
`sbatch_train_al_baseline.sh` (same directory); the configurations are `mattergen/conf/finetune-scm-al-proposed.yaml` and
`mattergen/conf/finetune-scm-al-baseline-dedup.yaml` together with `data_module/scm_al_proposed.yaml`, `data_module/scm_al_baseline.yaml`, `lightning_module/diffusion_module/corruption/scm_al_proposed.yaml` and `adapter/scm_al_baseline.yaml` (all in `mattergen/conf/`).
They start from the `mattergen_base` checkpoint of microsoft/mattergen on Hugging Face (see the top-level README), use `lr = 5.0e-06`, 200 epochs and seed 20290000.
**These commands were not re-run for the release (several GPU-days).** The scripts that learn from the data and the learning-side configuration writer (`stage_e_write_configs.py`, `stage_e_precheck.py`) were not verified on the released layout; `stage_e_write_configs.py` and the copies of `stage_d1_build_csv.py` and `build_tilt_baseline_csv.py` in that directory still check the original compute-node path notation and reject the placeholder form (the copies in `structure_builder/` do not). The pipeline that builds `training/` from the CNID templates is in `structure_builder/` (see `structure_builder/README.md`, with smoke tests that rebuild parts of the data from the released inputs and print the differences: rebuilding the 310 dx-shifted templates from the CNID grid gives a maximum atomic-position difference of 0.0 Å, the 6,200 atom-removal structures and their fixed-atom indices match, gamma recomputed from the released relaxed structures differs by 0.0, the split matches, and the deduplication of one group keeps the same 28 structures; re-relaxing 3 released structures with the MLIP reproduces the released energies to within 4e-13 eV). The old copies under `gb_eval/experiments/data_rebuild/scripts/` are kept only because evaluation scripts import `stage_d_common.py` from there (`gb_eval/experiments/relax_novelty/scripts/common.py`, `gb_eval/experiments/verification/scripts/common.py`).
The adapter configuration `adapter/scm_al_baseline.yaml` of the baseline's training configuration is included (it was untracked in the fork at training time; see `NOTICE`).

## What is included, and the file selection rule

- `mattergen/`, `scripts/`, `sampling_conf/`, ...: files of upstream MatterGen that exist in the fork's tag, plus every module reachable by `import` from the entry points
  used for the paper (the generator `scripts/generate_symmetry_constrained.py`, the two training entry points, and every `_target_` of both resolved training configurations)
  and the configuration files those entry points compose.
- `gb_eval/`, `figures/`: the evaluation, aggregation and plotting scripts that produced the files in `data/paper_numbers/` and the data record, with the scripts they import.
  Absolute paths were replaced mechanically (see `gb_eval/paths.py`); no logic was changed apart from this. `gb_eval/vendor/README.md` lists files that came from outside the project.
- `structure_builder/`: the scripts of the training-structure pipeline (originals with path changes only; see `structure_builder/PATCHES.md`).
- Removed from the files above: the variable-atom-count extension and the fixed-length layout code that no entry point reaches (see `NOTICE`, section 3).
- Not included: other research lines of the same fork, the paper manuscript and its ledgers, VASP POTCAR files, large intermediate files.

## Known limitations

- Several scripts call laboratory-internal tools (job pre-flight checks, the job scheduler registry); the lines that call them are kept but the tools are not included. The calls of the model-specification check were removed from the sbatch generation scripts.
  They are not needed for R0-R2.
- The copies of the data-pipeline scripts under `gb_eval/experiments/data_rebuild/scripts/` that validate "compute-node path notation" assume the original directory layout and will reject the placeholder form; use the versions in `structure_builder/`, whose complete run was not repeated for the release.
- `figures/scripts/fig_F6_si_dft_vs_mlip.py`, `fig_F7_si_representative_structures.py`, `fig_F8_si_dx_distribution.py` and `fig_F9_si_dmax_ecdf.py` read an extract of the
  paper's numbers table (`data/paper_numbers/ledger_subset.yaml`), not the full ledger. Numbers of the paper that were taken from internal working documents (design notes,
  decision note, a drafting reference table) are not backed by a file in this repository.
- The checkpoints contain a few strings of the original absolute paths of the training run (inside the pickled configuration); they are not used.

## Names used before this release

The extension packages, their classes, the environment variables, the configuration and script names and the num_atoms key were renamed for this release. The checkpoints and the configuration files of the data record carry the new names only; no compatibility layer for earlier names is included.
