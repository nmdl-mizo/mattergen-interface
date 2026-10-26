# data/paper_numbers

Primary files from which the numbers and figures of the paper are computed (the files that the paper's numbers table points to, plus the inputs of the figure scripts).
Paths below are relative to `data/paper_numbers/`; they mirror the directory layout of the original analysis project so that the figure scripts find them
(see `gb_eval/make_compat_tree.py`). Text files had their absolute paths replaced by `${GBREL_*}` / `<ORIGINAL_ROOT>` placeholders; nothing else was changed.

| file or pattern | kind | used for (number keys of the paper's ledger / figures) |
|---|---|---|
| `workspace/experiments/verification/results/gb_stage2/stage2_check_inputs.json` | table | 68 number keys (P12, P13, P14, …) |
| `workspace/experiments/relax_novelty/results/gb_equivalence/gb_symfit_additional_g2_and_e.md` | report | 50 number keys (A03, A02, A05, …) |
| `workspace/experiments/verification/results/summary_tables/m3_support.csv` | table | 33 number keys (TGT0, TGT1, TGT2, …); figure F3 (input) |
| `workspace/experiments/verification/results/w0_baseline/base_rate.csv` | table | 31 number keys (BR0, BR1, BR2, …) |
| `workspace/experiments/verification/results/regeneration/g2_by_layer.csv` | table | 30 number keys (A01, B60, B61, …) |
| `workspace/experiments/verification/results/stol_sensitivity/summary.md` | report | 26 number keys (STLAB, SG0, SD0, …) |
| `workspace/experiments/verification/results/summary_tables/m2_proposed_tracking.csv` | table | 15 number keys (TSS, TSL, TSH, …); figure F3 (input) |
| `workspace/experiments/relax_novelty/scripts/gb_symfit.py` | code | 13 number keys (P01, P02, P03, …) |
| `workspace/experiments/relax_novelty/results/gb_equivalence/gb_symfit_main.csv` | table | 13 number keys (S1NEQ, DMXPL, DMXPH, …); figure F2 (note), F9 (input) |
| `workspace/experiments/relax_novelty/results/relax_novelty_report.md` | report | 11 number keys (N04, N01, N03a, …) |
| `workspace/experiments/verification/results/summary_tables/m7_dft_stats.csv` | DFT | 11 number keys (DFTRHO, DFTR, DFTOFF, …); figure F6 (input) |
| `workspace/paper/figures/F4_classification_counts.csv` | table | 11 number keys (NVS0, NVS1, NVS2, …); figure F4 (input) |
| `workspace/experiments/relax_novelty/results/train_ref_manifest.csv` | table | 10 number keys (N02_112, N02_114, N02_116, …); figure F7 (input) |
| `workspace/experiments/verification/results/low_gamma_novel_view/nn/summary.md` | report | 8 number keys (LGN, LGHMIN, LGDLO, …) |
| `workspace/experiments/verification/results/checks/check_2_table.md` | report | 7 number keys (DFTPH, DFTPV, DFTNC, …) |
| `workspace/experiments/relax_novelty/results/relax_novelty_per_target.csv` | table | 7 number keys (AGX0, AGX1, AGX2, …) |
| `workspace/experiments/verification/results/dft_recalc/verdict.json` | DFT | 6 number keys (DFTNP, DFTK, DFTP, …) |
| `workspace/experiments/verification/results/summary_tables/m1_classes.csv` | table | 6 number keys (F1NGB, F1NGBP, F1XPRE, …); figure F2 (input) |
| `tmp/cnid_grid_sweep/unified_10x10/summary.txt` | table (text) | 6 number keys (CNIDN, CNIDP, TPLPAIR, …) |
| `workspace/experiments/data_rebuild/results/split.json` | table | 6 number keys (DSTOT, DSVAL, DSTEST, …); figure F10 (input), F3 (input), F5 (input) |
| `workspace/experiments/verification/results/grid_interp/summary.md` | report | 5 number keys (GRMG, GRME, GRMX, …) |
| `workspace/experiments/verification/results/w0_baseline/w0_gamma_summary.json` | table | 5 number keys (W0MAX, W0MIN, W0N, …) |
| `workspace/paper/figure_sources/fig4/distance_audit/data/Fig4_distance_stats.csv` | table | 4 number keys (GRMN, GRP10, GRP90, …); figure F4 (input) |
| `workspace/paper/figures/F4_classification_sensitivity.csv` | table | 4 number keys (STU0, STU1, STU2, …) |
| `workspace/experiments/relax_novelty/results/structure_filtered/report.md` | report | 4 number keys (A04, K01, K02, …) |
| `workspace/experiments/verification/results/gb_stage2/gb_stage2_main.csv` | table | 4 number keys (EQ91, EQ92, EQ93, …); figure F2 (note), F7 (input), F9 (input) |
| `workspace/experiments/verification/results/summary_tables/m2_baseline_g2_tracking.csv` | table | 4 number keys (V2SPRE, V2SPOST, V2SN, …) |
| `workspace/counts/dx_range.out` | table (text) | 4 number keys (DXN, DXOUT, DXMIN, …) |
| `workspace/experiments/data_rebuild/scripts/builder_common.py` | code | 4 number keys (DSDXST, REMP5, REMPAT, …) |
| `workspace/experiments/relax_novelty/scripts/common.py` | code | 3 number keys (P17, ANGTOL, ANGREF) |
| `workspace/experiments/verification/dft_results/gb_proposed_t0_hi/OUTCAR` | DFT | 3 number keys (DFTVS643, DFTV643, DFTVP643) |
| `workspace/experiments/verification/results/novelty_control/verdicts_novelty_control.md` | report | 3 number keys (PCOK, PCNOV, PC14) |
| `workspace/experiments/verification/results/theta_all5/criterion_inputs_4.json` | table | 3 number keys (EQ99, EQ99lo, EQ99hi) |
| `workspace/experiments/relax_novelty/results/relax/main/proposed_w2.0/chunk00/relax_rows.csv` | table | 3 number keys (BULKN, F1FIXP, FREEN) |
| `workspace/experiments/range_targets/results/eval_targets_range.json` | table | 3 number keys (TGTW, TGTD, GENREP) |
| `workspace/experiments/data_rebuild/results/dxvpsde_stats.json` | table | 3 number keys (DSDX, DSDXLO, DSDXHI) |
| `workspace/experiments/data_rebuild/results/template_map.csv` | table | 3 number keys (CELLA, CELLB, CELLC) |
| `workspace/experiments/data_rebuild/results/num_atoms_dist_train.json` | table | 3 number keys (NATSET, NATLV, MRANGE) |
| `workspace/experiments/bulk_reference/bulk/energy.json` | table | 3 number keys (EBULK, EBULKN, EBULKA) |
| `workspace/experiments/range_targets/results/eval/baseline_w2.0/gamma_pre.csv` | table | 3 number keys (AGLO, AGHI, AGRM) |
| `workspace/paper/figure_sources/fig2/candidates/candidates.csv` | table | 2 number keys (F2DMP, F2DMV) |
| `workspace/paper/figures/F5_w_sweep_counts.csv` | table | 2 number keys (F5NDIFF, F5CLIPN) |
| `workspace/experiments/relax_novelty/results/inputs_manifest_additional.csv` | table | 2 number keys (NALLO, NALHI) |
| `workspace/experiments/verification/dft_results_b/gb_baseline_t0_hi/OUTCAR` | DFT | 2 number keys (DFTVS642, DFTVP642) |
| `workspace/experiments/verification/scripts/dft_select.py` | DFT | 2 number keys (DFTQLO, DFTQHI) |
| `workspace/experiments/count_recheck/results/summary.txt` | table (text) | 2 number keys (BNEQH, BNEQV) |
| `workspace/tmp/vasp_census/vasp_census.py` | code | 2 number keys (DFTV642G, DFTV642B) |
| `workspace/experiments/data_rebuild/results/gamma_dedup.csv` | table | 2 number keys (GMEAN, GSD); figure F10 (input), F3 (input), F5 (input) |
| `workspace/experiments/range_targets/results/gen/proposed/w2.0/chunk00/run_meta.json` | table | 2 number keys (A06, GENPER) |
| `workspace/experiments/verification/dft_inputs/gb_proposed_t0_hi/INCAR` | DFT | 2 number keys (DFTENC, DFTEDIFF) |
| `workspace/scripts/config/config.py` | code | 2 number keys (FMAX, FIREMAX) |
| `workspace/experiments/dx_gamma_analysis/results/B_near_by_range.csv` | table | 2 number keys (DXRIN, DXROUT) |
| `workspace/paper/figure_sources/fig3/scripts/make_fig3.py` | code | 1 number keys (F3BIN); figure F3 (input) |
| `workspace/paper/figure_sources/fig4/distance_audit/Fig4_distance_audit.md` | report | 1 number keys (STLNN) |
| `workspace/paper/figures/F10_main_w0_counts.csv` | table | 1 number keys (F10NDIFF) |
| `workspace/paper/figures/scripts/fig_F5_si_w_sweep.py` | code | 1 number keys (F5YMAX); figure F5 (script) |
| `workspace/experiments/data_rebuild/scripts/stage_c_relax_chunk.py` | code | 1 number keys (DSFIX) |
| `workspace/experiments/range_targets/results/ckpt_resolved_proposed.json` | table | 1 number keys (EPOCHH) |
| `workspace/experiments/range_targets/results/ckpt_resolved_baseline.json` | table | 1 number keys (EPOCHV) |
| `workspace/experiments/relax_novelty/results/relax/main/proposed_w2.0/chunk02/relax_rows.csv` | table | 1 number keys (F1GPROPOSED) |
| `workspace/experiments/relax_novelty/results/relax/main/baseline_w2.0/chunk00/relax_rows.csv` | table | 1 number keys (F1FIXC) |
| `workspace/experiments/relax_novelty/results/relax/main/baseline_w2.0/chunk02/relax_rows.csv` | table | 1 number keys (F1GNEQ) |
| `workspace/experiments/relax_novelty/results/relax/main/baseline_w2.0/chunk03/relax_rows.csv` | table | 1 number keys (F1GNOGB) |
| `workspace/experiments/relax_novelty/results/relax_novelty_summary.json` | table | 1 number keys (N05) |
| `workspace/experiments/relax_novelty/scripts/plot_g2_overview_manuscript.py` | code | 1 number keys (P19) |
| `workspace/experiments/relax_novelty/scripts/plot_relax_novelty_grid_manuscript.py` | code | 1 number keys (P20) |
| `workspace/experiments/relax_novelty/scripts/summarize_g2_and_e.py` | code | 1 number keys (P18) |
| `workspace/experiments/verification/results/low_gamma_novel_view/README.md` | report | 1 number keys (LGT5) |
| `workspace/experiments/verification/results/summary_tables/m7_total_energy.csv` | table | 1 number keys (DFTTE) |
| `workspace/experiments/verification/results/summary_tables/repro_checks.csv` | table | 1 number keys (F1MOVED) |
| `workspace/experiments/verification/results/summary_tables/scripts/summarize_tables.py` | code | 1 number keys (TSCL) |
| `workspace/experiments/verification/scripts/grid_interp.py` | code | 1 number keys (NTEST) |
| `workspace/counts/near_novel_count.out` | table (text) | 1 number keys (NVNOV) |
| `workspace/tmp/nonal5_equiv/pre_equiv_by_layer.csv` | table | 1 number keys (EQPRE) |
| `tmp/cnid_grid_sweep/cnid_grid_sweep.py` | code | 1 number keys (CNIDG) |
| `workspace/experiments/verification/scripts/criteria.py` | code | 1 number keys (LHRE) |
| `workspace/experiments/data_rebuild/results/stage_b_summary.json` | table | 1 number keys (DSRAW) |
| `workspace/experiments/data_rebuild/results/stage_d2_summary.json` | table | 1 number keys (DSDUP) |
| `workspace/experiments/verification/results/gb_stage2/gb_stage2_validation.json` | table | 1 number keys (THINL) |
| `workspace/experiments/relax_novelty/scripts/sbatch_n.sh` | code | 1 number keys (PMGVER) |
| `workspace/paper/figure_sources/fig2/render/Fig2.svg` | figure | figure F2 (input) |
| `workspace/paper/figure_sources/fig2/render/scripts/make_fig2.py` | code | figure F2 (input) |
| `workspace/paper/figure_sources/fig2/shift/shift_primary.json` | table | figure F2 (input) |
| `workspace/paper/figure_sources/fig2/scripts/common.py` | code | figure F2 (input) |
| `workspace/paper/figure_sources/fig3/Fig3.svg` | figure | figure F3 (input) |
| `workspace/paper/figure_sources/fig4/data/fig4_rows.csv` | table | figure F4 (input) |
| `workspace/paper/figure_sources/fig4/render/Fig4.svg` | figure | figure F4 (input) |
| `workspace/paper/figure_sources/fig4/render/scripts/make_fig4.py` | code | figure F4 (input) |
| `workspace/paper/figure_sources/fig4/distance_audit/data/Fig4_distance_rows.csv` | table | figure F4 (input), F7 (input) |
| `workspace/paper/figures/png/F10_si_main_w0_tracking.png` | figure | figure F10 (png) |
| `workspace/paper/figures/png/F1_schematic.png` | figure | figure F1 (png) |
| `workspace/paper/figures/png/F2_validity.png` | figure | figure F2 (png) |
| `workspace/paper/figures/png/F3_tracking.png` | figure | figure F3 (png) |
| `workspace/paper/figures/png/F4_novelty.png` | figure | figure F4 (png) |
| `workspace/paper/figures/png/F5_si_w_sweep_tracking.png` | figure | figure F5 (png) |
| `workspace/paper/figures/png/F6_si_dft_vs_mlip.png` | DFT | figure F6 (png) |
| `workspace/paper/figures/png/F7_si_representative_structures.png` | figure | figure F7 (png) |
| `workspace/paper/figures/png/F8_si_dx_distribution.png` | figure | figure F8 (png) |
| `workspace/paper/figures/png/F9_si_dmax_ecdf.png` | figure | figure F9 (png) |
| `workspace/paper/figures/scripts/fig_F10_si_main_w0_tracking.py` | code | figure F10 (script) |
| `workspace/paper/figures/scripts/fig_F1_schematic.py` | code | figure F1 (script) |
| `workspace/paper/figures/scripts/fig_F2_validity.py` | code | figure F2 (script) |
| `workspace/paper/figures/scripts/fig_F3_tracking.py` | code | figure F3 (script) |
| `workspace/paper/figures/scripts/fig_F4_novelty.py` | code | figure F4 (script) |
| `workspace/paper/figures/scripts/fig_F6_si_dft_vs_mlip.py` | DFT | figure F6 (script) |
| `workspace/paper/figures/scripts/fig_F7_si_representative_structures.py` | code | figure F7 (script) |
| `workspace/paper/figures/scripts/fig_F8_si_dx_distribution.py` | code | figure F8 (script) |
| `workspace/paper/figures/scripts/fig_F9_si_dmax_ecdf.py` | code | figure F9 (script) |
| `workspace/experiments/relax_novelty/results/relax/main/*/chunk*/relax_rows.csv  (28 files)` | table | figure F2 (input) |
| `workspace/experiments/relax_novelty/results/relax/main/*/chunk*/relaxed.extxyz  (28 files)` | structures | figure F2 (input) |
| `workspace/experiments/relax_novelty/results/relax/main/proposed_w2.0/chunk*/relax_rows.csv  (14 files)` | table | figure F10 (input), F3 (input) |
| `workspace/experiments/relax_novelty/results/relax/additional/proposed_w*/chunk*/relax_rows.csv  (56 files)` | table | figure F5 (input) |
| `workspace/experiments/verification/results/dft_recalc/gamma_vasp.csv` | DFT | figure F6 (input) |
| `workspace/experiments/verification/results/gb_equivalence/*  (44 files)` | other | figure F7 (input) |
| `workspace/experiments/verification/results/grid_interp/G_list.csv` | table | figure F7 (input) |
| `workspace/experiments/verification/results/grid_interp/nn_G_[0-9][0-9].csv  (10 files)` | table | figure F7 (input) |
| `workspace/experiments/verification/results/low_gamma_novel_view/nn/nn_L.csv` | table | figure F7 (input) |
| `workspace/experiments/verification/results/low_gamma_novel_view/table.csv` | table | figure F7 (input) |
| `workspace/experiments/verification/results/relax/proposed_w0/proposed_w0.0/chunk*/relax_rows.csv  (14 files)` | table | figure F10 (input), F3 (input) |
| `workspace/experiments/dx_gamma_analysis/results/merged_proposed_w2.csv` | table | figure F3 (input), F8 (input) |
| `workspace/experiments/dx_gamma_analysis/results/train_ref.csv` | table | figure F3 (input) |

Some number keys of the ledger cite files of the data record instead: `weights/proposed_resolved_config.yaml` (EPMAX, VALEP, PUNCOND).
Nine number keys (TGTNI, TGTNE, TGTQ, TGTDW, P09b, P22lo, P22hi, PC9, LSEED) record settings of the experiment design; their only source is an internal working document (design notes, a decision note or a model specification) that is not distributed. Bulk data (generated and relaxed structures, DFT outputs) are in the data record, not in this directory.

## Paths inside data files

Text, CSV and JSON files that held an absolute path of the original storage now hold a placeholder: `${GBREL_DATA}/...` (the root of the extracted data record),
`${GBREL_WORK}/...`, or `<ORIGINAL_ROOT>/...` (a file that is not part of the release). Where a program opens such a path, expand the placeholder first, for example

```bash
export GBREL_DATA=/path/to/extracted/data-record
envsubst '${GBREL_DATA}' < $GBREL_DATA/training/gamma_dedup.csv > gamma_dedup.expanded.csv
```

(`envsubst` is part of GNU gettext.) Only the `dir` column of the training CSV files and a few path columns of the tables need this.
