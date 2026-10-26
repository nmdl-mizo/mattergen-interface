# gb_symfit stage 0 and stage 1 aggregation

- Generated: 2026-10-08 10:52:14
- Rows: 700 (= 350 x 2. Checked at start. Duplicate keys: 0)
- Reference orientations (measured value of W8, deg mod 90): 18.491 / 71.509
- additional layer. Additional application after theta=0.064 A was frozen. The counts of G2 and E are in gb_symfit_additional_g2_and_e.md

## 1. Counts of the classes (status): arm x w x stage (the denominator is the requested number = number of manifest rows per (arm,w): baseline w=2.0: 350)

| arm / w | stage | fitted | no_gb | multi_gb | disordered | axis_mismatch | non_al | excluded_pre | error | total | n_req |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline w=2.0 | pre | 76 | 260 | 0 | 0 | 0 | 0 | 14 | 0 | 350 | 350 |
| baseline w=2.0 | post | 76 | 260 | 0 | 0 | 0 | 0 | 14 | 0 | 350 | 350 |

Number of error rows (reported only): 0 (of all 700 rows)

## 2. Quantiles of gb_symfit_max_A [A] (status=fitted only; structures judged to have n_gb=2)

| arm / w | stage | n | min | 25% | 50% | 75% | 90% | 99% | max |
|---|---|---|---|---|---|---|---|---|---|
| baseline w=2.0 | pre | 76 | 5.326e-02 | 2.434e+00 | 2.747e+00 | 3.357e+00 | 3.888e+00 | 4.877e+00 | 4.998e+00 |
| baseline w=2.0 | post | 76 | 2.397e-02 | 2.304e+00 | 2.617e+00 | 3.206e+00 | 3.638e+00 | 7.441e+00 | 1.562e+01 |

## 3. Breakdown of the adopted operations (op_name) (fitted only)

| arm / w | stage | C2n | m_p | C2t | inv |
|---|---|---|---|---|---|
| baseline w=2.0 | pre | 14 | 10 | 24 | 28 |
| baseline w=2.0 | post | 13 | 14 | 20 | 29 |

Median of the max residual per operation [A] (fitted only; measured classification of the table on paper):

| arm / w | stage | C2n | m_p | C2t | inv |
|---|---|---|---|---|---|
| baseline w=2.0 | pre | 3.412e+00 | 3.413e+00 | 2.861e+00 | 2.884e+00 |
| baseline w=2.0 | post | 3.213e+00 | 3.204e+00 | 2.859e+00 | 2.898e+00 |

## 4. Distribution of tau_n (tau_n mod 1 of the adopted operation; fitted only; counts in steps of 0.1)

Note: tau_n of the operations that reverse the direction of n (C2t, inv) depends on the choice of the origin (because the n component of R is -1). tau_n of C2n and m_p does not depend on the origin.

| arm / w | stage | [0.0,0.1) | [0.1,0.2) | [0.2,0.3) | [0.3,0.4) | [0.4,0.5) | [0.5,0.6) | [0.6,0.7) | [0.7,0.8) | [0.8,0.9) | [0.9,1.0) | 0.5±1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline w=2.0 | pre | 7 | 5 | 5 | 7 | 7 | 25 | 5 | 5 | 5 | 5 | 15 |
| baseline w=2.0 | post | 9 | 4 | 3 | 6 | 8 | 27 | 3 | 7 | 6 | 3 | 18 |

## 5. Cross table of the suspected grain-boundary loss (gamma_relaxed < training minimum of gamma 0.5302 J/m2) and gb0_status

gamma_relaxed is the value of relax_rows.csv (the value of the same structure was attached also to the pre rows). For rows with empty gb0_status, status is used.

| arm / w | stage | suspected | two_gb | no_gb | multi_gb | disordered | error | (stage 0 not reached) |
|---|---|---|---|---|---|---|---|---|
| baseline w=2.0 | pre | suspected | 0 | 94 | 0 | 0 | 0 | 0 |
| baseline w=2.0 | pre | not suspected | 76 | 166 | 0 | 0 | 0 | 0 |
| baseline w=2.0 | post | suspected | 0 | 94 | 0 | 0 | 0 | 0 |
| baseline w=2.0 | post | not suspected | 76 | 166 | 0 | 0 | 0 | 0 |

## 6. Monitoring columns of the no_gb rows (production version of the measurement of W10; not used for decisions)

| arm / w | stage | n(no_gb) | median of the max residual of inv [A] | median of the max residual of C2t [A] |
|---|---|---|---|---|
| baseline w=2.0 | pre | 260 | 2.903e+00 | 2.926e+00 |
| baseline w=2.0 | post | 260 | 2.801e+00 | 2.720e+00 |

