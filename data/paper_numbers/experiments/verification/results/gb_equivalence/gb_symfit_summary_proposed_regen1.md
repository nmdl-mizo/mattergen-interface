# gb_symfit stage 0 and stage 1 aggregation

- Generated: 2026-10-07 23:01:54
- Rows: 700 (= 350 x 2. Checked at start. Duplicate keys: 0)
- Reference orientations (measured value of W8, deg mod 90): 18.491 / 71.509
- additional layer. Additional application after theta=0.064 A was frozen. The counts of G2 and E are in gb_symfit_additional_g2_and_e.md

## 1. Counts of the classes (status): arm x w x stage (the denominator is the requested number = number of manifest rows per (arm,w): proposed w=2.0: 350)

| arm / w | stage | fitted | no_gb | multi_gb | disordered | axis_mismatch | non_al | excluded_pre | error | total | n_req |
|---|---|---|---|---|---|---|---|---|---|---|---|
| proposed w=2.0 | pre | 350 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 350 | 350 |
| proposed w=2.0 | post | 350 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 350 | 350 |

Number of error rows (reported only): 0 (of all 700 rows)

## 2. Quantiles of gb_symfit_max_A [A] (status=fitted only; structures judged to have n_gb=2)

| arm / w | stage | n | min | 25% | 50% | 75% | 90% | 99% | max |
|---|---|---|---|---|---|---|---|---|---|
| proposed w=2.0 | pre | 350 | 3.088e-06 | 3.349e-06 | 4.025e-06 | 4.986e-06 | 5.325e-06 | 6.669e-06 | 6.874e-06 |
| proposed w=2.0 | post | 350 | 2.159e-06 | 3.332e-06 | 4.143e-06 | 5.003e-06 | 5.669e-06 | 7.769e-06 | 1.163e-05 |

## 3. Breakdown of the adopted operations (op_name) (fitted only)

| arm / w | stage | C2n | m_p | C2t | inv |
|---|---|---|---|---|---|
| proposed w=2.0 | pre | 350 | 0 | 0 | 0 |
| proposed w=2.0 | post | 350 | 0 | 0 | 0 |

Median of the max residual per operation [A] (fitted only; measured classification of the table on paper):

| arm / w | stage | C2n | m_p | C2t | inv |
|---|---|---|---|---|---|
| proposed w=2.0 | pre | 4.025e-06 | 1.613e-01 | 2.292e+00 | 2.280e+00 |
| proposed w=2.0 | post | 4.143e-06 | 1.621e-01 | 2.284e+00 | 2.260e+00 |

## 4. Distribution of tau_n (tau_n mod 1 of the adopted operation; fitted only; counts in steps of 0.1)

Note: tau_n of the operations that reverse the direction of n (C2t, inv) depends on the choice of the origin (because the n component of R is -1). tau_n of C2n and m_p does not depend on the origin.

| arm / w | stage | [0.0,0.1) | [0.1,0.2) | [0.2,0.3) | [0.3,0.4) | [0.4,0.5) | [0.5,0.6) | [0.6,0.7) | [0.7,0.8) | [0.8,0.9) | [0.9,1.0) | 0.5±1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proposed w=2.0 | pre | 0 | 0 | 0 | 0 | 0 | 350 | 0 | 0 | 0 | 0 | 350 |
| proposed w=2.0 | post | 0 | 0 | 0 | 0 | 0 | 350 | 0 | 0 | 0 | 0 | 350 |

## 5. Cross table of the suspected grain-boundary loss (gamma_relaxed < training minimum of gamma 0.5302 J/m2) and gb0_status

gamma_relaxed is the value of relax_rows.csv (the value of the same structure was attached also to the pre rows). For rows with empty gb0_status, status is used.

| arm / w | stage | suspected | two_gb | no_gb | multi_gb | disordered | error | (stage 0 not reached) |
|---|---|---|---|---|---|---|---|---|
| proposed w=2.0 | pre | suspected | 0 | 0 | 0 | 0 | 0 | 0 |
| proposed w=2.0 | pre | not suspected | 350 | 0 | 0 | 0 | 0 | 0 |
| proposed w=2.0 | post | suspected | 0 | 0 | 0 | 0 | 0 | 0 |
| proposed w=2.0 | post | not suspected | 350 | 0 | 0 | 0 | 0 | 0 |

## 6. Monitoring columns of the no_gb rows (production version of the measurement of W10; not used for decisions)

| arm / w | stage | n(no_gb) | median of the max residual of inv [A] | median of the max residual of C2t [A] |
|---|---|---|---|---|
| proposed w=2.0 | pre | 0 | - | - |
| proposed w=2.0 | post | 0 | - | - |

