# gb_symfit stage 0 and stage 1 aggregation

- Generated: 2026-10-06 22:51:49
- Rows: 1400 (= 700 x 2. Checked at start. Duplicate keys: 0)
- Reference orientations (measured value of W8, deg mod 90): 18.491 / 71.509
- theta is not frozen. **The counts of EQUIV/INEQUIV are not given**. Only the quantile table is looked at. The equivalence rate before relaxation is not used for claims.

## 1. Counts of the classes (status): arm x stage (the denominator is the requested number = 350)

| arm | stage | fitted | no_gb | multi_gb | disordered | axis_mismatch | non_al | excluded_pre | error | total |
|---|---|---|---|---|---|---|---|---|---|---|
| proposed | pre | 350 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 350 |
| proposed | post | 350 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 350 |
| baseline | pre | 74 | 256 | 0 | 0 | 0 | 0 | 20 | 0 | 350 |
| baseline | post | 74 | 256 | 0 | 0 | 0 | 0 | 20 | 0 | 350 |

## 2. Quantiles of gb_symfit_max_A [A] (status=fitted only; structures judged to have n_gb=2)

| arm | stage | n | min | 25% | 50% | 75% | 90% | 99% | max |
|---|---|---|---|---|---|---|---|---|---|
| proposed | pre | 350 | 3.087e-06 | 3.361e-06 | 3.800e-06 | 4.868e-06 | 5.448e-06 | 6.769e-06 | 6.851e-06 |
| proposed | post | 350 | 1.682e-06 | 3.345e-06 | 3.872e-06 | 4.898e-06 | 5.530e-06 | 6.807e-06 | 8.485e-06 |
| baseline | pre | 74 | 3.747e-01 | 2.514e+00 | 2.893e+00 | 3.475e+00 | 3.931e+00 | 4.556e+00 | 4.613e+00 |
| baseline | post | 74 | 2.063e-01 | 2.319e+00 | 2.851e+00 | 3.403e+00 | 3.787e+00 | 1.327e+01 | 1.538e+01 |

## 3. Breakdown of the adopted operations (op_name) (fitted only)

| arm | stage | C2n | m_p | C2t | inv |
|---|---|---|---|---|---|
| proposed | pre | 350 | 0 | 0 | 0 |
| proposed | post | 350 | 0 | 0 | 0 |
| baseline | pre | 18 | 9 | 20 | 27 |
| baseline | post | 16 | 10 | 17 | 31 |

Median of the max residual per operation [A] (fitted only; measured classification of the table on paper):

| arm | stage | C2n | m_p | C2t | inv |
|---|---|---|---|---|---|
| proposed | pre | 3.800e-06 | 1.419e-01 | 2.255e+00 | 2.324e+00 |
| proposed | post | 3.872e-06 | 1.402e-01 | 2.218e+00 | 2.288e+00 |
| baseline | pre | 3.530e+00 | 3.465e+00 | 3.185e+00 | 3.155e+00 |
| baseline | post | 3.320e+00 | 3.262e+00 | 3.291e+00 | 3.222e+00 |

## 4. Distribution of tau_n (tau_n mod 1 of the adopted operation; fitted only; counts in steps of 0.1)

Note: tau_n of the operations that reverse the direction of n (C2t, inv) depends on the choice of the origin (because the n component of R is -1). tau_n of C2n and m_p does not depend on the origin.

| arm | stage | [0.0,0.1) | [0.1,0.2) | [0.2,0.3) | [0.3,0.4) | [0.4,0.5) | [0.5,0.6) | [0.6,0.7) | [0.7,0.8) | [0.8,0.9) | [0.9,1.0) | 0.5±1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proposed | pre | 0 | 0 | 0 | 0 | 0 | 350 | 0 | 0 | 0 | 0 | 350 |
| proposed | post | 0 | 0 | 0 | 0 | 0 | 350 | 0 | 0 | 0 | 0 | 350 |
| baseline | pre | 5 | 6 | 5 | 3 | 9 | 33 | 2 | 2 | 4 | 5 | 14 |
| baseline | post | 4 | 6 | 6 | 2 | 14 | 25 | 6 | 2 | 6 | 3 | 14 |

## 5. Cross table of the suspected grain-boundary loss (gamma_relaxed < training minimum of gamma 0.5302 J/m2) and gb0_status

gamma_relaxed is the value of relax_rows.csv (the value of the same structure was attached also to the pre rows). For rows with empty gb0_status, status is used.

| arm | stage | suspected | two_gb | no_gb | multi_gb | disordered | error | (stage 0 not reached) |
|---|---|---|---|---|---|---|---|---|
| proposed | pre | suspected | 0 | 0 | 0 | 0 | 0 | 0 |
| proposed | pre | not suspected | 350 | 0 | 0 | 0 | 0 | 0 |
| proposed | post | suspected | 0 | 0 | 0 | 0 | 0 | 0 |
| proposed | post | not suspected | 350 | 0 | 0 | 0 | 0 | 0 |
| baseline | pre | suspected | 0 | 101 | 0 | 0 | 0 | 0 |
| baseline | pre | not suspected | 74 | 155 | 0 | 0 | 0 | 0 |
| baseline | post | suspected | 0 | 101 | 0 | 0 | 0 | 0 |
| baseline | post | not suspected | 74 | 155 | 0 | 0 | 0 | 0 |

## 6. Monitoring columns of the no_gb rows (production version of the measurement of W10; not used for decisions)

| arm | stage | n(no_gb) | median of the max residual of inv [A] | median of the max residual of C2t [A] |
|---|---|---|---|---|
| proposed | pre | 0 | - | - |
| proposed | post | 0 | - | - |
| baseline | pre | 256 | 2.836e+00 | 2.840e+00 |
| baseline | post | 256 | 2.774e+00 | 2.749e+00 |

