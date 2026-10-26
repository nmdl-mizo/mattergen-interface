# Records of the decisions of the novelty-instrument control (2026-10-08)

- Plan: novelty instrument positive control (r1)
- Jobs (all COMPLETED 0:0): stage S, pilot, production (array), stage A, determinism (array)
- Job verification (stage A log `logs/novctl_A.out`, `-t generic -m 'done rc=0 stage=A'`): `VERDICT: PASS (2/3)`
- Check: recounted independently from the 300 rows of rows_base00..19.csv (cells.csv not used)

| item | measured | verdict |
|---|---|---|
| C-2 | H x N1 novel 20/20 (max_disp_applied 0.448 to 0.732 A) | **PASS** |
| C-1 | all 14 positive cells n_matched = 20 (core 11, periphery 3), 0 not applied | **PASS** (rule 5) |
| D-1 | bases 20 rows, natoms 5 kinds x 4, reproduced by the rule | PASS |
| D-2 | 20 files x 15 rows = 300 rows, 20 DONE | PASS |
| D-3 | H x P0 novel=False 20/20 (match_is_source 17/20) | PASS |
| check-4 | H×N1 novel=True 20/20 | PASS |
| D-5 | route V applied 100/100, angle_fail 0 | PASS |
| D-6 | degeneracy check maximum rms H 6.27e-15, V 5.40e-15 (normalized values) | PASS |
| D-7 | self_check 7/7 MATCH. The exit code was not recorded and was FLAG; re-run and exit=0 with identical output was recorded in `self_check_exitcode.txt`, which resolved it | PASS |
| D-8 | determinism: sha256 agrees for base00 and base19 excluding elapsed_s | PASS |
| D-9 | canonicalize exceptions 0 (logs, rows) | PASS |
| D-10 | `constraints=0` for 20/20 tasks | PASS |

## Reference (not subject to decision)
- match_is_source = False for 3 cases at P0 (base 11: row 4697 -> 3455, base 14: 3724 -> 3704, base 17: 2803 -> 2783). In all of them a different training row matched, i.e. **there are pairs in the training set that the instrument judges to be identical** (it appears to be the same phenomenon as "5.0% of the dx-neighbor 0.15 A pairs are judged identical" in an earlier fact check; a guess). It does not affect the novelty decision (novel = False even if the matching target is not the source).
- At P7 (sigma = 0.01 A) the maximum normalized rms is 0.00725 and the maximum max_disp_applied is 0.0403 A. All agree.
- P3 (rotation), P4 (basis) and P5 (cyclic permutation of axes) of route H also all agree (absorbed by the lattice mapping of pymatgen and the Niggli reduction).

## Conclusion (facts only)
The production novelty instrument judged 20/20 as matching for translation of the training structures (with and without wrapping), atom order, rigid rotation, re-taking of the basis, cyclic permutation of axes, the two-fold rotation that connects the two grain boundaries and displacement within the tolerance, on both the after-relaxation route and the baseline before-relaxation route, and judged 20/20 as novel for displacement outside the tolerance.
