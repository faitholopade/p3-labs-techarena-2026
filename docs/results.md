# Results

The organizer's 9 November 2026 announcement listed P3 Labs ninth among the 15
teams advancing from Phase 1 to Phase 2. The scores below are from the supplied
competition portal records.

| Submission | ROC AUC |
| --- | ---: |
| V38 | 0.832634 |
| V65 | 0.832700 |
| V79 | 0.832715 |
| V108 | 0.832723 |
| V129 | 0.832679 |
| V136 | 0.832744 |
| V142 | **0.832748** |

Selected original records are in [results/](../results). They retain the dates,
rank snapshots and validation decisions recorded at the time. An earlier rank
in one of those files is not the later official qualification outcome.

V142 improved V136 by 0.000004. V129 was a regression, and the recovered V143–V152
experiments produced no replacement submission. Their
[summary](../results/v143_v152_summary.json) retains the unsuccessful screens.

The local gains were often small and measured on reused development cohorts.
Later experiments sometimes used proxy baselines rather than exact held-out
predictions from the complete ensemble. Those results helped choose experiments,
but did not reliably predict leaderboard gains.

## Reproduction work

A 7 November recovery recomputed V65 from saved models over 529,390 evaluation
rows. Every row ID matched. The maximum probability difference from the scored
file was 7.28e-8, within a fixed 1e-7 tolerance; the CSV bytes differed. That run
took 381.99 seconds in its CPU environment. See the
[record](../results/recovery/v65_fresh_command.json).

V142 replay reproduced the original CSV exactly from saved V136 probabilities
and the saved correction. See its [record](../results/recovery/v142_replay.json).
Neither check retrained the complete model stack.

The historical held-user benchmark had a larger reported maximum difference,
0.0014441895435540197, which remains unresolved. It is not explained away by the
smaller V65 replay error. The proposed Sobol/PyTorch explanation was unconfirmed.
