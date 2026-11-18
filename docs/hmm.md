# HMM calibration and Viterbi route reconstruction

The decoder uses the tuned histogram gradient boosting artifact in `artifacts/classifier/model.joblib`. Its configuration in `artifacts/hmm/decoder_config.json` contains a SHA-256 checksum tying it to that model. Hidden states are the directed candidate arcs for each GPS observation. Normalized binary match odds provide log emissions.

## Transitions and decoding

Directed road distance includes the remaining previous-arc length, the shortest directed path between endpoints, and progress along the arriving arc. Staying on the same arc uses along-arc progress, allowing up to 15 m backward motion for GPS jitter. Distances are capped by the smaller of 600 m and `55 m/s × elapsed seconds + 40 m`.

Viterbi maximizes weighted log emissions plus transition scores. A transition's log score is the negative absolute difference between road distance and GPS displacement divided by `beta_m`. Unreachable or over-budget transitions have score negative infinity. A time gap above 10 s, a missing candidate set, or an entirely disconnected next step starts a new segment. Unresolved observations retain blank predictions. Route reconstruction inserts intermediate directed arcs within each continuous segment.

This is a classifier-driven HMM with calibrated transition and emission parameters. No Baum–Welch fitting or learned transition matrix is claimed. [hmm.py](../pathsnap/hmm_training/hmm.py) implements the graph, lattice, recurrence and route reconstruction.

## Calibration and reproduction

`pathsnap.hmm_training.experiments` tries nine combinations of transition scale and emission weight on ten student-test trips. Highest observation Top-1 wins, with directed reachability as the tie-breaker. The saved settings are `beta_m=10`, `emission_weight=0.5`.

```powershell
python -m pathsnap.hmm_training.experiments --data-root data/raw --output outputs/hmm_training --artifacts outputs/artifacts/hmm --map-trip 00000000
python -m pathsnap.hmm_training.decode data/raw/student_test/00000000 --config artifacts/hmm/decoder_config.json --output outputs/decoded/00000000
```

The first command writes trials, aggregate and per-trip metrics, predictions, reconstructed routes, a run record and an offline Folium map. Configuration goes under `--artifacts`. The second uses the checked-in configuration and emits observation-level `predictions.csv` plus segmented `route.json`. `--model` can select another artifact; calibrate a matching decoder after retraining.

## Recorded comparison

| Student-test metric | Classifier only | Classifier + HMM |
| --- | ---: | ---: |
| Observation Top-1 | 89.26% | 92.53% |
| Strict directed adjacency | 95.14% | 98.21% |
| Directed reachability | 96.47% | 99.89% |

[summary.csv](../results/hmm_training/summary.csv), [by_trip.csv](../results/hmm_training/by_trip.csv) and [calibration trials](../results/hmm_training/student_test_decoder_trials.csv) retain the executed comparison. Student-test was used to select parameters, so these are development scores. Candidate F1 here uses hard Top-1 selections and differs from the classifier's probability-threshold F1.

Historical CSVs also contain reserved evaluator scores. They are retained as existing evaluation evidence and must not guide further selection. The optional `--evaluator-zip` runner argument is for authorised final evaluation only: it locks settings and computes predictions before opening evaluator references. Development commands in the README do not use that archive.

Reference labels are alignment estimates. High continuity does not establish that every decoded arc is correct. The final report should discuss per-trip cases where decoding helps or hurts using retained metrics and actual Folium maps.
