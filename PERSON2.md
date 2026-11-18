# Person 2 model experiments

This work builds on the team's [nearest-road baseline](https://github.com/AtishayJain8046/pathsnap/tree/baseline/nearest-road). It uses the existing trip parser, nearby candidate generator, and ordered reference alignment. The model ranks directed road arcs separately for each GPS observation. The decoder can use the same outputs as emissions.

## Reproduce the run

Use Python 3.12. Extract the organizer's `Student folder.zip` so that `Student folder/train` and `Student folder/student_test` are next to this file. From the project root, run:

```powershell
python -m pip install -r requirements-person2.txt
python -m pathsnap.experiments --data-root "Student folder" --output results/person2 --seed 42 --trials 12 --threads 4
python -m pip install matplotlib
python -m pathsnap.plot_experiments --results results/person2
```

All 80 training trips take part in five `GroupKFold` folds keyed by trip ID. Each fold's estimator learns its imputation and any scaling from its training rows. The initial comparison uses the mean fold Macro F1 to select the model family. Bayesian optimization then maximizes the same mean score over the same folds. The search starts with four scrambled Sobol configurations and uses a Matérn Gaussian process with expected improvement for the remaining eight trials. The selected configuration is refitted on all 80 trips, and the fixed configurations are scored on the 10 student-test trips. The tuned CV mean is a search score, so student-test performance is the separate comparison after optimization.

The candidate radius is 50 m, with at most eight arcs per observation. Features describe proximity, candidate rank, distance to arc endpoints, agreement with GPS movement, local speed, and road connectivity at the arc endpoints. Movement features may use the next GPS reading because the input is a completed trip. The ordered reference route is used only to derive labels for fitting or evaluation. Those labels are alignment estimates and can be ambiguous for nearby or repeated arcs. Full definitions and the feature column order are in `pathsnap/features.py` and `results/person2/preprocessing.json`.

## Model configurations

| Model | Initial settings | Fit objective |
| --- | --- | --- |
| Logistic regression | Median imputation, standard scaling, L-BFGS, L2 with C=1 | Regularized binary log loss |
| Gaussian Naive Bayes | Median imputation, standard scaling, variance smoothing 1e-9 | Fit class priors and per-class Gaussian feature distributions |
| Decision tree | Gini, maximum depth 12, minimum 30 samples per leaf | Greedy Gini impurity reduction |
| Extra Trees | 48 randomized trees, maximum depth 16, minimum 12 samples per leaf, 80% of features considered per split | Randomized splits scored by Gini impurity |
| Histogram gradient boosting | 100 iterations, 31 leaves, learning rate 0.1, minimum 30 samples per leaf, L2=1 | Binary log loss with additive trees |
| Nearest road | First candidate sorted by geometric distance | Distance-based comparison |

## Metrics and evidence

- **Candidate Macro F1** gives equal weight to the match and non-match classes at a fixed probability threshold above 0.5. Candidate log loss uses binary match probabilities and the natural logarithm. Candidate metrics include only rows from GPS observations with a reference alignment. A candidate miss contributes available negative rows and can also leave an observation with zero candidate rows.
- **Observation Top-1** picks the arc with the highest candidate probability and uses every aligned GPS observation in its denominator. A missing or wrong candidate prediction counts as an error. The additional observation log-loss column is conditional on the reference arc appearing in the candidate list. Reference coverage and candidate miss rate are recorded alongside these metrics.
- **Inference time** is the median of seven one-thread passes over the 10 student-test trips after one warm-up. It measures score prediction, top-1 selection, and emission normalization on precomputed features. The run separately logs trip parsing and feature time. The geometric baseline has no fitted-model timing, Macro F1, or log loss.
- **Provisional route continuity** pools consecutive pairs at most 10 seconds apart. A pair is continuous when both predictions resolve to the same directed arc or when the previous arc's end node equals the current arc's start node. Unmatched predictions count as breaks. Sparse samples can produce a break even if an intervening road connects the two arcs.

| File in `results/person2` | Contents |
| --- | --- |
| `model_comparison.csv` and `run.json` | Overall metrics, refit times, settings, data fingerprints, and inference timing repeats |
| `cv_folds.json` and `cv_metrics.csv` | Trip membership and fold-level metrics |
| `bayesian_search.json` and `bayesian_trials.csv` | Search space, 12 trial configurations, scores, timings, and best trial |
| `student_test_by_trip.csv` and `student_test_predictions.csv.gz` | Per-trip metrics and top-1 predictions by trip and observation index for all configurations |
| `model.joblib` and `preprocessing.json` | Fitted estimator, learned preprocessing, and feature settings |
| `model_comparison.png` and `bayesian_search.png` | Figures generated from the saved run |

## Decoder and API handoff

To score one trip and export one row per candidate, run:

```powershell
python -m pathsnap.score "Student folder/student_test/00000000" --output outputs/person2/example
```

`emissions.csv` identifies each candidate by trip, observation index, timestamp, and arc ID. The binary classifier's `match_probability` is the candidate-level score. `log_emission` is the clipped match odds normalized across that observation's candidates and then logged. It can be added to log transition scores in Viterbi. Empty candidate slots are absent from the CSV, and `predictions.csv` contains a blank arc ID for an unresolved observation.

The Python API is `CandidateScorer(Path("results/person2/model.joblib")).predict(trip)`. It returns the candidate arc grid, top-1 arc IDs, candidate probability grid, and log-emission grid. Candidate slots are in baseline distance order. An empty slot has arc ID `-1`, probability `NaN`, and log emission `-inf`. Probability ties keep the first candidate. The scorer only requires `.track`, `.nodes`, and `.arcs`.

Person 3 can provide a function that accepts `(trip, predicted_arc_ids)` and returns `(continuous_pairs, eligible_pairs)`. Apply it to all saved predictions with:

```powershell
python -m pathsnap.rescore_continuity --metric your_module:your_metric --output outputs/person2/team_continuity.csv
```

The resulting CSV contains a per-trip row and an `ALL` row for each model. To regenerate the main comparison with the team metric, pass `--continuity your_module:your_metric` to the experiment runner.

The experiment code and run were prepared with AI assistance. The recorded metrics come from the executed scripts. Team members should verify the analysis, explain the experiments in their own words, and disclose assistance as required by the organizer's assessment policy.
