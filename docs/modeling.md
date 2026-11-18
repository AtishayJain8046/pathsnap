# Features, classifier selection and Bayesian optimisation

## Data and labels

Raw `.track`, `.nodes` and `.arcs` files supply prediction inputs. Longitude/latitude are projected to local metre coordinates for geometry. Candidates are sorted by point-to-segment distance within 50 m, with at most eight directed arcs per observation. Arc IDs are local to each trip and are not model features.

The ordered `.route` sequence is aligned to observations using a forward-only distance-based alignment with a 50 m cutoff. Labels are estimates: parallel arcs, junctions and repeated roads can be ambiguous. Supervised fitting excludes unaligned observations. Top-1 evaluation retains aligned observations whose correct arc is absent from the candidates, counting those predictions as errors.

Historical exploration reported 208,417 observations in 80 training trips and 19,160 in 10 student-test trips. The saved run inventories and fingerprints are in [run.json](../results/classifier_selection/run.json). Recreate detailed EDA tables and maps with the preparation command in the README. The former feature-only ZIP was not a labeled, trip-grouped training dataset.

## Feature engineering and selection

[Feature implementation](../pathsnap/feature_selection/features.py) fixes this ordered set of 20 features:

| Group | Features | Purpose |
| --- | --- | --- |
| Proximity and ranking | `distance_m`, `candidate_rank`, `distance_above_nearest_m`, `candidate_count` | Relative and absolute fit to GPS |
| Arc geometry | `arc_fraction`, `arc_length_m`, `distance_to_start_m`, `distance_to_end_m` | Position along the road and distance to junctions |
| Direction | `previous_heading_cosine`, `next_heading_cosine`, `window_heading_cosine` | Agreement of directed road and GPS movement |
| Motion | `previous_speed_mps`, `next_speed_mps` | Local movement rate |
| Heading validity | `previous_heading_available`, `next_heading_available`, `window_heading_available` | Identify unavailable movement direction |
| Neighbour consistency | `previous_excess_to_arc_m`, `next_excess_to_arc_m` | Fit to neighbouring observations |
| Connectivity | `incoming_degree`, `outgoing_degree` | Road-network structure at candidate endpoints |

Heading needs at least 3 m movement and a positive interval no longer than 10 s. The window extends up to four observations on either side. Features can use future observations because inference receives a completed trip; this is not a streaming predictor.

The saved model retains all 20 features in that fixed order. Imputation and scaling are learned within training folds, preserving trip grouping. The exported preprocessing specification records the feature schema used during inference.

## Five-model comparison

| Model | Initial settings | Objective |
| --- | --- | --- |
| Logistic regression | Median imputation, standard scaling, L-BFGS, L2, C=1 | Regularized binary log loss |
| Gaussian Naive Bayes | Median imputation, standard scaling, variance smoothing 1e-9 | Gaussian class-conditional distributions |
| Decision tree | Gini, depth 12, minimum leaf size 30 | Greedy Gini reduction |
| Extra Trees | 48 trees, depth 16, minimum leaf size 12, feature fraction 0.8 | Randomized splits with Gini reduction |
| Histogram gradient boosting | 100 iterations, 31 leaves, learning rate 0.1, minimum leaf size 30, L2=1 | Additive-tree binary log loss |

All 80 training trips participate in five `GroupKFold` folds grouped by trip ID. Estimators learn imputation and any scaling only on their fit partition. Initial family selection maximizes mean fold candidate Macro F1. The tuned CV score is a search score rather than an unbiased estimate after model selection. The 10 student-test trips provide development comparisons after fitting.

## Bayesian fine-tuning

[Search implementation and spaces](../pathsnap/hyperparameter_tuning/bayesian.py) define the search for each family. The saved run selected histogram gradient boosting and searched `max_iter`, `max_leaf_nodes`, `learning_rate`, `min_samples_leaf` and `l2_regularization`.

Twelve trials maximize the same training GroupKFold mean Macro F1. Four scrambled Sobol configurations initialize a Matérn Gaussian-process surrogate; later trials use expected improvement. Complete ranges, configurations, seeds, fold results and timings are in [bayesian_search.json](../results/hyperparameter_tuning/bayesian_search.json) and [bayesian_trials.csv](../results/hyperparameter_tuning/bayesian_trials.csv). The best trial was refitted on all training trips.

The runner writes classifier metrics under `--output`, search evidence under `--tuning-output`, and the model and preprocessing specification under `--artifacts`. The README directs all three to `outputs/reproduced/`. Checked-in artifacts remain under `artifacts/classifier/`.

## Metrics and saved evidence

- Candidate Macro F1 weights match and non-match classes equally at a probability threshold above 0.5. Candidate log loss uses binary probabilities and natural logs. Precision, recall and confusion counts are retained alongside them.
- Observation Top-1 selects the highest scoring candidate and divides correct predictions by all aligned observations. Ties preserve candidate distance order. Candidate miss rate and reference coverage are recorded separately.
- Observation log loss is conditional on the reference arc appearing among the candidates.
- Strict directed adjacency counts consecutive predictions on the same arc or sharing directed endpoints, for intervals no longer than 10 s. Missing predictions count as breaks. HMM evaluation additionally measures reachability through intermediate roads.
- Inference timing is the median of seven one-thread scoring passes after warm-up on precomputed features. It excludes raw parsing, feature extraction and HMM decoding; API timings have a different scope.

[Results summary](../results/classifier_selection/RESULTS.md), [comparison CSV](../results/classifier_selection/model_comparison.csv), [fold membership](../results/classifier_selection/cv_folds.json), [fold metrics](../results/classifier_selection/cv_metrics.csv), and [per-trip metrics](../results/classifier_selection/student_test_by_trip.csv) provide recorded evidence. `training.log` and `run.json` retain original historical command/module names; use the current README to execute reorganized modules.

## Classifier-to-decoder interface

```powershell
python -m pathsnap.classifier_selection.score data/raw/student_test/00000000 --output outputs/emissions/00000000
```

`emissions.csv` keys candidates by trip, observation, timestamp and arc ID. `match_probability` is the fitted binary classifier score. `log_emission` is clipped match odds normalized across that observation's candidates and then logged.

The Python interface is `CandidateScorer(Path("artifacts/classifier/model.joblib")).predict(trip)` in `pathsnap.classifier_selection.model`. It returns candidate arc IDs, Top-1 IDs, match probability grid and log-emission grid. Missing slots have arc ID `-1`, probability `NaN` and log emission `-inf`. Prediction needs only raw GPS and road-network files.

`pathsnap.classifier_selection.rescore_continuity` can rescore saved per-model predictions without fitting again; run it with `--help` for the metric callback interface.
