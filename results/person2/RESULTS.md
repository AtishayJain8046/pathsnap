# PathSnap Person 2 results

Five classifiers were fitted using fivefold trip-grouped cross-validation on 80 training trips. The fixed configurations were evaluated on 10 student-test trips with 19,160 GPS observations. 19,133 observations had a reference alignment (99.86% coverage). The aligned arc was absent for 0.28% of those observations.

| Model | CV Macro F1 | Student-test Macro F1 | Student-test Top-1 | Candidate log loss | Provisional continuity | Inference per 10 trips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.9095 ± 0.0174 | 0.9075 | 86.71% | 0.1399 | 93.41% | 80.7 ms |
| Gaussian Naive Bayes | 0.8601 ± 0.0173 | 0.8489 | 83.32% | 0.7212 | 92.70% | 131.2 ms |
| Decision tree | 0.9262 ± 0.0168 | 0.9289 | 88.92% | 0.1270 | 94.45% | 64.1 ms |
| Extra Trees | 0.9302 ± 0.0158 | 0.9282 | 88.18% | 0.0984 | 94.17% | 2175.3 ms |
| Histogram gradient boosting | 0.9320 ± 0.0170 | 0.9317 | 89.18% | 0.0956 | 95.16% | 2918.1 ms |
| Histogram gradient boosting (tuned) | 0.9326 ± 0.0168 | 0.9324 | 89.26% | 0.0948 | 95.14% | 2374.9 ms |
| Nearest road | N/A | N/A | 53.72% | N/A | 82.40% | N/A |

The initial CV comparison selected histogram gradient boosting. The Bayesian search took 14.5 minutes. Trial 9 gave the best mean CV Macro F1 of 0.93261.

On student-test, tuning changed candidate Macro F1 by +0.00064, Top-1 accuracy by +0.08 percentage points, log loss by -0.00073, and median inference by -543.2 ms.
Histogram gradient boosting (tuned) had the highest observed student-test Top-1 at 89.26%.

## Tuned settings

| Hyperparameter | Best value |
| --- | ---: |
| max_iter | 138 |
| max_leaf_nodes | 53 |
| learning_rate | 0.073871295 |
| min_samples_leaf | 37 |
| l2_regularization | 0.74414929 |

The CV column reports mean ± sample standard deviation across five folds. Inference is the median of seven one-thread scoring passes with features already computed. The nearest-road baseline ranks by distance, so its fitted-model metrics and timing are N/A.

Untuned boosting took 1.69–4.91 s per pass, and tuned boosting took 2.30–2.65 s. The timing ranges overlap, so the median difference describes this run.

Continuity uses provisional directed adjacency while Person 3's metric is pending. The per-model predictions are available for rescoring.

[Method and decoder handoff](../../PERSON2.md) · [Full experiment CSV](model_comparison.csv) · [Bayesian trial CSV](bayesian_trials.csv) · [Model comparison figure](model_comparison.png) · [Bayesian search figure](bayesian_search.png)
