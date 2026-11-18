# PathSnap

**UE24CS352A — Machine Learning, PES University**

PathSnap matches noisy GPS observations to directed road segments using a trained candidate classifier, then reconstructs connected routes with an HMM and Viterbi decoder. A FastAPI backend serves live predictions and an evaluator dashboard for the two-laptop demonstration.

## Team and contributions

| SRN | Student name | Contribution |
| --- | --- | --- |
| PES1UG24CS096 | ATIKSH GOUR | Dataset exploration, validation, preprocessing and candidate generation |
| PES1UG24CS097 | ATISHAY JAIN | Feature engineering, classifier comparison and Bayesian fine-tuning |
| PES1UG24CS102 | AYAAN AHMED | HMM/Viterbi decoding, route evaluation and Folium visualizations |
| PES1UG24CS103 | AYUSH M SHETTY | FastAPI backend, evaluator dashboard and two-laptop deployment |

## Repository structure

```text
pathsnap/
  data_preprocessing/      Raw-file parsing, validation, audit, candidates and labels
  feature_selection/       The 20 engineered features and preprocessing settings
  classifier_selection/    Five classifiers, grouped CV, metrics and model export
  hyperparameter_tuning/   Bayesian search spaces and Gaussian-process optimisation
  hmm_training/            Decoder calibration, directed transitions and Viterbi
  visualization/           Offline Folium maps and experiment comparison plots
  api/                     FastAPI endpoints and live model loading
  ui/                      Dashboard HTML, CSS and JavaScript
  vendor/leaflet/           Licensed assets used by the offline Folium maps
artifacts/
  classifier/              Saved model and preprocessing specification
  hmm/                     Selected decoder configuration and model checksum
results/
  classifier_selection/    Five-model comparison, CV folds, predictions and run log
  hyperparameter_tuning/   Bayesian trial history, search configuration and figure
  hmm_training/            Classifier-versus-decoder metrics and calibration trials
data/                      Raw-data layout instructions; local data are ignored
docs/                      Methods, deployment, assessment PDFs and demo screenshots
tests/                     Synthetic pipeline and API regression checks
```

Workflow: **preprocessing → feature engineering/selection → classifier comparison → Bayesian tuning → HMM calibration and decoding → API/UI evaluation**. The saved model retains all 20 engineered features. HMM calibration selects transition scale and emission weight.

## Install

Run commands from this repository root. The saved experiment used Python 3.12.10. Its core dependency versions are pinned in [requirements.txt](requirements.txt); Python 3.12 is recommended for reproduction.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`. The Python module commands below otherwise work the same way.

The supplied student and evaluator ZIPs are included under `data/archives/`. Extract the student data before running development commands:

```powershell
Expand-Archive -LiteralPath "data/archives/Student folder.zip" -DestinationPath outputs/student-extracted
New-Item -ItemType Directory -Force data/raw
Move-Item -LiteralPath "outputs/student-extracted/Student folder/train", "outputs/student-extracted/Student folder/student_test" -Destination data/raw
```

This places `train/` (80 trips) and `student_test/` (10 trips) under `data/raw/`. Each trip contains matching `<trip_id>.track`, `.nodes`, `.arcs`, and `.route` files. See [data layout and formats](data/README.md). The evaluator archive is loaded only by the backend for final evaluation.

## Run the workflow

Examples write regenerated output under ignored `outputs/`, preserving the recorded evidence. Replace example trip IDs with a trip present in your data.

### 1. Data preprocessing and exploration

```powershell
python -m pathsnap.data_preprocessing.prepare data/raw --output outputs/prepared --map-trip 00000002
```

This exports candidate tables, aligned reference labels, nearest-road predictions, per-trip quality summaries, split metrics, a dataset profile figure, and the selected trip's Folium map. Use a new output directory for each preparation run.

For one trip, including baseline evaluation and an offline map:

```powershell
python -m pathsnap data/raw/train/00000002 --output outputs/baseline/00000002 --evaluate --map
```

### 2–4. Features, classifier selection and Bayesian fine-tuning

```powershell
python -m pathsnap.classifier_selection.experiments --data-root data/raw --output outputs/reproduced/classifier_selection --tuning-output outputs/reproduced/hyperparameter_tuning --artifacts outputs/reproduced/artifacts/classifier --seed 42 --trials 12 --threads 4
python -m pathsnap.visualization.plots --results outputs/reproduced/classifier_selection --tuning-results outputs/reproduced/hyperparameter_tuning
```

The runner compares logistic regression, Gaussian Naive Bayes, decision tree, Extra Trees and histogram gradient boosting. It selects a family by mean candidate Macro F1 across five training folds grouped by trip, runs Bayesian optimisation on the same folds, and refits on all 80 training trips. Imputation and scaling are learned inside each fit. Fixed models are then compared on the 10 student-test trips. See [modeling and feature definitions](docs/modeling.md) and the [recorded comparison](results/classifier_selection/RESULTS.md).

### 5. HMM calibration and route reconstruction

```powershell
python -m pathsnap.hmm_training.experiments --data-root data/raw --model outputs/reproduced/artifacts/classifier/model.joblib --output outputs/reproduced/hmm_training --artifacts outputs/reproduced/artifacts/hmm --map-trip 00000000
```

This compares nine decoder configurations on student-test trips and records classifier-only versus Viterbi results. The Folium comparison map includes GPS readings, reference, classifier and decoded routes. Student-test is used for calibration, so the selected score is not an independent final-test estimate.

Use the checked-in model and decoder configuration for one trip:

```powershell
python -m pathsnap.hmm_training.decode data/raw/student_test/00000000 --config artifacts/hmm/decoder_config.json --output outputs/decoded/00000000
```

See [HMM implementation and metrics](docs/hmm.md). Candidate scoring without decoding is available through `python -m pathsnap.classifier_selection.score --help`.

### 6. API, UI and two-laptop deployment

On Laptop 1:

```powershell
python -m pathsnap.api.server --evaluator data/archives/evaluator_test.zip --host 0.0.0.0 --port 8000
```

On Laptop 2, open `http://LAPTOP_1_IPV4:8000/` on the same network, select a trip and run evaluation. The backend defaults to the saved classifier and checksum-matched decoder configuration. Predictions use GPS and road-network inputs; reference routes are opened afterwards for server-side scoring. Reference geometry and labels are not sent to the dashboard.

See [deployment instructions](docs/deployment.md) for Google Maps setup, API requests, network configuration and demonstration evidence. `.env.example` lists optional map settings; set them in the shell because the server does not automatically load `.env` files.

## Recorded results

These values come from retained experiment files; repository reorganization did not retrain the models.

| Result | Value |
| --- | ---: |
| Tuned classifier training-CV candidate Macro F1 | 0.9326 |
| Tuned classifier student-test candidate Macro F1 | 0.9324 |
| Nearest-road student-test observation Top-1 | 53.72% |
| Tuned classifier student-test observation Top-1 | 89.26% |
| Classifier + HMM student-test observation Top-1 | 92.53% |
| Classifier + HMM student-test directed reachability | 99.89% |

Classifier candidate F1 uses a probability threshold; HMM candidate F1 uses hard Top-1 assignments. Top-1 uses aligned observations, including candidate misses. Reference labels are geometric alignment estimates, not independently verified pointwise ground truth.

## Verification and assessment material

```powershell
python -m unittest discover -s tests -v
```

Checks use synthetic raw trips and the saved artifact without opening reserved evaluator data. Full reproduction uses the included student archive.

The original [submission requirements](docs/assessment/Deliverables.pdf), [model and training guidelines](docs/assessment/Models%20and%20Training%20Guidelines.pdf), [dataset guide](docs/assessment/Dataset%20Exploration.pdf), [main guidelines](docs/assessment/Main%20Guidelines.pdf) and [assessment policy](docs/assessment/Assessment%20and%20Plagiarism%20Policy.pdf) are included under `docs/assessment/`. Dashboard screenshots, including the [demo run](docs/evidence/demo_run.jpeg), are preserved under `docs/evidence/`.

Code and documentation have received AI assistance. Retained metrics are from recorded executions; team members must verify their explanations and disclose assistance as required by the assessment policy.
