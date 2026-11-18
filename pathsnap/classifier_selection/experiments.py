"""Run the PathSnap classifier comparison and Bayesian optimization."""

import argparse
import csv
import gzip
import importlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from pathsnap.classifier_selection.evaluation import benchmark, candidate_statistics, cross_validate, evaluate_partition, summarize_cv
from pathsnap.classifier_selection.models import MODELS, fit_model
from pathsnap.feature_selection.features import FEATURE_NAMES, Preprocessing
from pathsnap.classifier_selection.model import export_model, score_candidates
from pathsnap.classifier_selection.data import load_partition, training_rows
from pathsnap.hyperparameter_tuning.bayesian import SEARCH_SPACES, bayesian_search

DEFAULT_CONTINUITY = "pathsnap.classifier_selection.continuity:directed_adjacency"
FOLDS = 5
REFERENCE_MAX_DISTANCE_M = 50.0


def metric_function(spec):
    module_name, _, function_name = spec.partition(":")
    return getattr(importlib.import_module(module_name), function_name)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    open_csv = gzip.open if path.suffix == ".gz" else open
    with open_csv(path, "wt", encoding="utf-8", newline="") as file:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows(model, data, predicted):
    return [
        {"model": model, "trip_id": data.trip_id, "observation_index": i,
         "predicted_arc_id": int(arc) if arc >= 0 else ""}
        for i, arc in enumerate(predicted)
    ]


def run(data_root, output, tuning_output, artifacts, seed, trials, threads, continuity_spec):
    output.mkdir(parents=True, exist_ok=True)
    tuning_output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    preprocessing = Preprocessing()
    continuity = metric_function(continuity_spec)
    started = perf_counter()
    print("Preparing 80 training trips", flush=True)
    training_trips = load_partition(data_root / "train", 80, preprocessing, REFERENCE_MAX_DISTANCE_M)
    features, targets, groups = training_rows(training_trips)
    if len(np.unique(groups)) != len(training_trips):
        raise ValueError("Every training trip needs labeled candidate rows for grouped CV")
    splits = list(GroupKFold(n_splits=FOLDS).split(features, targets, groups))
    folds = []
    for number, (train, validation) in enumerate(splits, 1):
        train_ids = sorted(set(groups[train]))
        validation_ids = sorted(set(groups[validation]))
        if set(train_ids) & set(validation_ids):
            raise ValueError("Trip crossover in grouped CV")
        folds.append({"fold": number, "train_trip_ids": train_ids, "validation_trip_ids": validation_ids,
                      "train_candidate_rows": len(train), "validation_candidate_rows": len(validation)})
    write_json(output / "cv_folds.json", folds)
    print(f"Training data ready: {len(features):,} labeled candidate rows. Starting grouped CV.", flush=True)

    def evaluate(model, parameters):
        return cross_validate(model, parameters, seed, features, targets, splits)

    entries, cv_rows = [], []
    with threadpool_limits(limits=threads):
        for model in MODELS:
            rows = evaluate(model, {})
            summary = summarize_cv(rows)
            entries.append((model.name, model, {}, summary))
            cv_rows.extend({"model": model.name, **row} for row in rows)
            print(f"{model.name}: grouped CV Macro F1 {summary['cv_macro_f1_mean']:.5f}", flush=True)
        selected = max(entries, key=lambda entry: entry[3]["cv_macro_f1_mean"])
        model = selected[1]
        print(f"Selected {model.name} for Bayesian tuning using training CV.", flush=True)
        search_start = perf_counter()
        search_space = SEARCH_SPACES[model.name]
        search = bayesian_search(model, search_space, evaluate, trials, seed)
        search_seconds = perf_counter() - search_start

    best = max(search, key=lambda trial: trial["cv_macro_f1_mean"])
    tuned_name = f"{model.name}_tuned"
    entries.append((tuned_name, model, best["parameters"], summarize_cv(best["folds"])))
    cv_rows.extend({"model": tuned_name, **row} for row in best["folds"])
    write_csv(output / "cv_metrics.csv", cv_rows)
    write_json(tuning_output / "bayesian_search.json", {
        "selected_model": model.name, "best_trial": best["trial"], "seed": seed,
        "objective": "maximize mean five-fold training GroupKFold candidate Macro F1",
        "surrogate": "GaussianProcessRegressor with Matérn 2.5 kernel, length scale 0.35",
        "acquisition": "Expected improvement with exploration margin 0.0001 over a scrambled Sobol pool",
        "warmup_trials": 4, "seconds": search_seconds,
        "search_space": [asdict(parameter) for parameter in search_space], "trials": search,
    })
    write_csv(tuning_output / "bayesian_trials.csv", [
        {**{key: value for key, value in trial.items() if key not in ("folds", "parameters")},
         **trial["parameters"]} for trial in search
    ])

    print("Preparing 10 student-test trips and evaluating the fixed configurations", flush=True)
    student_test = load_partition(data_root / "student_test", 10, preprocessing, REFERENCE_MAX_DISTANCE_M)
    if {data.trip_id for data in training_trips} & {data.trip_id for data in student_test}:
        raise ValueError("Training and student-test trip IDs overlap")

    comparison, per_trip_rows, predicted_rows = [], [], []
    for name, definition, overrides, summary in [*entries, ("nearest_road", None, {}, {})]:
        estimator, fit_seconds, train_f1 = None, None, None
        if definition:
            with threadpool_limits(limits=threads):
                estimator, fit_seconds = fit_model(definition, overrides, seed, features, targets)
                train_f1 = candidate_statistics(targets, estimator.predict_proba(features)[:, 1])["macro_f1"]
        timing = benchmark(estimator, student_test)
        with threadpool_limits(limits=1):
            metrics, per_trip, predictions = evaluate_partition(
                student_test,
                (lambda data: score_candidates(estimator, data.batch)) if estimator else (lambda data: None),
                continuity,
            )
        elapsed_ms = float(np.median(timing)) if timing else None
        comparison.append({
            "model": name, "family": definition.family if definition else "Geometric baseline",
            **summary, **metrics, "full_train_macro_f1": train_f1, "refit_seconds": fit_seconds,
            "inference_ms_for_10_trips": elapsed_ms,
            "inference_ms_per_1000_observations": (
                elapsed_ms * 1_000 / metrics["observations"] if elapsed_ms is not None else None
            ),
            "inference_repeats_ms": timing,
        })
        per_trip_rows.extend({"model": name, **row} for row in per_trip)
        for data, predicted in zip(student_test, predictions):
            predicted_rows.extend(prediction_rows(name, data, predicted))
        if name == tuned_name:
            export_model(artifacts / "model.joblib", estimator, preprocessing)
        print(f"{name}: student-test Top-1 {metrics['top1_accuracy']:.5f}", flush=True)

    preprocessing_config = {
        "format_version": 1, "settings": asdict(preprocessing), "feature_names": FEATURE_NAMES,
        "reference_alignment_max_distance_m": REFERENCE_MAX_DISTANCE_M,
        "reference_alignment_used_only_for": ["training labels", "evaluation labels"],
        "missing_value_strategy": "Median imputation learned on the fit partition",
        "feature_scaling": "StandardScaler for Logistic Regression and Gaussian Naive Bayes",
        "candidate_probability": "P(match) for each candidate from the fitted binary classifier",
        "log_emission": "Log of clipped match odds normalized across the candidates for each observation",
    }
    write_json(artifacts / "preprocessing.json", preprocessing_config)
    write_csv(output / "model_comparison.csv", [
        {key: value for key, value in row.items() if key != "inference_repeats_ms"} for row in comparison
    ])
    write_csv(output / "student_test_by_trip.csv", per_trip_rows)
    write_csv(output / "student_test_predictions.csv.gz", predicted_rows)
    write_json(output / "run.json", {
        "seed": seed, "fit_cpu_threads": threads, "benchmark_cpu_threads": 1,
        "python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
        "dependencies": {name: version(name) for name in
                         ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "partitions": {
            name: [{"trip_id": item.trip_id, "observations": len(item.reference),
                    "labeled_candidate_rows": len(item.candidate_targets), "source_sha256": item.source_sha256,
                    "preprocessing_ms": item.preprocessing_ms} for item in partition]
            for name, partition in [("train", training_trips), ("student_test", student_test)]
        },
        "baseline_configurations": {model.name: model.parameters for model in MODELS},
        "selection_metric": "mean candidate Macro F1 across five training folds",
        "selected_model": model.name, "exported_model": tuned_name, "best_trial": best["trial"],
        "tuned_parameters": best["parameters"], "continuity_metric": continuity_spec,
        "continuity_status": "provisional" if continuity_spec == DEFAULT_CONTINUITY else "external",
        "comparison": comparison, "elapsed_seconds": perf_counter() - started,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("results/classifier_selection"))
    parser.add_argument("--tuning-output", type=Path, default=Path("results/hyperparameter_tuning"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/classifier"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--continuity", default=DEFAULT_CONTINUITY, metavar="MODULE:FUNCTION")
    args = parser.parse_args()
    if args.trials < 5 or args.threads < 1:
        parser.error("Use at least five trials and one CPU thread")
    run(args.data_root, args.output, args.tuning_output, args.artifacts,
        args.seed, args.trials, args.threads, args.continuity)


if __name__ == "__main__":
    main()
