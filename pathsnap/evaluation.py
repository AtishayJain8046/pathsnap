"""Measure candidate classification, observation ranking, and scoring time."""

from time import perf_counter

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from .candidates import UNMATCHED
from .model import candidate_outputs, score_candidates

TIMING_REPEATS = 7


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def candidate_statistics(targets, positive):
    predicted = positive > 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predicted, labels=[False, True], average="macro", zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(targets, predicted, labels=[False, True]).ravel()
    probability_correct = np.where(targets, positive, 1 - positive)
    negative_log_sum = -np.log(np.clip(probability_correct.astype(float), 1e-15, 1)).sum()
    return {
        "candidate_rows": len(targets), "candidate_accuracy": ratio(tn + tp, len(targets)),
        "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
        "log_loss": ratio(negative_log_sum, len(targets)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def observation_counts(data, predictions, log_emission=None):
    labeled = data.reference != UNMATCHED
    hits = (data.batch.arc_ids == data.reference[:, None]) & labeled[:, None]
    hit_rows, hit_cols = np.where(hits)
    correct = labeled & (predictions == data.reference)
    return {
        "observations": len(labeled),
        "labeled_observations": int(labeled.sum()),
        "candidate_hits": len(hit_rows),
        "top1_correct": int(correct.sum()),
        "log_emission_sum": float(-log_emission[hit_rows, hit_cols].sum()) if log_emission is not None else None,
    }


def observation_rates(counts):
    return {
        "reference_coverage": ratio(counts["labeled_observations"], counts["observations"]),
        "candidate_miss_rate": ratio(
            counts["labeled_observations"] - counts["candidate_hits"], counts["labeled_observations"],
        ),
        "top1_accuracy": ratio(counts["top1_correct"], counts["labeled_observations"]),
        "observation_log_loss_candidate_hits": (
            ratio(counts["log_emission_sum"], counts["candidate_hits"])
            if counts["log_emission_sum"] is not None else None
        ),
    }


def evaluate_partition(trips, score, continuity):
    per_trip = []
    predictions = []
    target_arrays = []
    score_arrays = []
    for data in trips:
        probabilities = score(data)
        if probabilities is None:
            predicted = data.batch.arc_ids[:, 0]
            log_emission = None
            counts = observation_counts(data, predicted)
        else:
            predicted, _, log_emission = candidate_outputs(data.batch, probabilities)
            counts = observation_counts(data, predicted, log_emission)
            target_arrays.append(data.candidate_targets)
            score_arrays.append(probabilities[data.labeled_candidate_rows])
        continuous, eligible = continuity(data.trip, predicted)
        per_trip.append({
            "trip_id": data.trip_id, **counts, **observation_rates(counts),
            "continuity_pairs": continuous, "eligible_pairs": eligible,
            "route_continuity": ratio(continuous, eligible),
        })
        predictions.append(predicted)

    def aggregate(key):
        return sum(trip[key] for trip in per_trip)

    keys = ("observations", "labeled_observations", "candidate_hits", "top1_correct",
            "continuity_pairs", "eligible_pairs")
    total = {key: aggregate(key) for key in keys}
    total["log_emission_sum"] = aggregate("log_emission_sum") if score_arrays else None
    total.update(observation_rates(total))
    total["route_continuity"] = ratio(total["continuity_pairs"], total["eligible_pairs"])
    if score_arrays:
        total.update(candidate_statistics(np.concatenate(target_arrays), np.concatenate(score_arrays)))
    return total, per_trip, predictions


def cross_validate(model, parameters, seed, features, targets, splits):
    rows = []
    for number, (train, validation) in enumerate(splits, 1):
        estimator = model.create(seed, parameters)
        start = perf_counter()
        estimator.fit(features[train], targets[train])
        fit_seconds = perf_counter() - start
        probabilities = estimator.predict_proba(features[validation])[:, 1]
        result = candidate_statistics(targets[validation], probabilities)
        rows.append({"fold": number, "fit_seconds": fit_seconds, **result})
    return rows


def summarize_cv(rows):
    scores = [row["macro_f1"] for row in rows]
    return {
        "cv_macro_f1_mean": float(np.mean(scores)),
        "cv_macro_f1_sd": float(np.std(scores, ddof=1)),
        "cv_log_loss_mean": float(np.mean([row["log_loss"] for row in rows])),
        "cv_fit_seconds": sum(row["fit_seconds"] for row in rows),
    }


def benchmark(estimator, trips):
    if estimator is None:
        return None
    classifier = estimator.steps[-1][1]
    if hasattr(classifier, "n_jobs"):
        classifier.set_params(n_jobs=1)
    measurements = []
    with threadpool_limits(limits=1):
        for _ in range(TIMING_REPEATS + 1):
            start = perf_counter()
            for data in trips:
                candidate_outputs(data.batch, score_candidates(estimator, data.batch))
            measurements.append((perf_counter() - start) * 1_000)
    return measurements[1:]
