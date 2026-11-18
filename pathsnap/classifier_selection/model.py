"""Export and load fitted pipelines for candidate scoring."""

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from pathsnap.feature_selection.features import FEATURE_NAMES, CandidateBatch, Preprocessing, build_features
from pathsnap.data_preprocessing.trip import Trip


def score_candidates(estimator, batch: CandidateBatch) -> NDArray:
    """Return binary match probabilities in candidate-row order."""
    if not len(batch.features):
        return np.empty(0)
    return estimator.predict_proba(batch.features)[:, list(estimator.classes_).index(1)]


def candidate_outputs(batch: CandidateBatch, probabilities: NDArray):
    """Return (top arcs, probability grid, normalized log-emission grid)."""
    scores = np.full(batch.arc_ids.shape, -np.inf)
    scores[batch.rows, batch.columns] = probabilities
    predictions = batch.arc_ids[np.arange(len(batch.arc_ids)), scores.argmax(axis=1)]
    probabilities_grid = np.where(np.isfinite(scores), scores, np.nan)

    # Conditional on exactly one correct candidate, normalize binary match odds.
    clipped = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    odds = np.full(batch.arc_ids.shape, -np.inf)
    odds[batch.rows, batch.columns] = np.log(clipped) - np.log1p(-clipped)
    matched = np.isfinite(scores).any(axis=1)
    odds[matched] -= logsumexp(odds[matched], axis=1, keepdims=True)
    return predictions, probabilities_grid, odds


class CandidateScorer:
    """Read the exported pipeline and score a parsed trip for the decoder/API."""

    def __init__(self, artifact: Path):
        bundle = joblib.load(artifact)
        if bundle["format_version"] != 1 or tuple(bundle["feature_names"]) != FEATURE_NAMES:
            raise ValueError("The model artifact and feature implementation do not match")
        self.estimator = bundle["estimator"]
        self.preprocessing = Preprocessing(**bundle["preprocessing"])

    def predict(self, trip: Trip):
        """Return (candidate arc IDs, top-1 arc IDs, match probabilities, log emissions)."""
        batch = build_features(trip, self.preprocessing)
        with threadpool_limits(limits=1):
            probabilities = score_candidates(self.estimator, batch)
        return (batch.arc_ids, *candidate_outputs(batch, probabilities))


def export_model(destination, estimator, preprocessing):
    joblib.dump({
        "format_version": 1,
        "estimator": estimator,
        "preprocessing": asdict(preprocessing),
        "feature_names": FEATURE_NAMES,
    }, destination, compress=3)
