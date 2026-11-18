"""Assemble model feature rows and candidate labels for each trip."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .candidates import UNMATCHED
from .experiment_data import aligned_labels, partition_paths, source_fingerprint
from .features import CandidateBatch, build_features
from .trip import Trip, load_trip


@dataclass(frozen=True)
class LabeledTrip:
    trip: Trip
    batch: CandidateBatch
    reference: NDArray
    labeled_candidate_rows: NDArray
    candidate_targets: NDArray
    preprocessing_ms: float
    source_sha256: str

    @property
    def trip_id(self):
        return self.trip.trip_id


def load_partition(directory: Path, expected_count, preprocessing, reference_max_distance_m):
    trips = []
    for path in partition_paths(directory, expected_count):
        started = perf_counter()
        trip = load_trip(path)
        batch = build_features(trip, preprocessing)
        elapsed = (perf_counter() - started) * 1_000
        reference = aligned_labels(path, trip, reference_max_distance_m)
        labeled_candidates = reference[batch.rows] != UNMATCHED
        expected = reference[batch.rows[labeled_candidates]]
        observed = batch.arc_ids[batch.rows[labeled_candidates], batch.columns[labeled_candidates]]
        trips.append(LabeledTrip(trip, batch, reference, labeled_candidates, expected == observed,
                                 elapsed, source_fingerprint(path)))
    return trips


def training_rows(trips):
    features = np.concatenate([data.batch.features[data.labeled_candidate_rows] for data in trips])
    targets = np.concatenate([data.candidate_targets for data in trips])
    groups = np.concatenate([np.repeat(data.trip_id, len(data.candidate_targets)) for data in trips])
    return features, targets, groups
