"""Shared CSV schemas and baseline scoring for single trips and data splits."""

import csv
from pathlib import Path

import numpy as np

from .candidates import Candidates, UNMATCHED
from .reference import ReferenceAlignment
from .trip import Trip

PREDICTION_FIELDS = ("trip_id", "observation_index", "timestamp_s", "predicted_arc_id", "distance_m")
CANDIDATE_FIELDS = (
    "trip_id", "observation_index", "timestamp_s", "arc_id", "distance_m", "arc_fraction", "heading_cosine",
)
LABEL_FIELDS = ("trip_id", "observation_index", "reference_arc_id", "reference_distance_m")


def prediction_rows(trip: Trip, candidates: Candidates):
    for index, (timestamp, arc_id, distance) in enumerate(zip(
        trip.timestamp_s, candidates.arc_ids[:, 0], candidates.distance_m[:, 0],
    )):
        matched = arc_id != UNMATCHED
        yield trip.trip_id, index, timestamp, arc_id if matched else "", distance if matched else ""


def candidate_rows(trip: Trip, candidates: Candidates):
    for index, rank in zip(*np.where(candidates.arc_ids != UNMATCHED)):
        heading = candidates.heading_cosine[index, rank]
        yield (
            trip.trip_id, index, trip.timestamp_s[index], candidates.arc_ids[index, rank],
            candidates.distance_m[index, rank], candidates.arc_fraction[index, rank],
            heading if np.isfinite(heading) else "",
        )


def label_rows(trip: Trip, alignment: ReferenceAlignment):
    for index, (arc_id, distance) in enumerate(zip(alignment.arc_ids, alignment.distance_m)):
        yield trip.trip_id, index, arc_id if arc_id != UNMATCHED else "", distance


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(rows)


def baseline_counts(candidates: Candidates, alignment: ReferenceAlignment) -> dict[str, int]:
    labeled = alignment.arc_ids != UNMATCHED
    hits = candidates.arc_ids[labeled] == alignment.arc_ids[labeled, None]
    return {
        "observations": len(labeled),
        "matched_observations": int((candidates.arc_ids[:, 0] != UNMATCHED).sum()),
        "labeled_observations": int(labeled.sum()),
        "candidate_misses": int((~hits.any(axis=1)).sum()),
        "correct_top1": int(hits[:, 0].sum()),
    }


def baseline_rates(counts: dict[str, int]) -> dict[str, float | None]:
    labeled = counts["labeled_observations"]
    return {
        "reference_coverage": labeled / counts["observations"],
        "candidate_miss_rate": counts["candidate_misses"] / labeled if labeled else None,
        "top1_accuracy": counts["correct_top1"] / labeled if labeled else None,
    }
