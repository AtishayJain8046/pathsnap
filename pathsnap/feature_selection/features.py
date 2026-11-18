"""Build road candidate features from GPS and road geometry."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pathsnap.data_preprocessing.candidates import UNMATCHED, generate_candidates
from pathsnap.data_preprocessing.trip import Trip


@dataclass(frozen=True)
class Preprocessing:
    radius_m: float = 50.0
    candidate_limit: int = 8
    heading_max_gap_s: float = 10.0
    heading_min_motion_m: float = 3.0
    heading_window_observations: int = 4


FEATURE_NAMES = (
    "distance_m", "arc_fraction", "previous_heading_cosine",
    "candidate_rank", "distance_above_nearest_m", "candidate_count",
    "arc_length_m", "distance_to_start_m", "distance_to_end_m",
    "next_heading_cosine", "window_heading_cosine", "previous_speed_mps",
    "next_speed_mps", "previous_heading_available", "next_heading_available",
    "window_heading_available", "previous_excess_to_arc_m", "next_excess_to_arc_m",
    "incoming_degree", "outgoing_degree",
)


@dataclass(frozen=True)
class CandidateBatch:
    arc_ids: NDArray
    rows: NDArray
    columns: NDArray
    features: NDArray


def _motion(trip, before, after, preprocessing):
    change = trip.gps_xy[after] - trip.gps_xy[before]
    distance = np.linalg.norm(change, axis=1)
    elapsed = trip.timestamp_s[after] - trip.timestamp_s[before]
    nearby = (elapsed > 0) & (elapsed <= preprocessing.heading_max_gap_s)
    usable = nearby & (distance >= preprocessing.heading_min_motion_m)
    heading = np.divide(
        change, distance[:, None], out=np.full_like(change, np.nan), where=usable[:, None],
    )
    speed = np.divide(distance, elapsed, out=np.full_like(distance, np.nan), where=nearby)
    return heading, speed


def build_features(trip: Trip, preprocessing: Preprocessing) -> CandidateBatch:
    candidates = generate_candidates(
        trip, radius_m=preprocessing.radius_m, limit=preprocessing.candidate_limit,
    )
    rows, columns = np.where(candidates.arc_ids != UNMATCHED)
    arc_ids = candidates.arc_ids[rows, columns]
    from_nodes, to_nodes = trip.arcs[arc_ids].T
    starts = trip.nodes_xy[from_nodes]
    vectors = trip.nodes_xy[to_nodes] - starts
    lengths = np.linalg.norm(vectors, axis=1)
    unit = np.divide(vectors, lengths[:, None], out=np.zeros_like(vectors), where=lengths[:, None] > 0)
    fraction = candidates.arc_fraction[rows, columns]
    distance = candidates.distance_m[rows, columns]
    nearest = candidates.distance_m[:, 0]
    counts = np.bincount(rows, minlength=len(nearest))

    obs = np.arange(len(nearest))
    previous = np.maximum(obs - 1, 0)
    following = np.minimum(obs + 1, len(obs) - 1)
    window = preprocessing.heading_window_observations
    before = np.maximum(obs - window, 0)
    after = np.minimum(obs + window, len(obs) - 1)
    previous_heading, previous_speed = _motion(trip, previous, obs, preprocessing)
    next_heading, next_speed = _motion(trip, obs, following, preprocessing)
    window_heading, _ = _motion(trip, before, after, preprocessing)

    def heading_agreement(heading):
        return np.einsum("ij,ij->i", unit, heading[rows]).clip(-1, 1)

    def neighbor_excess(indices):
        offsets = trip.gps_xy[indices[rows]] - starts
        along = np.einsum("ij,ij->i", offsets, unit).clip(0, lengths)
        neighbor_distance = np.linalg.norm(offsets - along[:, None] * unit, axis=1)
        excess = np.maximum(neighbor_distance - nearest[indices[rows]], 0)
        elapsed = np.abs(trip.timestamp_s[indices[rows]] - trip.timestamp_s[rows])
        return np.where((elapsed > 0) & (elapsed <= preprocessing.heading_max_gap_s), excess, np.nan)

    indegree = np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes_xy))
    outdegree = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes_xy))
    features = np.column_stack((
        distance, fraction, heading_agreement(previous_heading),
        columns + 1, distance - nearest[rows], counts[rows],
        lengths, fraction * lengths, (1 - fraction) * lengths,
        heading_agreement(next_heading), heading_agreement(window_heading),
        previous_speed[rows], next_speed[rows],
        np.isfinite(previous_heading[rows, 0]), np.isfinite(next_heading[rows, 0]),
        np.isfinite(window_heading[rows, 0]), neighbor_excess(previous),
        neighbor_excess(following), indegree[from_nodes], outdegree[to_nodes],
    )).astype(np.float32)
    return CandidateBatch(candidates.arc_ids, rows, columns, features)
