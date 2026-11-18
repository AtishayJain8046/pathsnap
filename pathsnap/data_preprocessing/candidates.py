"""Generate nearby directed road candidates using GPS and map geometry."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from pathsnap.data_preprocessing.trip import Trip

UNMATCHED = -1


@dataclass(frozen=True)
class Candidates:
    # Each row corresponds to one GPS observation, in track order.
    arc_ids: NDArray
    distance_m: NDArray
    arc_fraction: NDArray
    heading_cosine: NDArray


def project_to_arcs(
    points: NDArray, starts: NDArray, vectors: NDArray, length_squared: NDArray,
) -> tuple[NDArray, NDArray]:
    offset = points - starts
    numerator = np.einsum("ij,ij->i", offset, vectors)
    fraction = np.divide(
        numerator, length_squared, out=np.zeros(len(starts)), where=length_squared > 0,
    )
    fraction = np.clip(fraction, 0, 1)
    distance = np.linalg.norm(offset - fraction[:, None] * vectors, axis=1)
    return distance, fraction


def _index_arcs(
    starts: NDArray, vectors: NDArray, lengths: NDArray,
) -> tuple[cKDTree, NDArray, float]:
    # Index short pieces so a long arc can be found anywhere along its length.
    piece_length_m = 100.0
    counts = np.maximum(1, np.ceil(lengths / piece_length_m).astype(int))
    piece_arc_ids = np.repeat(np.arange(len(starts)), counts)
    first_piece = np.cumsum(counts) - counts
    piece_number = np.arange(len(piece_arc_ids)) - first_piece[piece_arc_ids]
    fraction = (piece_number + 0.5) / counts[piece_arc_ids]
    midpoints = starts[piece_arc_ids] + fraction[:, None] * vectors[piece_arc_ids]
    return cKDTree(midpoints), piece_arc_ids, piece_length_m / 2


def _observed_headings(trip: Trip) -> NDArray:
    delta = np.diff(trip.gps_xy, axis=0, prepend=trip.gps_xy[:1])
    distance = np.linalg.norm(delta, axis=1)
    elapsed = np.diff(trip.timestamp_s, prepend=trip.timestamp_s[0])
    usable = (distance >= 3) & (elapsed > 0) & (elapsed <= 10)
    return np.divide(
        delta, distance[:, None], out=np.full_like(delta, np.nan), where=usable[:, None],
    )


def generate_candidates(trip: Trip, *, radius_m: float = 50.0, limit: int = 8) -> Candidates:
    if not np.isfinite(radius_m) or radius_m <= 0 or limit < 1:
        raise ValueError("Candidate radius and limit must be positive")
    starts, ends = trip.nodes_xy[trip.arcs].transpose(1, 0, 2)
    vectors = ends - starts
    length_squared = np.einsum("ij,ij->i", vectors, vectors)
    lengths = np.sqrt(length_squared)
    tree, piece_arc_ids, half_piece_m = _index_arcs(starts, vectors, lengths)
    headings = _observed_headings(trip)

    shape = (len(trip.gps_xy), limit)
    arc_ids = np.full(shape, UNMATCHED, dtype=int)
    distance_m = np.full(shape, np.nan)
    arc_fraction = np.full(shape, np.nan)
    heading_cosine = np.full(shape, np.nan)

    for index, point in enumerate(trip.gps_xy):
        pieces = tree.query_ball_point(point, radius_m + half_piece_m)
        nearby = np.unique(piece_arc_ids[pieces])
        distances, fractions = project_to_arcs(
            point, starts[nearby], vectors[nearby], length_squared[nearby],
        )
        within_radius = np.flatnonzero(distances <= radius_m)
        order = np.argsort(distances[within_radius], kind="stable")[:limit]
        chosen = within_radius[order]
        selected = nearby[chosen]
        columns = slice(len(selected))
        arc_ids[index, columns] = selected
        distance_m[index, columns] = distances[chosen]
        arc_fraction[index, columns] = fractions[chosen]
        heading_cosine[index, columns] = np.divide(
            vectors[selected] @ headings[index],
            lengths[selected],
            out=np.full(len(selected), np.nan),
            where=lengths[selected] > 0,
        ).clip(-1, 1)

    return Candidates(arc_ids, distance_m, arc_fraction, heading_cosine)
