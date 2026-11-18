"""Derive observation labels for evaluation from the ordered reference arcs."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .candidates import UNMATCHED, project_to_arcs
from .trip import Trip


@dataclass(frozen=True)
class ReferenceAlignment:
    arc_ids: NDArray
    distance_m: NDArray


def align_reference(trip: Trip, route: NDArray, *, max_distance_m: float = 50) -> ReferenceAlignment:
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("Reference alignment distance must be positive")

    starts, ends = trip.nodes_xy[trip.arcs[route]].transpose(1, 0, 2)
    vectors = ends - starts
    length_squared = np.einsum("ij,ij->i", vectors, vectors)
    distances, _ = project_to_arcs(trip.gps_xy[0], starts, vectors, length_squared)
    cost = np.minimum(distances, max_distance_m)
    indices = np.arange(len(route))
    predecessors = np.empty((len(trip.gps_xy) - 1, len(route)), dtype=np.int32)

    for index, point in enumerate(trip.gps_xy[1:]):
        # Remaining on the same route entry or moving forward is allowed.
        prefix_cost = np.minimum.accumulate(cost)
        improves = np.r_[True, prefix_cost[1:] < prefix_cost[:-1]]
        predecessors[index] = np.maximum.accumulate(np.where(improves, indices, 0))
        distances, _ = project_to_arcs(point, starts, vectors, length_squared)
        cost = prefix_cost + np.minimum(distances, max_distance_m)

    positions = np.empty(len(trip.gps_xy), dtype=int)
    positions[-1] = np.argmin(cost)
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = predecessors[index, positions[index + 1]]

    distance_m, _ = project_to_arcs(
        trip.gps_xy, starts[positions], vectors[positions], length_squared[positions],
    )
    arc_ids = np.where(distance_m <= max_distance_m, route[positions], UNMATCHED)
    return ReferenceAlignment(arc_ids, distance_m)
