"""Provisional directed adjacency while the team agrees on route continuity."""

import numpy as np

from pathsnap.data_preprocessing.candidates import UNMATCHED


def directed_adjacency(trip, predicted_arc_ids):
    """Return (continuous pairs, eligible pairs) for consecutive GPS readings."""
    eligible = np.diff(trip.timestamp_s) <= 10
    previous, current = predicted_arc_ids[:-1], predicted_arc_ids[1:]
    matched = eligible & (previous != UNMATCHED) & (current != UNMATCHED)
    previous, current = previous[matched], current[matched]
    continuous = (previous == current) | (trip.arcs[previous, 1] == trip.arcs[current, 0])
    return int(continuous.sum()), int(eligible.sum())
