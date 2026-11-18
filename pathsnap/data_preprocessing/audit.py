"""Summarize trip quality and flag places where reference labels are uncertain."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.data_preprocessing.candidates import Candidates, UNMATCHED
from pathsnap.data_preprocessing.reference import ReferenceAlignment
from pathsnap.data_preprocessing.trip import Trip


def profile_trip(trip: Trip, route, candidates: Candidates, alignment: ReferenceAlignment) -> dict:
    starts, ends = trip.arcs.T
    graph = csr_matrix(
        (np.ones(len(starts), dtype=bool), (starts, ends)),
        shape=(len(trip.nodes_xy), len(trip.nodes_xy)),
    )
    components, component_ids = connected_components(graph, directed=False)
    route_starts, route_ends = trip.arcs[route].T
    gaps = np.diff(trip.timestamp_s, prepend=trip.timestamp_s[0])
    lengths = np.linalg.norm(trip.nodes_xy[ends] - trip.nodes_xy[starts], axis=1)
    return {
        "nodes": len(trip.nodes_xy),
        "arcs": len(trip.arcs),
        "reference_entries": len(route),
        "weak_components": components,
        "isolated_nodes": len(trip.nodes_xy) - len(np.unique(trip.arcs)),
        "largest_component_nodes": int(np.bincount(component_ids).max()),
        "zero_length_arcs": int((lengths == 0).sum()),
        "longest_arc_m": float(lengths.max()),
        "one_second_steps": int(np.isclose(gaps[1:], 1.0).sum()),
        "max_gap_s": float(gaps.max()),
        "gps_width_m": float(np.ptp(trip.gps_xy[:, 0])),
        "gps_height_m": float(np.ptp(trip.gps_xy[:, 1])),
        "track_distance_m": float(np.linalg.norm(np.diff(trip.gps_xy, axis=0), axis=1).sum()),
        "reference_breaks": int(((route[1:] != route[:-1]) & (route_ends[:-1] != route_starts[1:])).sum()),
        "heading_missing_rows": int(((candidates.arc_ids != UNMATCHED) & np.isnan(candidates.heading_cosine)).sum()),
        **_alignment_flags(trip, route, candidates, alignment, gaps),
    }


def _alignment_flags(trip, route, candidates, alignment, gaps) -> dict[str, int]:
    # A candidate within 3 m of the assigned arc's distance is a plausible
    # geometric alternative. Separate junctions from nearby parallel roads.
    labeled = alignment.arc_ids != UNMATCHED
    valid_alternative = (
        labeled[:, None] & (candidates.arc_ids != UNMATCHED)
        & (candidates.arc_ids != alignment.arc_ids[:, None])
        & (np.abs(candidates.distance_m - alignment.distance_m[:, None]) <= 3)
    )
    rows, ranks = np.where(valid_alternative)
    ref_arcs = trip.arcs[alignment.arc_ids[rows]]
    alt_arcs = trip.arcs[candidates.arc_ids[rows, ranks]]
    shared_node = (ref_arcs[:, :, None] == alt_arcs[:, None, :]).any(axis=(1, 2))
    ref_vectors = trip.nodes_xy[ref_arcs[:, 1]] - trip.nodes_xy[ref_arcs[:, 0]]
    alt_vectors = trip.nodes_xy[alt_arcs[:, 1]] - trip.nodes_xy[alt_arcs[:, 0]]
    norms = np.linalg.norm(ref_vectors, axis=1) * np.linalg.norm(alt_vectors, axis=1)
    parallel = (norms > 0) & (np.abs((ref_vectors * alt_vectors).sum(axis=1)) >= 0.95 * norms)
    # A reverse edge shares both endpoints. Count it separately from junctions.
    reverse = (ref_arcs == alt_arcs[:, ::-1]).all(axis=1)
    repeated_ids = np.flatnonzero(np.bincount(route, minlength=len(trip.arcs)) > 1)
    after_gap = gaps > 10
    return {
        "junction_near_ties": len(np.unique(rows[shared_node & ~parallel])),
        "parallel_near_ties": len(np.unique(rows[parallel & ~reverse])),
        "reverse_direction_near_ties": len(np.unique(rows[reverse])),
        "repeated_reference_arcs": len(repeated_ids),
        "labels_on_repeated_reference_arcs": int(np.isin(alignment.arc_ids, repeated_ids).sum()),
        "gaps_over_10s": int(after_gap.sum()),
        "labeled_after_gap": int((after_gap & labeled).sum()),
    }


def save_profile_plots(path, summaries):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    panels = (
        (axes[0, 0], "observations", "GPS readings per trip"),
        (axes[0, 1], "arcs", "Directed arcs per trip"),
        (axes[1, 0], "max_gap_s", "Longest GPS gap per trip (seconds)"),
        (axes[1, 1], "reference_coverage", "Fraction of observations with a reference label"),
    )
    for axis, key, title in panels:
        values = [row[key] for rows in summaries.values() for row in rows]
        if key == "max_gap_s" and min(values) > 0 and max(values) > min(values):
            bins = np.geomspace(min(values), max(values), 13)
            axis.set_xscale("log")
        else:
            bins = np.histogram_bin_edges(values, bins=12)
        for split, rows in summaries.items():
            axis.hist([row[key] for row in rows], bins=bins, alpha=0.6, label=split)
        axis.set_title(title)
        axis.set_ylabel("Trips")
        axis.legend()
    figure.savefig(path, dpi=150)
