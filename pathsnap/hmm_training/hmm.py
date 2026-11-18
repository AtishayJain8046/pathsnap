"""Decode the saved classifier's road candidates using a directed-graph HMM."""

from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush

import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from pathsnap.data_preprocessing.candidates import UNMATCHED
from pathsnap.feature_selection.features import FEATURE_NAMES, build_features
from pathsnap.classifier_selection.model import CandidateScorer, candidate_outputs, score_candidates
from pathsnap.data_preprocessing.trip import Trip


@dataclass(frozen=True)
class TransitionConfig:
    max_gap_s: float = 10
    max_route_distance_m: float = 600
    max_speed_mps: float = 55
    position_slack_m: float = 40
    backward_jitter_m: float = 15


@dataclass(frozen=True)
class DecodeConfig:
    beta_m: float = 20
    emission_weight: float = 1.0


class DirectedRoadGraph:
    def __init__(self, trip: Trip, config: TransitionConfig):
        self.arcs = trip.arcs
        self.config = config
        self.lengths = np.linalg.norm(
            trip.nodes_xy[trip.arcs[:, 1]] - trip.nodes_xy[trip.arcs[:, 0]], axis=1,
        )
        self.outgoing_arcs = np.argsort(trip.arcs[:, 0], kind="stable")
        counts = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes_xy))
        self.offsets = np.r_[0, np.cumsum(counts)]
        self.reachable = lru_cache(maxsize=512)(self._reachable)

    def _reachable(self, source):
        distances = {source: 0.0}
        predecessors = {}
        queue = [(0.0, source)]
        while queue:
            distance, node = heappop(queue)
            if distance > distances[node]:
                continue
            for arc in self.outgoing_arcs[self.offsets[node]:self.offsets[node + 1]]:
                target = int(self.arcs[arc, 1])
                candidate = distance + self.lengths[arc]
                if candidate <= self.config.max_route_distance_m and candidate < distances.get(target, np.inf):
                    distances[target] = candidate
                    predecessors[target] = int(arc)
                    heappush(queue, (candidate, target))
        return distances, predecessors

    def transition_distances(
        self, previous_arcs: NDArray, previous_positions: NDArray, next_arcs: NDArray,
        next_positions: NDArray, budget: float,
    ) -> NDArray:
        next_starts = self.arcs[next_arcs, 0]
        distances = np.full((len(previous_arcs), len(next_arcs)), np.inf)
        for row, (arc, position) in enumerate(zip(previous_arcs, previous_positions)):
            suffix = self.lengths[arc] - position
            if suffix <= budget:
                reachable, _ = self.reachable(int(self.arcs[arc, 1]))
                links = np.asarray([reachable.get(int(node), np.inf) for node in next_starts])
                distances[row] = suffix + links + next_positions
            delta = next_positions - position
            same_arc = (arc == next_arcs) & (delta >= -self.config.backward_jitter_m)
            distances[row, same_arc] = np.abs(delta[same_arc])
        distances[distances > budget] = np.inf
        return distances

    def bridge_arcs(self, previous_arc, previous_position, next_arc, next_position):
        """Return the intermediate directed arcs and the arriving arc."""
        if previous_arc == next_arc and next_position - previous_position >= -self.config.backward_jitter_m:
            return []
        start = int(self.arcs[previous_arc, 1])
        target = int(self.arcs[next_arc, 0])
        _, predecessors = self.reachable(start)
        backwards = []
        while target != start:
            arc = predecessors[target]
            backwards.append(arc)
            target = int(self.arcs[arc, 0])
        return [*reversed(backwards), int(next_arc)]


@dataclass(frozen=True)
class Lattice:
    trip: Trip
    graph: DirectedRoadGraph
    candidate_arcs: NDArray
    positions_m: NDArray
    classifier_arcs: NDArray
    log_emissions: NDArray
    transition_distances_m: NDArray
    gps_step_m: NDArray
    gap: NDArray


@dataclass(frozen=True)
class Segment:
    first_observation: int
    last_observation: int
    reason: str


@dataclass(frozen=True)
class DecodedTrip:
    arc_ids: NDArray
    candidate_columns: NDArray
    segments: tuple[Segment, ...]


def build_lattice(
    trip: Trip, scorer: CandidateScorer, config: TransitionConfig = TransitionConfig(),
) -> Lattice:
    if not all(np.isfinite(value) and value > 0 for value in vars(config).values()):
        raise ValueError("All transition settings must be finite and positive")
    batch = build_features(trip, scorer.preprocessing)
    with threadpool_limits(limits=1):
        probabilities = score_candidates(scorer.estimator, batch)
    classifier, _, log_emissions = candidate_outputs(batch, probabilities)
    graph = DirectedRoadGraph(trip, config)
    arcs = batch.arc_ids
    positions = np.full(arcs.shape, np.nan)
    candidate_ids = arcs[batch.rows, batch.columns]
    fractions = batch.features[:, FEATURE_NAMES.index("arc_fraction")]
    positions[batch.rows, batch.columns] = fractions * graph.lengths[candidate_ids]
    elapsed = np.diff(trip.timestamp_s)
    gap = elapsed > config.max_gap_s
    gps_step = np.linalg.norm(np.diff(trip.gps_xy, axis=0), axis=1)
    budget = np.minimum(config.max_route_distance_m, config.max_speed_mps * elapsed + config.position_slack_m)
    valid_counts = (arcs != UNMATCHED).sum(axis=1)
    transitions = np.full((len(elapsed), arcs.shape[1], arcs.shape[1]), np.inf, dtype=np.float32)
    for step in np.flatnonzero(~gap & (valid_counts[:-1] > 0) & (valid_counts[1:] > 0)):
        before, after = slice(valid_counts[step]), slice(valid_counts[step + 1])
        transitions[step, before, after] = graph.transition_distances(
            arcs[step, before], positions[step, before], arcs[step + 1, after], positions[step + 1, after], budget[step],
        )
    return Lattice(trip, graph, arcs, positions, classifier, log_emissions, transitions, gps_step, gap)


def viterbi(lattice: Lattice, config: DecodeConfig = DecodeConfig()) -> DecodedTrip:
    if not all(np.isfinite(value) and value > 0 for value in vars(config).values()):
        raise ValueError("Emission weight and transition scale must be finite and positive")
    emissions = lattice.log_emissions * config.emission_weight
    scores = None
    parents = np.full(emissions.shape, -1, dtype=np.int16)
    chosen_columns = np.full(len(emissions), UNMATCHED, dtype=int)
    endpoints = []
    start, reason = 0, "trip_start"
    columns = np.arange(emissions.shape[1])

    for index, observation in enumerate(emissions):
        available = np.isfinite(observation).any()
        if not available:
            if scores is not None:
                endpoints.append((Segment(start, index - 1, reason), int(scores.argmax())))
            scores = None
            reason = "missing_candidates"
            continue
        if scores is None:
            start, scores = index, observation
            continue
        if lattice.gap[index - 1]:
            break_reason = "gps_gap"
        else:
            mismatch = np.abs(lattice.transition_distances_m[index - 1] - lattice.gps_step_m[index - 1])
            options = scores[:, None] - mismatch / config.beta_m
            previous = options.argmax(axis=0)
            next_scores = observation + options[previous, columns]
            if np.isfinite(next_scores).any():
                parents[index] = previous
                scores = next_scores
                continue
            break_reason = "no_directed_transition"
        endpoints.append((Segment(start, index - 1, reason), int(scores.argmax())))
        start, scores, reason = index, observation, break_reason
    if scores is not None:
        endpoints.append((Segment(start, len(emissions) - 1, reason), int(scores.argmax())))

    predicted = np.full(len(emissions), UNMATCHED, dtype=int)
    for segment, column in endpoints:
        for index in range(segment.last_observation, segment.first_observation - 1, -1):
            chosen_columns[index] = column
            predicted[index] = lattice.candidate_arcs[index, column]
            column = parents[index, column]
    return DecodedTrip(predicted, chosen_columns, tuple(segment for segment, _ in endpoints))


def route_arc_sequences(lattice: Lattice, decoded: DecodedTrip) -> list[list[int]]:
    """Expand each continuous segment with its unobserved connecting road arcs."""
    routes = []
    for segment in decoded.segments:
        first, last = segment.first_observation, segment.last_observation
        arcs = [int(decoded.arc_ids[first])]
        for index in range(first + 1, last + 1):
            previous, arriving = decoded.candidate_columns[index - 1:index + 1]
            arcs.extend(lattice.graph.bridge_arcs(
                decoded.arc_ids[index - 1], lattice.positions_m[index - 1, previous],
                decoded.arc_ids[index], lattice.positions_m[index, arriving],
            ))
        routes.append(arcs)
    return routes


def reachable_pairs(lattice: Lattice, predicted_arcs: NDArray) -> tuple[int, int]:
    """Count plausible directed transitions over consecutive readings within the gap limit."""
    matches = lattice.candidate_arcs == predicted_arcs[:, None]
    candidate_columns = np.argmax(matches, axis=1)
    has_candidate = matches.any(axis=1) & (predicted_arcs != UNMATCHED)
    eligible = ~lattice.gap
    assigned = eligible & has_candidate[:-1] & has_candidate[1:]
    steps = np.flatnonzero(assigned)
    continuous = np.isfinite(lattice.transition_distances_m[
        steps, candidate_columns[steps], candidate_columns[steps + 1],
    ])
    return int(continuous.sum()), int(eligible.sum())
