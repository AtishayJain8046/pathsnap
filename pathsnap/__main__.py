"""Run the nearest-road baseline and export reusable candidate features."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .candidates import Candidates, UNMATCHED, generate_candidates
from .reference import align_reference
from .trip import Trip, load_reference_route, load_trip


def _write_predictions(output: Path, trip: Trip, candidates: Candidates) -> None:
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("trip_id", "observation_index", "timestamp_s", "predicted_arc_id", "distance_m"))
        predictions = zip(trip.timestamp_s, candidates.arc_ids[:, 0], candidates.distance_m[:, 0])
        for index, (timestamp, arc_id, distance) in enumerate(predictions):
            matched = arc_id != UNMATCHED
            writer.writerow((
                trip.trip_id, index, timestamp, arc_id if matched else "", distance if matched else "",
            ))


def _write_candidates(output: Path, trip: Trip, candidates: Candidates) -> None:
    with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "trip_id", "observation_index", "timestamp_s", "arc_id",
            "distance_m", "arc_fraction", "heading_cosine",
        ))
        for index, rank in zip(*np.where(candidates.arc_ids != UNMATCHED)):
            heading = candidates.heading_cosine[index, rank]
            writer.writerow((
                trip.trip_id, index, trip.timestamp_s[index], candidates.arc_ids[index, rank],
                candidates.distance_m[index, rank], candidates.arc_fraction[index, rank],
                heading if np.isfinite(heading) else "",
            ))


def _evaluate(
    output: Path, trip: Trip, candidates: Candidates, route: NDArray, max_distance_m: float,
) -> dict[str, int | float | None]:
    alignment = align_reference(trip, route, max_distance_m=max_distance_m)
    with (output / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("trip_id", "observation_index", "reference_arc_id", "reference_distance_m"))
        for index, (arc_id, distance) in enumerate(zip(alignment.arc_ids, alignment.distance_m)):
            writer.writerow((trip.trip_id, index, arc_id if arc_id != UNMATCHED else "", distance))

    labeled = alignment.arc_ids != UNMATCHED
    labeled_count = int(labeled.sum())
    candidate_hit = (candidates.arc_ids[labeled] == alignment.arc_ids[labeled, None]).any(axis=1)
    predicted = candidates.arc_ids[labeled, 0]
    return {
        "labeled_observations": labeled_count,
        "reference_coverage": labeled_count / len(labeled),
        "reference_max_distance_m": max_distance_m,
        "candidate_miss_rate": float((~candidate_hit).mean()) if labeled_count else None,
        "top1_accuracy": float((predicted == alignment.arc_ids[labeled]).mean()) if labeled_count else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Directory for local results")
    parser.add_argument("--radius-m", type=float, default=50.0, help="Candidate search radius (default: 50)")
    parser.add_argument("--limit", type=int, default=8, help="Maximum candidates per reading (default: 8)")
    parser.add_argument("--evaluate", action="store_true", help="Align a supplied .route file and score predictions")
    parser.add_argument(
        "--reference-max-distance-m", type=float, default=50.0,
        help="Maximum GPS distance from an assigned reference arc when scoring (default: 50)",
    )
    parser.add_argument("--map", action="store_true", help="Also generate a Folium map")
    args = parser.parse_args()

    try:
        trip = load_trip(args.trip_directory)
        candidates = generate_candidates(trip, radius_m=args.radius_m, limit=args.limit)
        route = load_reference_route(args.trip_directory, len(trip.arcs)) if args.evaluate else None

        args.output.mkdir(parents=True, exist_ok=True)
        _write_predictions(args.output, trip, candidates)
        _write_candidates(args.output, trip, candidates)
        metrics = {
            "trip_id": trip.trip_id,
            "observations": len(trip.gps_xy),
            "candidate_radius_m": args.radius_m,
            "candidate_limit": args.limit,
            "matched_observations": int((candidates.arc_ids[:, 0] != UNMATCHED).sum()),
        }
        if route is not None:
            metrics.update(_evaluate(args.output, trip, candidates, route, args.reference_max_distance_m))
        if args.map:
            from .visualize import save_map

            save_map(args.output / "map.html", trip, candidates, route)

        result = json.dumps(metrics, indent=2)
        (args.output / "metrics.json").write_text(result + "\n", encoding="utf-8")
        print(result)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"PathSnap: {exc}\n")


if __name__ == "__main__":
    main()
