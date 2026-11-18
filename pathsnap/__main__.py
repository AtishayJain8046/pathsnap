"""Run the nearest-road baseline and export reusable candidate features."""

import argparse
import json
from pathlib import Path

from .candidates import UNMATCHED, generate_candidates
from .exports import (
    CANDIDATE_FIELDS, LABEL_FIELDS, PREDICTION_FIELDS, baseline_counts, baseline_rates,
    candidate_rows, label_rows, prediction_rows, write_csv,
)
from .reference import align_reference
from .trip import load_reference_route, load_trip


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
        write_csv(args.output / "predictions.csv", PREDICTION_FIELDS, prediction_rows(trip, candidates))
        write_csv(args.output / "candidates.csv", CANDIDATE_FIELDS, candidate_rows(trip, candidates))
        metrics = {
            "trip_id": trip.trip_id,
            "observations": len(trip.gps_xy),
            "candidate_radius_m": args.radius_m,
            "candidate_limit": args.limit,
            "matched_observations": int((candidates.arc_ids[:, 0] != UNMATCHED).sum()),
        }
        if route is not None:
            alignment = align_reference(trip, route, max_distance_m=args.reference_max_distance_m)
            write_csv(args.output / "labels.csv", LABEL_FIELDS, label_rows(trip, alignment))
            counts = baseline_counts(candidates, alignment)
            metrics.update(baseline_rates(counts))
            metrics["labeled_observations"] = counts["labeled_observations"]
            metrics["reference_max_distance_m"] = args.reference_max_distance_m
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
