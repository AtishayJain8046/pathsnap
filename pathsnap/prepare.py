"""Prepare both student data splits for training and audit their quality."""

import argparse
import csv
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from .audit import profile_trip, save_profile_plots
from .candidates import generate_candidates
from .exports import (
    CANDIDATE_FIELDS, LABEL_FIELDS, PREDICTION_FIELDS, baseline_counts, baseline_rates,
    candidate_rows, label_rows, prediction_rows, write_csv,
)
from .reference import align_reference
from .trip import load_reference_route, load_trip
from .visualize import save_map

SPLITS = ("train", "student_test")
EXTENSIONS = ("track", "nodes", "arcs", "route")


def _trip_directories(source: Path) -> list[Path]:
    directories = sorted(entry for entry in source.iterdir() if entry.is_dir() and not entry.name.startswith("."))
    if not directories:
        raise ValueError(f"No trip directories in {source}")
    return directories


def _write_split(directories, destination, radius_m, limit, reference_max_distance_m, map_trip):
    destination.mkdir()
    summaries = []
    with ExitStack() as stack:
        writers = []
        for name, fields in (
            ("candidates", CANDIDATE_FIELDS),
            ("labels", LABEL_FIELDS),
            ("predictions", PREDICTION_FIELDS),
        ):
            file = stack.enter_context((destination / f"{name}.csv").open("w", newline="", encoding="utf-8"))
            writer = csv.writer(file)
            writer.writerow(fields)
            writers.append(writer)
        for directory in directories:
            trip = load_trip(directory)
            route = load_reference_route(directory, len(trip.arcs))
            candidates = generate_candidates(trip, radius_m=radius_m, limit=limit)
            alignment = align_reference(trip, route, max_distance_m=reference_max_distance_m)
            for writer, rows in zip(writers, (
                candidate_rows(trip, candidates), label_rows(trip, alignment), prediction_rows(trip, candidates),
            )):
                writer.writerows(rows)
            counts = baseline_counts(candidates, alignment)
            summaries.append({
                "trip_id": trip.trip_id,
                **counts,
                **baseline_rates(counts),
                **profile_trip(trip, route, candidates, alignment),
                **{f"{ext}_bytes": (directory / f"{trip.trip_id}.{ext}").stat().st_size for ext in EXTENSIONS},
            })
            if trip.trip_id == map_trip:
                save_map(destination / "map.html", trip, candidates, route)

    fields = list(summaries[0])
    write_csv(destination / "trip_summary.csv", fields, (row.values() for row in summaries))
    return summaries


def _split_metrics(rows):
    counts = {key: sum(row[key] for row in rows) for key in (
        "observations", "matched_observations", "labeled_observations", "candidate_misses", "correct_top1",
    )}
    return {"trips": len(rows), **counts, **baseline_rates(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Directory containing train and student_test")
    parser.add_argument("--output", type=Path, required=True, help="New directory for batch outputs")
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--reference-max-distance-m", type=float, default=50.0)
    parser.add_argument("--map-trip", help="Trip ID to plot in its split's output folder")
    args = parser.parse_args()

    try:
        source = args.data_root.resolve()
        output = args.output.absolute()
        if output.exists():
            raise ValueError(f"{output} already exists. Choose a new output directory.")
        directories = {split: _trip_directories(source / split) for split in SPLITS}
        train_ids = {p.name for p in directories["train"]}
        test_ids = {p.name for p in directories["student_test"]}
        if train_ids & test_ids:
            raise ValueError(f"Trip IDs appear in both splits: {', '.join(sorted(train_ids & test_ids))}")
        if args.map_trip is not None and args.map_trip not in train_ids | test_ids:
            raise ValueError(f"Unknown trip ID for the map: {args.map_trip}")

        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".pathsnap-", dir=output.parent) as temporary:
            staging = Path(temporary) / "prepared"
            staging.mkdir()
            summaries = {}
            for split in SPLITS:
                print(f"Preparing {split}…", file=sys.stderr, flush=True)
                summaries[split] = _write_split(
                    directories[split], staging / split, args.radius_m, args.limit,
                    args.reference_max_distance_m, args.map_trip,
                )
            save_profile_plots(staging / "dataset_profile.png", summaries)
            report = {
                "candidate_radius_m": args.radius_m,
                "candidate_limit": args.limit,
                "reference_max_distance_m": args.reference_max_distance_m,
                "splits": {split: _split_metrics(rows) for split, rows in summaries.items()},
            }
            result = json.dumps(report, indent=2)
            (staging / "metrics.json").write_text(result + "\n", encoding="utf-8")
            staging.rename(output)
        print(result)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"PathSnap: {exc}\n")


if __name__ == "__main__":
    main()
