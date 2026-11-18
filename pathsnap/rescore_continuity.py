"""Apply a team continuity metric to saved student-test predictions without refitting."""

import argparse
import csv
import gzip
import importlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from .candidates import UNMATCHED
from .trip import load_trip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Student folder"))
    parser.add_argument("--predictions", type=Path, default=Path("results/person2/student_test_predictions.csv.gz"))
    parser.add_argument("--metric", required=True, metavar="MODULE:FUNCTION")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module, _, name = args.metric.partition(":")
    evaluate = getattr(importlib.import_module(module), name)

    predictions = defaultdict(dict)
    open_csv = gzip.open if args.predictions.suffix == ".gz" else open
    with open_csv(args.predictions, "rt", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            key = row["model"], row["trip_id"]
            index = int(row["observation_index"])
            if index in predictions[key]:
                raise ValueError(f"Duplicate observation {index} in {key}")
            predictions[key][index] = int(row["predicted_arc_id"] or UNMATCHED)

    trips = {trip_id: load_trip(args.data_root / "student_test" / trip_id)
             for trip_id in sorted({trip_id for _, trip_id in predictions})}
    totals = defaultdict(lambda: [0, 0])
    rows = []
    for (model, trip_id), indices in sorted(predictions.items()):
        trip = trips[trip_id]
        if set(indices) != set(range(len(trip.timestamp_s))):
            raise ValueError(f"Incomplete predictions for {model}/{trip_id}")
        arcs = np.array([indices[i] for i in range(len(trip.timestamp_s))])
        if np.any((arcs < UNMATCHED) | (arcs >= len(trip.arcs))):
            raise ValueError(f"Invalid arc ID for {model}/{trip_id}")
        numerator, denominator = evaluate(trip, arcs)
        totals[model][0] += numerator
        totals[model][1] += denominator
        rows.append((model, trip_id, numerator, denominator, numerator / denominator if denominator else ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("model", "trip_id", "continuous_pairs", "eligible_pairs", "route_continuity"))
        writer.writerows(rows)
        for model, (numerator, denominator) in sorted(totals.items()):
            writer.writerow((model, "ALL", numerator, denominator, numerator / denominator if denominator else ""))


if __name__ == "__main__":
    main()
