"""Export model scores and top-1 arcs for one trip without loading its reference."""

import argparse
import csv
from pathlib import Path

import numpy as np

from pathsnap.data_preprocessing.candidates import UNMATCHED
from pathsnap.classifier_selection.model import CandidateScorer
from pathsnap.data_preprocessing.trip import load_trip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_directory", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/classifier/model.joblib"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trip = load_trip(args.trip_directory)
    arc_ids, top1, probabilities, log_emissions = CandidateScorer(args.model).predict(trip)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "emissions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("trip_id", "observation_index", "timestamp_s", "arc_id", "match_probability", "log_emission"))
        for i, j in zip(*np.where(arc_ids != UNMATCHED)):
            writer.writerow((trip.trip_id, i, trip.timestamp_s[i], arc_ids[i, j], probabilities[i, j],
                             log_emissions[i, j]))
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("trip_id", "observation_index", "timestamp_s", "predicted_arc_id"))
        for i, arc in enumerate(top1):
            writer.writerow((trip.trip_id, i, trip.timestamp_s[i], arc if arc != UNMATCHED else ""))
    print(f"{trip.trip_id}: scored {len(top1):,} observations and {np.isfinite(probabilities).sum():,} candidates")


if __name__ == "__main__":
    main()
