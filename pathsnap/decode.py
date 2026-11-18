"""Run classifier scoring and HMM/Viterbi decoding on one raw trip."""

import argparse
import csv
import json
from pathlib import Path

from .candidates import UNMATCHED
from .hmm import DecodeConfig, TransitionConfig, build_lattice, route_arc_sequences, viterbi
from .model import CandidateScorer
from .trip import load_trip


def load_config(path):
    if path is None:
        return TransitionConfig(), DecodeConfig()
    settings = json.loads(path.read_text(encoding="utf-8"))
    return TransitionConfig(**settings["transitions"]), DecodeConfig(**settings["viterbi"])


def write_decoding(output, lattice, decoded, routes):
    output.mkdir(parents=True, exist_ok=True)
    trip = lattice.trip
    segment_ids = {}
    for number, segment in enumerate(decoded.segments):
        for index in range(segment.first_observation, segment.last_observation + 1):
            segment_ids[index] = number
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("trip_id", "observation_index", "classifier_arc_id", "decoded_arc_id", "segment_id"))
        for index, (classifier, arc) in enumerate(zip(lattice.classifier_arcs, decoded.arc_ids)):
            writer.writerow((trip.trip_id, index, classifier if classifier != UNMATCHED else "",
                             arc if arc != UNMATCHED else "", segment_ids.get(index, "")))
    segments = [
        {"segment_id": index, "first_observation": segment.first_observation,
         "last_observation": segment.last_observation, "start_reason": segment.reason, "arc_ids": arcs}
        for index, (segment, arcs) in enumerate(zip(decoded.segments, routes))
    ]
    (output / "route.json").write_text(
        json.dumps({"trip_id": trip.trip_id, "segments": segments}, indent=2) + "\n", encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_directory", type=Path)
    parser.add_argument("--model", type=Path, default=Path("results/person2/model.joblib"))
    parser.add_argument("--config", type=Path, help="Decoder JSON produced by the experiment runner")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trip = load_trip(args.trip_directory)
    transitions, config = load_config(args.config)
    lattice = build_lattice(trip, CandidateScorer(args.model), transitions)
    decoded = viterbi(lattice, config)
    routes = route_arc_sequences(lattice, decoded)
    write_decoding(args.output, lattice, decoded, routes)
    print(f"{trip.trip_id}: {len(decoded.arc_ids):,} GPS observations, {len(routes)} decoded segments")


if __name__ == "__main__":
    main()
