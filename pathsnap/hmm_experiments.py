"""Tune the HMM on student-test trips and score the fixed decoder on evaluator trips."""

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from .candidates import UNMATCHED
from .continuity import directed_adjacency
from .decode import write_decoding
from .hmm import DecodeConfig, TransitionConfig, build_lattice, reachable_pairs, route_arc_sequences, viterbi
from .model import CandidateScorer
from .reference import align_reference
from .trip import load_reference_route, load_trip

BETA_GRID = (10.0, 20.0, 35.0)
WEIGHT_GRID = (0.5, 1.0, 2.0)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        fields = list(dict.fromkeys(field for row in rows for field in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def raw_lattices(directories, scorer, transitions):
    prepared = []
    for path in directories:
        start = perf_counter()
        lattice = build_lattice(load_trip(path), scorer, transitions)
        elapsed = perf_counter() - start
        prepared.append((path, lattice, elapsed))
        print(f"Prepared {lattice.trip.trip_id}: {len(lattice.trip.timestamp_s):,} observations", flush=True)
    return prepared


def references(rows):
    labels = []
    for directory, lattice, _ in rows:
        route = load_reference_route(directory, len(lattice.trip.arcs))
        aligned = align_reference(lattice.trip, route, max_distance_m=50).arc_ids
        labels.append((route, aligned))
    return labels


def match_counts(lattice, reference, predicted):
    labeled = reference != UNMATCHED
    adjacent, eligible = directed_adjacency(lattice.trip, predicted)
    reachable, eligible_reachable = reachable_pairs(lattice, predicted)
    if eligible != eligible_reachable:
        raise ValueError("Comparison requires the same 10 s gap limit for both continuity scores")
    rows, cols = np.where((lattice.candidate_arcs != UNMATCHED) & labeled[:, None])
    arcs = lattice.candidate_arcs[rows, cols]
    actual = arcs == reference[rows]
    selected = arcs == predicted[rows]
    return {
        "observations": len(reference), "labeled": int(labeled.sum()),
        "top1_correct": int(np.sum(labeled & (predicted == reference))),
        "adjacent_pairs": adjacent, "reachable_pairs": reachable, "eligible_pairs": eligible,
        "true_positive": int(np.sum(actual & selected)), "true_negative": int(np.sum(~actual & ~selected)),
        "false_positive": int(np.sum(~actual & selected)), "false_negative": int(np.sum(actual & ~selected)),
    }


def ratio(value, count):
    return value / count if count else None


def with_rates(counts):
    tp, tn = counts["true_positive"], counts["true_negative"]
    mistakes = counts["false_positive"] + counts["false_negative"]
    positive_f1 = ratio(2 * tp, 2 * tp + mistakes)
    negative_f1 = ratio(2 * tn, 2 * tn + mistakes)
    return {
        **counts, "top1_accuracy": ratio(counts["top1_correct"], counts["labeled"]),
        "directed_adjacency": ratio(counts["adjacent_pairs"], counts["eligible_pairs"]),
        "directed_reachability": ratio(counts["reachable_pairs"], counts["eligible_pairs"]),
        "top1_candidate_macro_f1": ((positive_f1 + negative_f1) / 2
                                   if positive_f1 is not None and negative_f1 is not None else None),
    }


def tune(prepared, refs):
    trials = []
    for beta, weight in product(BETA_GRID, WEIGHT_GRID):
        config = DecodeConfig(beta, weight)
        scores = [
            match_counts(lattice, reference, viterbi(lattice, config).arc_ids)
            for (_, lattice, _), (_, reference) in zip(prepared, refs)
        ]
        values = with_rates({key: sum(row[key] for row in scores) for key in scores[0]})
        trials.append({**asdict(config), **values})
        print(f"Student-test beta={beta:g} weight={weight:g}: Top-1 {values['top1_accuracy']:.4%}", flush=True)
    # The set and order of trials, including tie breaking, are independent of evaluator results.
    best = max(trials, key=lambda row: (row["top1_accuracy"], row["directed_reachability"]))
    return DecodeConfig(best["beta_m"], best["emission_weight"]), trials


def decode_all(prepared, config):
    outputs = []
    for _, lattice, preparation_seconds in prepared:
        start = perf_counter()
        decoded = viterbi(lattice, config)
        routes = route_arc_sequences(lattice, decoded)
        for route in routes:
            if any(lattice.trip.arcs[before, 1] != lattice.trip.arcs[after, 0]
                   for before, after in zip(route, route[1:])):
                raise ValueError(f"Recovered disconnected arc sequence on {lattice.trip.trip_id}")
        outputs.append((decoded, routes, preparation_seconds, perf_counter() - start))
    return outputs


def evaluate(split, prepared, refs, outputs):
    by_trip, summary = [], []
    for method in ("classifier_only", "hmm_viterbi"):
        scored = []
        for (_, lattice, _), (_, reference), (decoded, routes, prep_time, decode_time) in zip(prepared, refs, outputs):
            prediction = lattice.classifier_arcs if method == "classifier_only" else decoded.arc_ids
            counts = match_counts(lattice, reference, prediction)
            scored.append(counts)
            by_trip.append({
                "split": split, "trip_id": lattice.trip.trip_id, "method": method, **with_rates(counts),
                "lattice_seconds": prep_time,
                "viterbi_and_route_seconds": decode_time if method == "hmm_viterbi" else None,
                "route_segments": len(routes) if method == "hmm_viterbi" else None,
            })
        aggregate = {key: sum(row[key] for row in scored) for key in scored[0]}
        summary.append({"split": split, "method": method, **with_rates(aggregate)})
    return summary, by_trip


def evaluator_lattices(zip_file, scratch, scorer, transitions, excluded, config, output):
    manifest = {}
    pattern = re.compile(r"evaluator_test/([0-9]{8})/\1\.(track|nodes|arcs|route)")
    with ZipFile(zip_file) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            match = pattern.fullmatch(member.filename)
            if match is None:
                raise ValueError(f"Unexpected archive member {member.filename}")
            trip, suffix = match.groups()
            parts = manifest.setdefault(trip, {})
            if suffix in parts:
                raise ValueError(f"Duplicate archive member for {trip}.{suffix}")
            parts[suffix] = member
        if len(manifest) != 10 or set(manifest) & excluded:
            raise ValueError("Expected ten evaluator trips distinct from train and student-test trips")

        for trip, files in manifest.items():
            path = scratch / trip
            path.mkdir()
            if set(files) != {"track", "nodes", "arcs", "route"}:
                raise ValueError(f"Incomplete raw files for evaluator trip {trip}")
            for suffix in ("track", "nodes", "arcs"):
                (path / f"{trip}.{suffix}").write_bytes(archive.read(files[suffix]))
        prepared = raw_lattices(sorted(scratch.iterdir()), scorer, transitions)
        decoded = decode_all(prepared, config)
        print("Evaluator paths are fixed. Opening evaluator references for metrics.", flush=True)
        for trip, files in manifest.items():
            (scratch / trip / f"{trip}.route").write_bytes(archive.read(files["route"]))
        refs = references(prepared)
        for (_, lattice, _), (result, routes, _, _) in zip(prepared, decoded):
            write_decoding(output / "evaluator_routes" / lattice.trip.trip_id, lattice, result, routes)
        return evaluate("evaluator_test", prepared, refs, decoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("results/person2/model.joblib"))
    parser.add_argument("--evaluator-zip", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-trip", help="Student-test trip ID to export as an offline map")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scorer = CandidateScorer(args.model)
    transitions = TransitionConfig()
    directories = sorted(path for path in (args.data_root / "student_test").iterdir() if path.is_dir())
    if len(directories) != 10:
        raise ValueError("Expected ten student-test trips")
    prepared = raw_lattices(directories, scorer, transitions)
    refs = references(prepared)
    config, trials = tune(prepared, refs)
    selected = {
        "transitions": asdict(transitions), "viterbi": asdict(config), "model_sha256": sha256(args.model),
        "selection_partition": "student_test", "selection_metric": "observation Top-1, then directed reachability",
    }
    write_json(args.output / "decoder_config.json", selected)
    write_csv(args.output / "student_test_decoder_trials.csv", trials)
    print(f"Locked student-test configuration: {asdict(config)}", flush=True)
    outputs = decode_all(prepared, config)
    summary, by_trip = evaluate("student_test", prepared, refs, outputs)
    for (_, lattice, _), (decoded, routes, _, _) in zip(prepared, outputs):
        write_decoding(args.output / "student_test_routes" / lattice.trip.trip_id, lattice, decoded, routes)
    if args.map_trip:
        from .visualize import save_decoded_map

        sample = next((i for i, (_, lattice, _) in enumerate(prepared) if lattice.trip.trip_id == args.map_trip), None)
        if sample is None:
            raise ValueError("--map-trip must identify a student-test trip")
        _, lattice, _ = prepared[sample]
        save_decoded_map(args.output / f"student_test_{args.map_trip}.html", lattice.trip, lattice.classifier_arcs,
                         outputs[sample][1], refs[sample][0])

    if args.evaluator_zip:
        excluded = {path.name for path in directories} | {
            path.name for path in (args.data_root / "train").iterdir() if path.is_dir()
        }
        with TemporaryDirectory(dir=args.output) as scratch:
            evaluator_summary, evaluator_trips = evaluator_lattices(
                args.evaluator_zip, Path(scratch), scorer, transitions, excluded, config, args.output,
            )
        summary.extend(evaluator_summary)
        by_trip.extend(evaluator_trips)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "by_trip.csv", by_trip)
    write_json(args.output / "run.json", {
        "decoder": selected, "evaluator_zip_sha256": sha256(args.evaluator_zip) if args.evaluator_zip else None,
        "comparison": summary,
    })


if __name__ == "__main__":
    main()
