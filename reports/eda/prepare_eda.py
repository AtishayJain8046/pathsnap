"""Generate the PathSnap training and student-test EDA figures and tables."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from pathsnap.candidates import UNMATCHED
from pathsnap.experiment_data import partition_paths
from pathsnap.features import FEATURE_NAMES, Preprocessing, build_features
from pathsnap.reference import align_reference
from pathsnap.trip import load_reference_route, load_trip

SPLITS = {"train": (80, "00000002"), "student_test": (10, "00000065")}
LABELS = {"train": "Training", "student_test": "Student test"}
BLUE, GREEN, GOLD = "#326c9c", "#14654e", "#bb713c"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile(root):
    totals, missing_rows, examples = {}, [], {}
    preprocessing = Preprocessing()
    for split, (count, example) in SPLITS.items():
        totals[split] = data = {
            "trips": count, "gps_observations": 0, "aligned_observations": 0, "candidate_rows": 0,
            "labeled_candidate_rows": 0, "candidate_free_observations": 0,
            "candidate_misses_on_aligned_observations": 0, "nearest_correct": 0,
            "gps_intervals": 0, "one_second_intervals": 0, "gaps_over_10_seconds": 0,
            "nodes": 0, "directed_arcs": 0, "zero_length_arcs": 0, "raw_nonfinite_values": 0,
            "max_gap_seconds": 0, "max_gap_trip": "",
        }
        missing_candidates = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
        missing_supervised = missing_candidates.copy()
        for directory in partition_paths(root / split, count):
            trip = load_trip(directory)
            route = load_reference_route(directory, len(trip.arcs))
            reference = align_reference(trip, route, max_distance_m=50).arc_ids
            features = build_features(trip, preprocessing)
            aligned = reference != UNMATCHED
            supervised = aligned[features.rows]
            invalid = ~np.isfinite(features.features)
            missing_candidates += invalid.sum(axis=0)
            missing_supervised += invalid[supervised].sum(axis=0)
            gps_elapsed = np.diff(trip.timestamp_s)
            ends = trip.nodes_xy[trip.arcs]
            data["gps_observations"] += len(aligned)
            data["aligned_observations"] += int(aligned.sum())
            data["candidate_rows"] += len(features.rows)
            data["labeled_candidate_rows"] += int(supervised.sum())
            data["candidate_free_observations"] += int((features.arc_ids[:, 0] == UNMATCHED).sum())
            data["candidate_misses_on_aligned_observations"] += int(
                np.sum(aligned & ~(features.arc_ids == reference[:, None]).any(axis=1))
            )
            data["nearest_correct"] += int(np.sum(aligned & (features.arc_ids[:, 0] == reference)))
            data["gps_intervals"] += len(gps_elapsed)
            data["one_second_intervals"] += int((gps_elapsed == 1).sum())
            data["gaps_over_10_seconds"] += int((gps_elapsed > 10).sum())
            data["nodes"] += len(trip.nodes_xy)
            data["directed_arcs"] += len(trip.arcs)
            data["zero_length_arcs"] += int(np.sum(np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1) == 0))
            data["raw_nonfinite_values"] += sum(int((~np.isfinite(field)).sum()) for field in
                                               (trip.gps_lonlat, trip.timestamp_s, trip.nodes_lonlat, trip.arcs, route))
            if len(gps_elapsed) and gps_elapsed.max() > data["max_gap_seconds"]:
                data["max_gap_seconds"] = float(gps_elapsed.max())
                data["max_gap_trip"] = trip.trip_id
            if trip.trip_id == example:
                examples[split] = trip, route
        for index, feature in enumerate(FEATURE_NAMES):
            missing_rows.append({
                "partition": split, "feature": feature, "all_candidate_rows": data["candidate_rows"],
                "missing_all_candidates": int(missing_candidates[index]),
                "missing_all_candidates_pct": 100 * missing_candidates[index] / data["candidate_rows"],
                "labeled_candidate_rows": data["labeled_candidate_rows"],
                "missing_labeled_candidates": int(missing_supervised[index]),
                "missing_labeled_candidates_pct": 100 * missing_supervised[index] / data["labeled_candidate_rows"],
            })
        print(f"{split}: profiled {count} trips and {data['candidate_rows']:,} candidates", flush=True)
    return totals, missing_rows, examples


def finish_axis(ax, title):
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel("Easting from trip origin (km)")
    ax.set_ylabel("Northing from trip origin (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color="#e4e9e6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_gps(output, examples):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6), layout="constrained")
    fig.suptitle("GPS trajectories by partition", fontsize=17, fontweight="bold")
    for ax, (split, (trip, _)) in zip(axes, examples.items()):
        xy = trip.gps_xy / 1000
        gaps = np.flatnonzero(np.diff(trip.timestamp_s) > 10) + 1
        for segment in np.split(xy, gaps):
            ax.plot(*segment.T, color=BLUE, linewidth=1.4, alpha=0.55)
        ax.scatter(*xy.T, s=3, c=BLUE, alpha=0.6)
        ax.scatter(*xy[0], s=60, c=GREEN, edgecolors="white", linewidths=0.8, label="Start", zorder=4)
        ax.scatter(*xy[-1], s=60, marker="s", c=GOLD, edgecolors="white", linewidths=0.8, label="End", zorder=4)
        ax.legend(loc="best", frameon=False, fontsize=9)
        finish_axis(ax, f"{LABELS[split]} · {trip.trip_id} · {len(xy):,} GPS readings")
    fig.savefig(output / "gps_trajectories.png", dpi=165)
    plt.close(fig)


def plot_roads(output, examples):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6), layout="constrained")
    fig.suptitle("Supplied road networks and reference routes", fontsize=17, fontweight="bold")
    for ax, (split, (trip, route)) in zip(axes, examples.items()):
        unique = np.unique(np.sort(trip.arcs, axis=1), axis=0)
        ax.add_collection(LineCollection(trip.nodes_xy[unique] / 1000, colors="#86978e", linewidths=0.55, alpha=0.6))
        ax.add_collection(LineCollection(trip.nodes_xy[trip.arcs[route]] / 1000, colors=GREEN, linewidths=1.8))
        ax.scatter(*(trip.gps_xy / 1000).T, s=3, c=BLUE, alpha=0.65)
        ax.autoscale()
        ax.legend(handles=[
            Line2D([0], [0], color="#86978e", label="Supplied road network"),
            Line2D([0], [0], color=GREEN, label="Reference route"),
            Line2D([0], [0], color=BLUE, marker=".", linestyle="", label="GPS readings"),
        ], loc="best", frameon=False, fontsize=9)
        finish_axis(ax, f"{LABELS[split]} · {trip.trip_id} · {len(trip.arcs):,} directed arcs")
    fig.savefig(output / "road_networks.png", dpi=165)
    plt.close(fig)


def plot_missing(output, missing_rows):
    active = [name for name in FEATURE_NAMES if any(row["feature"] == name and row["missing_all_candidates"]
                                                    for row in missing_rows)]
    fig, ax = plt.subplots(figsize=(10.6, 4.7), layout="constrained")
    indexed = {(row["partition"], row["feature"]): row for row in missing_rows}
    positions = np.arange(len(active))
    for split, shift, color in (("train", -0.17, BLUE), ("student_test", 0.17, GREEN)):
        ax.barh(positions + shift, [indexed[split, feature]["missing_all_candidates_pct"] for feature in active],
                height=0.32, label=LABELS[split], color=color)
    ax.set_title("Missing values before imputation", loc="left", fontsize=17, fontweight="bold")
    ax.set_yticks(positions, [name.replace("_", " ") for name in active])
    ax.invert_yaxis()
    ax.set_xlabel("Missing rows as a percentage of all candidates in the partition")
    ax.grid(axis="x", color="#e4e9e6")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "missing_values.png", dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    totals, missing_rows, examples = profile(args.data_root)
    (args.output / "dataset_summary.json").write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output / "missing_values.csv", missing_rows)
    plot_gps(args.output, examples)
    plot_roads(args.output, examples)
    plot_missing(args.output, missing_rows)
    print("Created the GPS, network and missing-value figures.")


if __name__ == "__main__":
    main()
