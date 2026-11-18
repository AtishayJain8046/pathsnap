"""Plot the saved HMM and classifier results for the two original test partitions."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).parent
BLUE, GREEN = "#436f9e", "#087457"


def rows(name):
    with (OUTPUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finish(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#dfe7e2")
    ax.set_axisbelow(True)


def plot_summary():
    summary = {(row["split"], row["method"]): row for row in rows("summary.csv")}
    fields = ("top1_accuracy", "directed_adjacency", "directed_reachability")
    labels = ("Observation Top-1", "Directed adjacency", "Directed reachability")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained", sharex=True)
    fig.suptitle("PathSnap · effect of HMM/Viterbi decoding", fontsize=17, fontweight="bold")
    for ax, split, partition in zip(axes, ("student_test", "evaluator_test"), ("Student test", "Evaluator test")):
        base, hmm = (summary[split, method] for method in ("classifier_only", "hmm_viterbi"))
        original, decoded = ([100 * float(row[key]) for key in fields] for row in (base, hmm))
        axis = np.arange(len(fields))
        ax.hlines(axis, original, decoded, color="#a7b9ad", linewidth=2.5)
        ax.scatter(original, axis, label="Classifier only", color=BLUE, s=70, zorder=4)
        ax.scatter(decoded, axis, label="HMM/Viterbi", color=GREEN, s=70, zorder=4)
        for y, (before, after) in enumerate(zip(original, decoded)):
            ax.text(before - 0.2, y - 0.21, f"{before:.2f}%", ha="right", color=BLUE, fontsize=9)
            ax.text(after + 0.2, y + 0.25, f"{after:.2f}%", ha="left", color=GREEN, fontsize=9)
        ax.set(yticks=axis, yticklabels=labels, ylim=(-0.55, len(fields) - 0.45),
               xlabel="Percentage", xlim=(87, 102.7), title=f"{partition} · 10 trips")
        ax.invert_yaxis()
        finish(ax)
    axes[0].legend(loc="upper left", frameon=False, ncols=2)
    fig.savefig(OUTPUT / "hmm_comparison.png", dpi=170)
    plt.close(fig)


def plot_trips():
    metrics = {(row["trip_id"], row["method"]): 100 * float(row["top1_accuracy"])
               for row in rows("by_trip.csv") if row["split"] == "evaluator_test"}
    trips = sorted({trip for trip, _ in metrics})
    base = np.asarray([metrics[trip, "classifier_only"] for trip in trips])
    decoded = np.asarray([metrics[trip, "hmm_viterbi"] for trip in trips])
    positions = np.arange(len(trips))
    fig, ax = plt.subplots(figsize=(10.8, 6), layout="constrained")
    ax.hlines(positions, base, decoded, color="#a7b9ad", linewidth=2)
    ax.scatter(base, positions - 0.07, color=BLUE, label="Classifier only", s=48, zorder=3)
    ax.scatter(decoded, positions + 0.07, color=GREEN, label="HMM/Viterbi", s=51, zorder=4)
    for idx, gain in enumerate(decoded - base):
        ax.text(102, idx, f"{gain:+.2f} pp", color=GREEN, va="center", fontsize=9)
    ax.set(xlim=(min(85, base.min() - 3), 108), yticks=positions, yticklabels=trips,
           xlabel="Observation Top-1 accuracy (%)", ylabel="Evaluator trip")
    ax.invert_yaxis()
    ax.set_title("Observation Top-1 on each evaluator trip", loc="left", fontsize=17, fontweight="bold")
    ax.legend(loc="lower left", frameon=False, ncols=2)
    finish(ax)
    fig.savefig(OUTPUT / "evaluator_trips_top1.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_summary()
    plot_trips()
