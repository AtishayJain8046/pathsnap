"""Draw model comparison and search figures from the saved experiment log."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {
    "logistic_regression": "Logistic regression",
    "gaussian_naive_bayes": "Gaussian Naive Bayes",
    "decision_tree": "Decision tree",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "Hist. gradient boosting",
    "nearest_road": "Nearest road",
}


def display_name(name):
    base = name.removesuffix("_tuned")
    return NAMES[base] + (" (tuned)" if base != name else "")


def draw_comparison(output, run):
    rows = run["comparison"]
    names = [display_name(row["model"]) for row in rows]
    x = np.arange(len(rows))
    trained = [i for i, row in enumerate(rows) if "macro_f1" in row]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), layout="constrained")
    fig.suptitle("PathSnap model comparison", fontsize=17, fontweight="bold")
    axes[0, 0].bar(
        trained, [rows[i]["cv_macro_f1_mean"] for i in trained],
        yerr=[rows[i]["cv_macro_f1_sd"] for i in trained], capsize=3, color="#3668a2",
    )
    axes[0, 0].set_title("Training CV candidate Macro F1 (mean ± SD)")
    axes[0, 0].set_ylim(0, 1)

    width = 0.35
    axes[0, 1].bar(x - width / 2, [row["top1_accuracy"] for row in rows], width,
                   label="Observation Top-1", color="#136d66")
    axes[0, 1].bar(np.array(trained) + width / 2, [rows[i]["macro_f1"] for i in trained], width,
                   label="Candidate Macro F1", color="#3668a2")
    axes[0, 1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                     ncol=2, frameon=False, fontsize=8)
    axes[0, 1].set_title("Student-test classification", pad=27)
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].bar(trained, [rows[i]["log_loss"] for i in trained], color="#964c27")
    axes[1, 0].set_title("Student-test candidate log loss")
    axes[1, 0].set_ylabel("Natural log")

    axes[1, 1].bar(x, [row["route_continuity"] for row in rows], color="#64628e")
    continuity = "provisional" if run["continuity_status"] == "provisional" else "team metric"
    axes[1, 1].set_title(f"Student-test route continuity ({continuity})")
    axes[1, 1].set_ylim(0, 1)

    for ax in axes.flat:
        ax.set_xticks(x, names, rotation=29, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.grid(axis="y", alpha=0.24)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "model_comparison.png", dpi=165)
    plt.close(fig)


def draw_search(output, search, run):
    trials = search["trials"]
    x = [trial["trial"] for trial in trials]
    scores = [trial["cv_macro_f1_mean"] for trial in trials]
    baseline = next(row["cv_macro_f1_mean"] for row in run["comparison"]
                    if row["model"] == search["selected_model"])
    fig, ax = plt.subplots(figsize=(8.5, 4.4), layout="constrained")
    ax.axhline(baseline, color="#964c27", linestyle="--", label="Untuned")
    ax.scatter(x, scores, color="#3668a2", label="Trial")
    ax.step(x, np.maximum.accumulate(scores), where="post", color="#136d66", label="Best trial so far")
    ax.set(xticks=x, xlabel="Trial", ylabel="Mean grouped CV candidate Macro F1",
           title=f"Bayesian optimization of {display_name(search['selected_model'])}")
    ax.grid(axis="y", alpha=0.24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.savefig(output / "bayesian_search.png", dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/person2"))
    args = parser.parse_args()
    run = json.loads((args.results / "run.json").read_text(encoding="utf-8"))
    search = json.loads((args.results / "bayesian_search.json").read_text(encoding="utf-8"))
    draw_comparison(args.results, run)
    draw_search(args.results, search, run)


if __name__ == "__main__":
    main()
