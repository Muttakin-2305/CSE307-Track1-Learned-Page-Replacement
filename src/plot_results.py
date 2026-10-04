"""Create report-ready figures from results/summary.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    summary = pd.read_csv(RESULTS / "summary.csv")
    primary = summary[summary["frame_count"] == 6].copy()

    labels = primary["policy"].tolist()
    before = primary["before_hit_ratio_mean"].tolist()
    after = primary["after_hit_ratio_mean"].tolist()

    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar([i - width / 2 for i in x], before, width, label="Before shift")
    ax.bar([i + width / 2 for i in x], after, width, label="After shift")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Hit ratio")
    ax.set_ylim(0, 1)
    ax.set_title("Track 1: Hit ratio before and after workload shift (6 frames)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "hit_ratio_shift.png", dpi=200)
    plt.close(fig)

    before_f = primary["before_fault_mean"].tolist()
    after_f = primary["after_fault_mean"].tolist()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar([i - width / 2 for i in x], before_f, width, label="Before shift")
    ax.bar([i + width / 2 for i in x], after_f, width, label="After shift")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Mean page faults")
    ax.set_title("Track 1: Page faults before and after workload shift (6 frames)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "page_faults_shift.png", dpi=200)
    plt.close(fig)

    learned = primary[primary["policy"] == "Learned"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(["Victim-choice accuracy"], learned["victim_accuracy_mean"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Learned policy: exact Belady-victim agreement (6 frames)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "learned_victim_accuracy.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
