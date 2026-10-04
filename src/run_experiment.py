"""Run the complete CSE-307 Track 1 experiment and write reproducible results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from page_replacement import simulate_with_phases, train_learned_model, readable_tree
from trace_generator import generate_shifted_trace


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TRACE_DIR = RESULTS / "traces"


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 1 page replacement experiment")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--frames", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--phase-length", type=int, default=300)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    tree_rule_paths: Dict[int, Path] = {}

    for seed in args.seeds:
        trace, boundary = generate_shifted_trace(seed, phase_length=args.phase_length)
        pd.DataFrame({"t": range(len(trace)), "page": trace, "phase": ["before_shift"] * boundary + ["after_shift"] * boundary}).to_csv(
            TRACE_DIR / f"trace_seed_{seed}.csv", index=False
        )

        for frame_count in args.frames:
            model = train_learned_model(trace, frame_count, boundary)
            if frame_count not in tree_rule_paths:
                tree_rule_path = RESULTS / f"decision_tree_rules_frames_{frame_count}.txt"
                tree_rule_path.write_text(readable_tree(model), encoding="utf-8")
                tree_rule_paths[frame_count] = tree_rule_path

            sim_results = simulate_with_phases(trace, frame_count, boundary, model)
            for result in sim_results:
                rows.append(
                    {
                        "seed": seed,
                        "frame_count": frame_count,
                        "policy": result.policy,
                        "hits": result.hits,
                        "faults": result.faults,
                        "hit_ratio": result.hit_ratio,
                        "before_hits": result.phase_hits["before_shift"],
                        "before_faults": result.phase_faults["before_shift"],
                        "before_hit_ratio": result.phase_hit_ratio["before_shift"],
                        "after_hits": result.phase_hits["after_shift"],
                        "after_faults": result.phase_faults["after_shift"],
                        "after_hit_ratio": result.phase_hit_ratio["after_shift"],
                        "victim_accuracy": result.victim_accuracy,
                    }
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "raw_results.csv", index=False)

    summary = (
        raw.groupby(["frame_count", "policy"], as_index=False)
        .agg(
            hit_ratio_mean=("hit_ratio", "mean"),
            hit_ratio_std=("hit_ratio", "std"),
            fault_mean=("faults", "mean"),
            fault_std=("faults", "std"),
            before_hit_ratio_mean=("before_hit_ratio", "mean"),
            after_hit_ratio_mean=("after_hit_ratio", "mean"),
            before_fault_mean=("before_faults", "mean"),
            after_fault_mean=("after_faults", "mean"),
            victim_accuracy_mean=("victim_accuracy", "mean"),
        )
    )
    summary.to_csv(RESULTS / "summary.csv", index=False)

    # Primary frame setting for report/demo: 6 frames.
    primary = summary[summary["frame_count"] == 6].copy()
    primary["hit_ratio_drop_pp"] = (primary["before_hit_ratio_mean"] - primary["after_hit_ratio_mean"]) * 100
    primary["fault_increase"] = primary["after_fault_mean"] - primary["before_fault_mean"]
    primary.to_csv(RESULTS / "primary_6_frame_summary.csv", index=False)

    config = {
        "seeds": args.seeds,
        "frames": args.frames,
        "phase_length": args.phase_length,
        "total_accesses_per_trace": args.phase_length * 2,
        "page_count": 32,
        "training": {
            "source": "before-shift phase",
            "base_state_for_candidate_examples": "LRU",
            "label": "Belady victim among current candidates",
            "classifier": "DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, class_weight='balanced', random_state=42)",
        },
        "primary_frame_count_for_report": 6,
    }
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Wrote:")
    print(RESULTS / "raw_results.csv")
    print(RESULTS / "summary.csv")
    print(RESULTS / "primary_6_frame_summary.csv")


if __name__ == "__main__":
    main()
