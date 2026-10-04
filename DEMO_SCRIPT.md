# 3-5 Minute Demo Script

## 0:00-0:45 - Problem
"When memory has a fixed number of frames and a requested page is missing, the OS has to decide which resident page to evict. FIFO and LRU use fixed heuristics. Belady's Optimal gives a benchmark using future information. My question is whether a small learned rule can make a good eviction decision and how it behaves after the workload changes."

## 0:45-1:30 - Code structure
Open `src/page_replacement.py` and point out:

1. `fifo()` - evicts the oldest page.
2. `lru()` - evicts the least recently used page.
3. `optimal()` - chooses the farthest next use in the future.
4. `build_training_data()` - creates candidate rows and Belady labels.
5. `train_learned_model()` - fits the shallow decision tree.
6. `learned_policy()` - uses only past-derived features at runtime.

Then show `src/trace_generator.py` and explain the two phases.

## 1:30-2:15 - Experiment
"I use the same trace for every policy, five deterministic seeds, 32 possible pages, 600 references per trace, and frame counts 4, 6, and 8. The main report compares six frames."

Run:

```bash
PYTHONPATH=src python src/run_experiment.py
PYTHONPATH=src python src/plot_results.py
```

## 2:15-3:15 - Results
Open the final `results/hit_ratio_shift.png` generated from your own experiment.

Use your final 6-frame values from `results/primary_6_frame_summary.csv` and say which policy had the best pre-shift hit ratio, which had the best post-shift hit ratio, and which suffered the largest degradation.

The repository also contains a clearly labelled development/reference run under `results/development_example/`. Its numbers are only there as a worked example while preparing the project and should not be presented as personal experimental measurements.

## 3:15-4:15 - Why
"The model learns recency/frequency patterns from the locality-heavy phase. After the shift, those correlations may change. Because the tree is frozen, it cannot immediately learn the new pattern. I explain the observed robustness or degradation using the actual measurements from my final run."

## 4:15-5:00 - Questions
Be ready to explain:

- Why Optimal is not realistic: it needs future references.
- Why future references do not leak into the learned features: they are used only for offline labels and evaluation.
- Why use a decision tree: lightweight, interpretable, explicitly allowed by the brief.
- Why the shift matters: it tests whether the learned heuristic generalizes when the data distribution changes.
