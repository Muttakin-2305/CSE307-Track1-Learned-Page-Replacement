# Results folder

This folder contains the result structure required for Track 1, including raw data, summary tables, traces, and figures.

## Required final outputs

The final student run should generate:

- `raw_results.csv` - per-seed measurements
- `summary.csv` - mean and standard deviation by frame count and policy
- `primary_6_frame_summary.csv` - compact 6-frame comparison for the report
- `hit_ratio_shift.png` - hit-ratio comparison before/after the shift
- `page_faults_shift.png` - page-fault comparison before/after the shift
- `learned_victim_accuracy.png` - learned-policy agreement with Belady
- `decision_tree_rules_frames_*.txt` - readable learned-model rules
- `traces/` - the exact traces used in the final experiment
- `config.json` - final experiment configuration

Run:

```bash
./run_all.sh
```

## Worked development/reference result

A complete development/reference run is kept under `development_example/` so the repository contains a worked result table and figures while the project is being prepared.

At 6 frames, the reference run produced:

| Policy | Before HR | After HR | Drop (pp) | Before faults | After faults |
|---|---:|---:|---:|---:|---:|
| FIFO | 0.8540 | 0.3553 | 49.87 | 43.8 | 193.4 |
| LRU | 0.9067 | 0.3633 | 54.33 | 28.0 | 191.0 |
| Learned | 0.9247 | 0.3020 | 62.27 | 22.6 | 209.4 |
| Optimal | 0.9387 | 0.5647 | 37.40 | 18.4 | 130.6 |

**These are development/reference measurements, not the student's final experimental evidence.** Before submission, I rerun the experiment myself and replace/regenerate the files in the main `results/` directory. I use the same final run for the report, figures, tables, and conclusions.

## Analysis to report after the final run

I compare the policies in two stages:

1. Which policy has the highest hit ratio before the shift?
2. Which policy has the highest hit ratio after the shift?
3. Which policy shows the largest hit-ratio degradation?
4. How do page-fault counts change?
5. Does the learned policy remain competitive after the workload distribution changes?

The explanation should connect the observed results to locality, recency, frequency, the broader/random post-shift workload, and the fact that the learned model is frozen after training on the first phase.
