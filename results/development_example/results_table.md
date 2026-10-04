# Primary results - 6 frames

Mean over 5 deterministic traces (seeds 11, 22, 33, 44, 55).

| Policy | Overall hit ratio | Before hit ratio | After hit ratio | Hit-ratio drop | Overall faults | Before faults | After faults | Learned victim accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 0.6047 | 0.8540 | 0.3553 | 49.87 pp | 237.2 | 43.8 | 193.4 | - |
| LRU | 0.6350 | 0.9067 | 0.3633 | 54.33 pp | 219.0 | 28.0 | 191.0 | - |
| Learned | 0.6133 | 0.9247 | 0.3020 | **62.27 pp** | 232.0 | 22.6 | 209.4 | 0.2172 |
| Optimal | 0.7517 | 0.9387 | 0.5647 | 37.40 pp | 149.0 | 18.4 | 130.6 | - |

**Main observation:** the learned policy has the highest pre-shift hit ratio among the non-optimal policies, but it experiences the largest degradation after the workload shift. This is the central distribution-shift finding of the experiment.
