"""Synthetic trace generator matching the Track 1 shift requirement."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def generate_shifted_trace(
    seed: int,
    phase_length: int = 300,
    page_count: int = 32,
) -> Tuple[List[int], int]:
    """Generate locality-heavy/sequential first half and random/bursty second half."""
    rng = np.random.default_rng(seed)
    first: List[int] = []

    # First phase: strong locality around a five-page working set. Most
    # references follow a sequential cursor, with occasional accesses outside
    # the hot set so the replacement policies still have meaningful misses.
    hot_sets = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
    ]
    cursor = 0
    while len(first) < phase_length:
        block = hot_sets[(len(first) // 100) % len(hot_sets)]
        for _ in range(min(100, phase_length - len(first))):
            r = rng.random()
            if r < 0.72:
                page = block[cursor % len(block)]
                cursor += 1
            elif r < 0.90:
                page = int(rng.choice(block))
            else:
                page = int(rng.integers(0, 12))
            first.append(page)
            if len(first) >= phase_length:
                break

    # Second phase: mostly global random accesses, with shorter bursts mixed
    # in. The page universe expands from the first phase's small working sets
    # to the full page space, creating a clear locality/distribution shift.
    second: List[int] = []
    while len(second) < phase_length:
        r = rng.random()
        if r < 0.30:
            burst_start = int(rng.integers(0, page_count - 4))
            burst_set = np.arange(burst_start, burst_start + 4)
            burst_len = int(rng.integers(5, 10))
            for _ in range(burst_len):
                if len(second) >= phase_length:
                    break
                if rng.random() < 0.65:
                    second.append(int(rng.choice(burst_set)))
                else:
                    second.append(int(rng.integers(0, page_count)))
        else:
            second.append(int(rng.integers(0, page_count)))

    return first + second, phase_length
