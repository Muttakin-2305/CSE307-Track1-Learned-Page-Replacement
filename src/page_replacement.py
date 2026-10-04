"""Classical and learned page-replacement simulators for CSE-307 Track 1.

The learned policy uses a small decision tree trained on past-only candidate
features (recency, frequency, and rank-based summaries). Training labels are
created by asking Belady's optimal policy which currently resident page should
be evicted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text


@dataclass
class SimulationResult:
    policy: str
    frame_count: int
    hits: int
    faults: int
    hit_ratio: float
    fault_ratio: float
    victim_accuracy: float | None = None
    total_accesses: int = 0
    phase_hits: Dict[str, int] | None = None
    phase_faults: Dict[str, int] | None = None
    phase_hit_ratio: Dict[str, float] | None = None


def fifo(trace: Sequence[int], frame_count: int) -> Tuple[int, int]:
    """Return (hits, faults) under FIFO."""
    frames: List[int] = []
    pointer = 0
    hits = faults = 0
    for page in trace:
        if page in frames:
            hits += 1
            continue
        faults += 1
        if len(frames) < frame_count:
            frames.append(page)
        else:
            frames[pointer] = page
            pointer = (pointer + 1) % frame_count
    return hits, faults


def lru(trace: Sequence[int], frame_count: int) -> Tuple[int, int]:
    """Return (hits, faults) under LRU."""
    frames: List[int] = []
    last_used: Dict[int, int] = {}
    hits = faults = 0
    for t, page in enumerate(trace):
        if page in frames:
            hits += 1
        else:
            faults += 1
            if len(frames) < frame_count:
                frames.append(page)
            else:
                victim = min(frames, key=lambda p: last_used[p])
                frames[frames.index(victim)] = page
        last_used[page] = t
    return hits, faults


def _optimal_victim(frames: Sequence[int], future_trace: Sequence[int]) -> int:
    """Return the page whose next reference is farthest in the future."""
    best_page = frames[0]
    best_next = -1
    future_index = {page: [] for page in frames}
    for idx, page in enumerate(future_trace):
        if page in future_index:
            future_index[page].append(idx)

    for page in frames:
        positions = future_index[page]
        next_use = positions[0] if positions else float("inf")
        if next_use > best_next:
            best_page = page
            best_next = next_use
    return best_page


def optimal(trace: Sequence[int], frame_count: int) -> Tuple[int, int]:
    """Return (hits, faults) using Belady's optimal replacement."""
    frames: List[int] = []
    hits = faults = 0
    for t, page in enumerate(trace):
        if page in frames:
            hits += 1
            continue
        faults += 1
        if len(frames) < frame_count:
            frames.append(page)
        else:
            victim = _optimal_victim(frames, trace[t + 1 :])
            frames[frames.index(victim)] = page
    return hits, faults


FEATURE_NAMES = [
    "recency_distance",
    "frequency",
    "recency_rank",
    "frequency_rank",
]


def _candidate_features(
    candidate: int,
    frames: Sequence[int],
    last_used: Dict[int, int],
    frequency: Dict[int, int],
    now: int,
) -> List[float]:
    """Past-only features available immediately before an eviction decision."""
    recencies = {p: now - last_used[p] for p in frames}
    freqs = {p: frequency[p] for p in frames}

    # Rank 1 is best: most recent / most frequent. Ties are broken by page id.
    recency_order = sorted(frames, key=lambda p: (recencies[p], p))
    freq_order = sorted(frames, key=lambda p: (-freqs[p], p))
    recency_rank = recency_order.index(candidate) + 1
    frequency_rank = freq_order.index(candidate) + 1

    return [
        float(recencies[candidate]),
        float(freqs[candidate]),
        float(recency_rank),
        float(frequency_rank),
    ]


def build_training_data(
    trace: Sequence[int],
    frame_count: int,
    train_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect candidate examples from an LRU state during the training phase.

    Each eviction candidate becomes one row. The label is 1 iff Belady would
    evict that candidate from the current LRU resident set.
    """
    trace = list(trace)
    frames: List[int] = []
    last_used: Dict[int, int] = {}
    frequency: Dict[int, int] = {}
    X: List[List[float]] = []
    y: List[int] = []

    for t, page in enumerate(trace[:train_end]):
        frequency[page] = frequency.get(page, 0) + 1
        if page in frames:
            last_used[page] = t
            continue

        if len(frames) >= frame_count:
            future = trace[t + 1 : train_end]
            victim = _optimal_victim(frames, future)
            for candidate in frames:
                X.append(_candidate_features(candidate, frames, last_used, frequency, t))
                y.append(1 if candidate == victim else 0)
            lru_victim = min(frames, key=lambda p: last_used[p])
            frames[frames.index(lru_victim)] = page
        else:
            frames.append(page)

        last_used[page] = t

    if not X:
        raise ValueError("Not enough training misses to build learned-policy examples.")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def train_learned_model(trace: Sequence[int], frame_count: int, train_end: int) -> DecisionTreeClassifier:
    """Train a shallow decision tree on pre-shift data only."""
    X, y = build_training_data(trace, frame_count, train_end)
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y)
    return model


def learned_policy(
    trace: Sequence[int],
    frame_count: int,
    model: DecisionTreeClassifier,
) -> Tuple[int, int, float]:
    """Run the frozen learned evictor and return (hits, faults, victim accuracy)."""
    frames: List[int] = []
    last_used: Dict[int, int] = {}
    frequency: Dict[int, int] = {}
    hits = faults = correct_victims = decision_count = 0

    for t, page in enumerate(trace):
        frequency[page] = frequency.get(page, 0) + 1
        if page in frames:
            hits += 1
            last_used[page] = t
            continue

        faults += 1
        if len(frames) < frame_count:
            frames.append(page)
        else:
            candidates = list(frames)
            X = np.asarray(
                [_candidate_features(p, candidates, last_used, frequency, t) for p in candidates],
                dtype=float,
            )
            if 1 in model.classes_:
                class1_index = int(np.where(model.classes_ == 1)[0][0])
                probabilities = model.predict_proba(X)[:, class1_index]
                # Deterministic tie-breaks: highest eviction probability,
                # then oldest (LRU), then lowest page id.
                scored = list(zip(candidates, probabilities))
                victim = max(
                    scored,
                    key=lambda item: (float(item[1]), -last_used[item[0]], -item[0]),
                )[0]
            else:
                victim = min(candidates, key=lambda p: (last_used[p], p))

            oracle = _optimal_victim(candidates, trace[t + 1 :])
            correct_victims += int(victim == oracle)
            decision_count += 1
            frames[frames.index(victim)] = page

        last_used[page] = t

    accuracy = correct_victims / decision_count if decision_count else 1.0
    return hits, faults, accuracy


def _run_fifo_phases(trace: Sequence[int], frame_count: int, boundary: int):
    frames: List[int] = []
    pointer = 0
    hits = faults = 0
    phase_hits = {"before_shift": 0, "after_shift": 0}
    phase_faults = {"before_shift": 0, "after_shift": 0}
    for t, page in enumerate(trace):
        phase = "before_shift" if t < boundary else "after_shift"
        if page in frames:
            hits += 1
            phase_hits[phase] += 1
            continue
        faults += 1
        phase_faults[phase] += 1
        if len(frames) < frame_count:
            frames.append(page)
        else:
            frames[pointer] = page
            pointer = (pointer + 1) % frame_count
    return hits, faults, phase_hits, phase_faults, None


def _run_lru_phases(trace: Sequence[int], frame_count: int, boundary: int):
    frames: List[int] = []
    last_used: Dict[int, int] = {}
    hits = faults = 0
    phase_hits = {"before_shift": 0, "after_shift": 0}
    phase_faults = {"before_shift": 0, "after_shift": 0}
    for t, page in enumerate(trace):
        phase = "before_shift" if t < boundary else "after_shift"
        if page in frames:
            hits += 1
            phase_hits[phase] += 1
        else:
            faults += 1
            phase_faults[phase] += 1
            if len(frames) < frame_count:
                frames.append(page)
            else:
                victim = min(frames, key=lambda p: (last_used[p], p))
                frames[frames.index(victim)] = page
        last_used[page] = t
    return hits, faults, phase_hits, phase_faults, None


def _run_optimal_phases(trace: Sequence[int], frame_count: int, boundary: int):
    frames: List[int] = []
    hits = faults = 0
    phase_hits = {"before_shift": 0, "after_shift": 0}
    phase_faults = {"before_shift": 0, "after_shift": 0}
    for t, page in enumerate(trace):
        phase = "before_shift" if t < boundary else "after_shift"
        if page in frames:
            hits += 1
            phase_hits[phase] += 1
            continue
        faults += 1
        phase_faults[phase] += 1
        if len(frames) < frame_count:
            frames.append(page)
        else:
            victim = _optimal_victim(frames, trace[t + 1 :])
            frames[frames.index(victim)] = page
    return hits, faults, phase_hits, phase_faults, None


def _run_learned_phases(
    trace: Sequence[int], frame_count: int, boundary: int, model: DecisionTreeClassifier
):
    frames: List[int] = []
    last_used: Dict[int, int] = {}
    frequency: Dict[int, int] = {}
    hits = faults = correct_victims = decision_count = 0
    phase_hits = {"before_shift": 0, "after_shift": 0}
    phase_faults = {"before_shift": 0, "after_shift": 0}

    for t, page in enumerate(trace):
        phase = "before_shift" if t < boundary else "after_shift"
        frequency[page] = frequency.get(page, 0) + 1
        if page in frames:
            hits += 1
            phase_hits[phase] += 1
            last_used[page] = t
            continue

        faults += 1
        phase_faults[phase] += 1
        if len(frames) < frame_count:
            frames.append(page)
        else:
            candidates = list(frames)
            X = np.asarray(
                [_candidate_features(p, candidates, last_used, frequency, t) for p in candidates],
                dtype=float,
            )
            if 1 in model.classes_:
                class1_index = int(np.where(model.classes_ == 1)[0][0])
                probabilities = model.predict_proba(X)[:, class1_index]
                victim = max(
                    zip(candidates, probabilities),
                    key=lambda item: (float(item[1]), -last_used[item[0]], -item[0]),
                )[0]
            else:
                victim = min(candidates, key=lambda p: (last_used[p], p))

            oracle = _optimal_victim(candidates, trace[t + 1 :])
            correct_victims += int(victim == oracle)
            decision_count += 1
            frames[frames.index(victim)] = page
        last_used[page] = t

    accuracy = correct_victims / decision_count if decision_count else 1.0
    return hits, faults, phase_hits, phase_faults, accuracy


def simulate_with_phases(
    trace: Sequence[int],
    frame_count: int,
    phase_boundary: int,
    model: DecisionTreeClassifier | None = None,
) -> List[SimulationResult]:
    """Evaluate all required policies on the exact same full trace.

    Phase counts are accumulated during one continuous run so the second phase
    starts with the memory state produced by the first phase.
    """
    runners = [
        ("FIFO", _run_fifo_phases),
        ("LRU", _run_lru_phases),
        ("Optimal", _run_optimal_phases),
    ]
    if model is not None:
        runners.append(("Learned", _run_learned_phases))

    results: List[SimulationResult] = []
    for name, runner in runners:
        if name == "Learned":
            h, f, phase_hits, phase_faults, acc = runner(trace, frame_count, phase_boundary, model)
        else:
            h, f, phase_hits, phase_faults, acc = runner(trace, frame_count, phase_boundary)
        phase_lengths = {"before_shift": phase_boundary, "after_shift": len(trace) - phase_boundary}
        phase_ratios = {
            phase: phase_hits[phase] / max(1, phase_lengths[phase]) for phase in phase_lengths
        }
        results.append(
            SimulationResult(
                policy=name,
                frame_count=frame_count,
                hits=h,
                faults=f,
                hit_ratio=h / len(trace),
                fault_ratio=f / len(trace),
                victim_accuracy=acc,
                total_accesses=len(trace),
                phase_hits=phase_hits,
                phase_faults=phase_faults,
                phase_hit_ratio=phase_ratios,
            )
        )
    return results


def readable_tree(model: DecisionTreeClassifier) -> str:
    """Return simple text rules for README/demo use."""
    return export_text(model, feature_names=FEATURE_NAMES, decimals=2)
