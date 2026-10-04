"""Correctness and regression tests for CSE-307 Track 1 page replacement."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import numpy as np
from page_replacement import (
    fifo,
    lru,
    optimal,
    build_training_data,
    train_learned_model,
    learned_policy,
    simulate_with_phases,
    SimulationResult,
    FEATURE_NAMES,
)
from trace_generator import generate_shifted_trace


# ---------------------------------------------------------------------------
# 1. Classical algorithms: known reference string
# ---------------------------------------------------------------------------

CLASSIC_TRACE = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2]


def test_fifo_reference_string():
    """FIFO on classic OS-textbook example (3 frames -> 10 faults, 3 hits)."""
    hits, faults = fifo(CLASSIC_TRACE, 3)
    assert hits == 3
    assert faults == 10


def test_lru_reference_string():
    """LRU on classic OS-textbook example (3 frames -> 9 faults, 4 hits)."""
    hits, faults = lru(CLASSIC_TRACE, 3)
    assert hits == 4
    assert faults == 9


def test_optimal_reference_string():
    """Optimal on classic OS-textbook example (3 frames -> 7 faults, 6 hits)."""
    hits, faults = optimal(CLASSIC_TRACE, 3)
    assert hits == 6
    assert faults == 7


def test_classical_reference_string():
    """Combined assertion kept for backwards compatibility."""
    assert fifo(CLASSIC_TRACE, 3) == (3, 10)
    assert lru(CLASSIC_TRACE, 3) == (4, 9)
    assert optimal(CLASSIC_TRACE, 3) == (6, 7)


# ---------------------------------------------------------------------------
# 2. Edge cases
# ---------------------------------------------------------------------------


def test_fifo_single_frame():
    trace = [1, 2, 3, 1, 2, 3]
    hits, faults = fifo(trace, 1)
    assert hits == 0
    assert faults == 6


def test_lru_all_hits():
    """With enough frames all accesses should be hits after the first pass."""
    trace = [0, 1, 2, 3, 0, 1, 2, 3]
    hits, faults = lru(trace, 4)
    assert hits == 4
    assert faults == 4


def test_optimal_with_large_frames():
    """With more frames than distinct pages there should be no faults after warm-up."""
    trace = [0, 1, 2, 0, 1, 2]
    hits, faults = optimal(trace, 10)
    assert faults == 3  # cold misses only
    assert hits == 3


def test_fifo_equals_lru_single_page():
    """Single page repeated should give identical results for all policies."""
    trace = [5] * 20
    assert fifo(trace, 3) == lru(trace, 3) == optimal(trace, 3)


def test_hits_plus_faults_equals_trace_length():
    """Invariant: hits + faults == len(trace) for all classical policies."""
    trace = [1, 3, 0, 3, 5, 6, 3]
    for frames in (2, 3, 5):
        h, f = fifo(trace, frames)
        assert h + f == len(trace), f"FIFO {frames} frames: {h}+{f} != {len(trace)}"
        h, f = lru(trace, frames)
        assert h + f == len(trace), f"LRU {frames} frames: {h}+{f} != {len(trace)}"
        h, f = optimal(trace, frames)
        assert h + f == len(trace), f"Optimal {frames} frames: {h}+{f} != {len(trace)}"


def test_optimal_never_worse_than_lru():
    """Optimal should never produce more faults than LRU."""
    trace, _ = generate_shifted_trace(seed=42)
    for frames in (4, 6, 8):
        _, opt_faults = optimal(trace, frames)
        _, lru_faults = lru(trace, frames)
        assert opt_faults <= lru_faults, (
            f"Optimal ({opt_faults}) > LRU ({lru_faults}) at {frames} frames"
        )


def test_optimal_never_worse_than_fifo():
    """Optimal should never produce more faults than FIFO."""
    trace, _ = generate_shifted_trace(seed=99)
    for frames in (4, 6, 8):
        _, opt_faults = optimal(trace, frames)
        _, fifo_faults = fifo(trace, frames)
        assert opt_faults <= fifo_faults, (
            f"Optimal ({opt_faults}) > FIFO ({fifo_faults}) at {frames} frames"
        )


# ---------------------------------------------------------------------------
# 3. More frames -> fewer or equal faults (monotonicity)
# ---------------------------------------------------------------------------


def test_more_frames_fewer_faults_lru():
    trace, _ = generate_shifted_trace(seed=11)
    _, prev_faults = lru(trace, 2)
    for frames in range(3, 9):
        _, faults = lru(trace, frames)
        assert faults <= prev_faults, (
            f"LRU fault count rose from {prev_faults} (frames={frames - 1}) "
            f"to {faults} (frames={frames})"
        )
        prev_faults = faults


def test_more_frames_fifo_no_crash():
    """FIFO may exhibit Belady anomaly but must never crash."""
    trace, _ = generate_shifted_trace(seed=22)
    results = [fifo(trace, f)[1] for f in range(2, 9)]
    assert all(f >= 0 for f in results)


# ---------------------------------------------------------------------------
# 4. Trace generator
# ---------------------------------------------------------------------------


def test_trace_generator_deterministic():
    """Same seed should produce the same trace."""
    t1, b1 = generate_shifted_trace(seed=42)
    t2, b2 = generate_shifted_trace(seed=42)
    assert t1 == t2
    assert b1 == b2


def test_trace_generator_different_seeds():
    """Different seeds should produce different traces."""
    t1, _ = generate_shifted_trace(seed=1)
    t2, _ = generate_shifted_trace(seed=2)
    assert t1 != t2


def test_trace_generator_phase_boundary():
    """Boundary should equal phase_length and total trace is 2x."""
    for pl in (100, 200, 300):
        trace, boundary = generate_shifted_trace(seed=7, phase_length=pl)
        assert boundary == pl
        assert len(trace) == pl * 2


def test_trace_generator_page_range():
    """All pages must be in [0, page_count)."""
    trace, _ = generate_shifted_trace(seed=13, page_count=32)
    assert all(0 <= p < 32 for p in trace), "Page out of range"


# ---------------------------------------------------------------------------
# 5. Learned policy training
# ---------------------------------------------------------------------------


def test_build_training_data_shapes():
    trace, boundary = generate_shifted_trace(seed=11)
    X, y = build_training_data(trace, frame_count=6, train_end=boundary)
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(FEATURE_NAMES)


def test_build_training_data_binary_labels():
    trace, boundary = generate_shifted_trace(seed=22)
    _, y = build_training_data(trace, frame_count=6, train_end=boundary)
    assert set(y).issubset({0, 1}), "Labels must be 0 or 1"


def test_train_learned_model_returns_fitted():
    """Model should be fitted and have learned classes."""
    trace, boundary = generate_shifted_trace(seed=33)
    model = train_learned_model(trace, frame_count=6, train_end=boundary)
    assert hasattr(model, "classes_")
    assert model.n_features_in_ == len(FEATURE_NAMES)


def test_learned_policy_hits_plus_faults():
    """Invariant: hits + faults == len(trace) for the learned policy."""
    trace, boundary = generate_shifted_trace(seed=44)
    model = train_learned_model(trace, frame_count=6, train_end=boundary)
    hits, faults, accuracy = learned_policy(trace, 6, model)
    assert hits + faults == len(trace)
    assert 0.0 <= accuracy <= 1.0


# ---------------------------------------------------------------------------
# 6. simulate_with_phases
# ---------------------------------------------------------------------------


def test_simulate_with_phases_returns_four_policies():
    trace, boundary = generate_shifted_trace(seed=55)
    model = train_learned_model(trace, 6, boundary)
    results = simulate_with_phases(trace, 6, boundary, model)
    names = {r.policy for r in results}
    assert names == {"FIFO", "LRU", "Optimal", "Learned"}


def test_simulate_with_phases_without_model():
    trace, boundary = generate_shifted_trace(seed=11)
    results = simulate_with_phases(trace, 6, boundary, model=None)
    names = {r.policy for r in results}
    assert names == {"FIFO", "LRU", "Optimal"}


def test_simulate_phase_counts_sum_to_total():
    trace, boundary = generate_shifted_trace(seed=22)
    model = train_learned_model(trace, 6, boundary)
    results = simulate_with_phases(trace, 6, boundary, model)
    for r in results:
        phase_h = sum(r.phase_hits.values())
        phase_f = sum(r.phase_faults.values())
        assert phase_h == r.hits, f"Phase hits mismatch for {r.policy}"
        assert phase_f == r.faults, f"Phase faults mismatch for {r.policy}"


def test_simulate_result_ratios_in_range():
    trace, boundary = generate_shifted_trace(seed=33)
    model = train_learned_model(trace, 6, boundary)
    results = simulate_with_phases(trace, 6, boundary, model)
    for r in results:
        assert 0.0 <= r.hit_ratio <= 1.0, f"hit_ratio out of range for {r.policy}"
        assert 0.0 <= r.fault_ratio <= 1.0, f"fault_ratio out of range for {r.policy}"
        assert abs(r.hit_ratio + r.fault_ratio - 1.0) < 1e-9
