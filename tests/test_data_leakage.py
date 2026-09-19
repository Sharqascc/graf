"""Data leakage checks for windowed k-fold CV.

Catches the class of bug where a model appears to learn but is actually
memorising training data that leaked into the evaluation set — the
classic failure mode for sliding-window datasets with overlapping
windows.

We enforce two things:

  * blocked_folds produces no interior leakage (only windows immediately
    adjacent to a fold boundary can share frames with the other side).
  * random_folds *does* leak — recorded as a test so that if someone
    changes the splitter, they know which property to preserve.
"""

from __future__ import annotations

import numpy as np

from scripts.evaluate_vntraffic import blocked_folds, random_folds


def _leak(train_idx, val_idx, frames_per_window):
    train_frames = set()
    for i in train_idx:
        train_frames.update(frames_per_window[int(i)])
    return sum(
        1 for i in val_idx if any(f in train_frames for f in frames_per_window[int(i)])
    )


def _windows(n=40, window_size=5, stride=2):
    """Recreate the frame_ids sequence the dataset produces for one video."""
    return [list(range(i * stride, i * stride + window_size)) for i in range(n)]


def test_blocked_folds_no_interior_leakage():
    windows = _windows()
    folds = blocked_folds(windows, num_folds=4)
    for k, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        val_sorted = sorted(int(v) for v in val_idx)
        train_frames = set()
        for i in train_idx:
            train_frames.update(windows[int(i)])
        for pos, v in enumerate(val_sorted):
            if set(windows[v]) & train_frames:
                near_start = pos < 3
                near_end = pos >= len(val_sorted) - 3
                assert near_start or near_end, (
                    f"fold {k} window {v} at interior position {pos} leaked"
                )


def test_random_folds_do_leak():
    """Records the leakage property of random_folds — it is not safe for
    overlapping-window CV, which is why blocked_folds exists."""
    windows = _windows()
    folds = random_folds(len(windows), num_folds=4, seed=42)
    total_leak = 0
    for k, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        total_leak += _leak(train_idx, val_idx, windows)
    assert total_leak > 0, (
        "random_folds did not leak on synthetic overlapping windows — "
        "if this changes, the earlier finding needs revisiting"
    )


def test_blocked_folds_leak_less_than_random():
    windows = _windows()
    blocked = blocked_folds(windows, num_folds=4)
    random = random_folds(len(windows), num_folds=4, seed=42)

    def total(folds):
        n = 0
        for k, val_idx in enumerate(folds):
            train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
            n += _leak(train_idx, val_idx, windows)
        return n

    assert total(blocked) < total(random)


def test_all_windows_accounted_for_once():
    """Sanity: folds partition the window index space."""
    windows = _windows(n=40)
    folds = blocked_folds(windows, num_folds=5)
    all_idx = sorted(int(i) for f in folds for i in f)
    assert all_idx == list(range(len(windows)))


def test_blocked_folds_is_deterministic():
    windows = _windows(n=40)
    a = blocked_folds(windows, num_folds=5)
    b = blocked_folds(windows, num_folds=5)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa, fb)
