"""Tests for the graf.tracking placeholder modules.

The tracking package contains documented stubs, not real trackers.
These tests verify that the modules import cleanly, that stub methods
raise NotImplementedError at call time, and that the protocol is
well-formed. When a real tracker is integrated, replace this file.
"""

from __future__ import annotations

import pytest

from graf.tracking import botsort, bytetrack, postprocess
from graf.tracking.base import Tracker


def test_graf_tracking_imports():
    import graf.tracking as tracking_pkg

    assert tracking_pkg.__all__ == []


def test_base_protocol_is_runtime_checkable():
    class Fake:
        def update(self, detections):
            return detections

        def reset(self):
            pass

    assert isinstance(Fake(), Tracker)


def test_base_protocol_rejects_incomplete_class():
    class Incomplete:
        pass

    assert not isinstance(Incomplete(), Tracker)


def test_bytetrack_stub_raises():
    with pytest.raises(NotImplementedError, match="ByteTrack is not implemented"):
        bytetrack.ByteTrackTracker()


def test_bytetrack_stub_accepts_arbitrary_args():
    with pytest.raises(NotImplementedError):
        bytetrack.ByteTrackTracker(model="x", thresh=0.5)
    with pytest.raises(NotImplementedError):
        bytetrack.ByteTrackTracker(1, 2, 3)


def test_botsort_stub_raises():
    with pytest.raises(NotImplementedError, match="BoT-SORT is not implemented"):
        botsort.BotSortTracker()


def test_postprocess_interpolate_raises():
    with pytest.raises(NotImplementedError, match="interpolation is not implemented"):
        postprocess.interpolate_tracks()


def test_postprocess_filter_raises():
    with pytest.raises(NotImplementedError, match="Short-track filtering"):
        postprocess.filter_short_tracks()
