"""Tests for the event-mining placeholder module.

The event-mining module is documented as planned-but-unimplemented. These
tests verify two things:

  1. The module imports cleanly (no accidental syntax or import errors as
     the stub text evolves).
  2. Its public entry point raises ``NotImplementedError`` with a message
     that points a reader at the intended API.

When event mining is actually implemented, this file should be deleted and
replaced with real behavior tests. (DRAC was implemented in a prior change
and its tests now live in ``tests/test_ssm.py`` and
``tests/test_property_based.py``.)
"""

from __future__ import annotations

import pytest

from graf.ssm import event_mining as mining_mod


def test_event_mining_module_imports():
    assert mining_mod.__all__ == []


def test_mine_events_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Event mining is planned"):
        mining_mod.mine_events()


def test_mine_events_ignores_args():
    with pytest.raises(NotImplementedError):
        mining_mod.mine_events(None)
    with pytest.raises(NotImplementedError):
        mining_mod.mine_events(ssm_frame_values=None, thresholds={})
