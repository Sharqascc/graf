"""Tests for the DRAC and event-mining placeholder modules.

These modules are documented as planned-but-unimplemented. The tests here
verify two things:

  1. The modules import cleanly (no accidental syntax or import errors as
     the stub text evolves).
  2. Their public entry points raise ``NotImplementedError`` with a
     message that points a reader at the intended API.

When DRAC or event mining are actually implemented, these tests should be
deleted and replaced with real behavior tests.
"""

from __future__ import annotations

import pytest

from graf.ssm import drac as drac_mod
from graf.ssm import event_mining as mining_mod

# ------------------------------------------------------------------
# drac
# ------------------------------------------------------------------


def test_drac_module_imports():
    assert drac_mod.__all__ == []


def test_drac_result_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="DRAC is planned"):
        drac_mod.DRACResult(0.0, False, "uncomputed")


def test_compute_drac_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="DRAC is planned"):
        drac_mod.compute_drac_constant_velocity()


def test_compute_drac_ignores_positional_and_keyword_args():
    """Stub must raise regardless of what callers pass."""
    with pytest.raises(NotImplementedError):
        drac_mod.compute_drac_constant_velocity([0, 0], [1, 0], [10, 0], [-1, 0])
    with pytest.raises(NotImplementedError):
        drac_mod.compute_drac_constant_velocity(
            pos1=[0, 0], vel1=[1, 0], pos2=[10, 0], vel2=[-1, 0]
        )


# ------------------------------------------------------------------
# event_mining
# ------------------------------------------------------------------


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
