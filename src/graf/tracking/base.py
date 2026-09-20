"""Abstract Tracker interface — planned, not yet implemented.

The working tracker in this repository (see ``scripts/run_tracking.py``)
is a greedy IoU associator with motion-aware prediction. It does not
subclass anything from this module. This file is a placeholder for a
future interface that third-party tracker integrations can share.

Intended API::

    class Tracker(Protocol):
        def update(self, detections: list[dict]) -> list[dict]: ...
        def reset(self) -> None: ...
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tracker(Protocol):
    """Minimal tracker protocol — see module docstring."""

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def reset(self) -> None: ...
