"""BoT-SORT integration — planned, not yet implemented.

The working tracker in this repository is a greedy IoU associator with
motion-aware prediction, implemented in ``scripts/run_tracking.py``.
This module is a placeholder for a future integration of BoT-SORT
(camera-motion-compensated tracking).
"""

from __future__ import annotations

__all__: list[str] = []


class BotSortTracker:
    """Placeholder — see module docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "BoT-SORT is not implemented. The working tracker is the "
            "greedy IoU associator in scripts/run_tracking.py."
        )
