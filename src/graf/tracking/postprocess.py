"""Track post-processing — planned, not yet implemented.

Post-processing operations like interpolating gaps in short tracks,
filtering very short tracks, or smoothing trajectories are not yet
factored into reusable functions. This module is a placeholder for
those utilities when they are needed.
"""

from __future__ import annotations

__all__: list[str] = []


def interpolate_tracks(*args: object, **kwargs: object) -> object:
    """Placeholder — see module docstring."""
    raise NotImplementedError(
        "Track interpolation is not implemented. See module docstring."
    )


def filter_short_tracks(*args: object, **kwargs: object) -> object:
    """Placeholder — see module docstring."""
    raise NotImplementedError(
        "Short-track filtering is not implemented. See module docstring."
    )
