"""Event mining from SSM outputs — planned, not yet implemented.

Event mining is the step that turns per-frame SSM values (TTC, PET, DRAC)
into discrete, mappable safety events: contiguous windows where a given
metric crosses its critical threshold, attributed to a pair of tracks and
a video. The output is expected to be the ``SSMEventRecord`` schema already
defined in :mod:`graf.data.schema`.

The eventual public API will look something like:

    def mine_events(
        ssm_frame_values: pd.DataFrame,   # columns: video_id, frame_idx,
                                          #          track_id_a, track_id_b,
                                          #          metric_name, value
        thresholds: dict[str, float],     # e.g. {"TTC": 1.5, "PET": 2.0}
        min_duration_frames: int = 2,
    ) -> list[SSMEventRecord]: ...

This module is a placeholder kept so the README's SSM feature list has a
documented landing spot and the module path is stable.
"""

from __future__ import annotations

__all__: list[str] = []


def mine_events(*args: object, **kwargs: object) -> object:
    """Placeholder — see module docstring for the planned API."""
    raise NotImplementedError(
        "Event mining is planned but not implemented. "
        "See src/graf/ssm/event_mining.py for the intended API."
    )
