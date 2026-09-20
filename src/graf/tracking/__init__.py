"""Tracking subpackage — placeholders for future tracker integrations.

The working tracker in this repository lives in ``scripts/run_tracking.py``
as a greedy IoU associator with motion-aware prediction. The modules in
this package are placeholders for future third-party tracker
integrations (ByteTrack, BoT-SORT) and are not imported by any code
today. They exist so that ``import graf.tracking`` is stable for a
future integration.

Nothing is exported yet.
"""

__all__: list[str] = []
