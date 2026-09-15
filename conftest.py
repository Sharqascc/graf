"""Shared pytest configuration and Hypothesis profiles."""
from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("dev", max_examples=200, deadline=None)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
