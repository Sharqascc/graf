"""Shared pytest configuration and Hypothesis profiles."""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

# Three profiles, by cost:
#
#   dev      local iteration (default) — fast enough not to interrupt work
#   ci       PR gate — deep enough to catch regressions without slowing merges
#   nightly  scheduled deep search — maximum exploration, ~20-30 min
#
# Deliberately NOT setting derandomize=True. Hypothesis's default is to
# seed fresh each run and print the failing seed for reproduction; that
# is what makes a nightly sweep find new failures over time. A fixed
# seed tests the same inputs every night and finds fewer bugs as the
# code ages.

settings.register_profile(
    "ci",
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("dev", max_examples=200, deadline=None)
settings.register_profile(
    "nightly",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
