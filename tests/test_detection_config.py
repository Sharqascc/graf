"""Validate that detection configs use the GRAF actor vocabulary.

A detection config's ``class_map`` translates detector-native class
names (COCO, UVH-26) into GRAF actor classes. If a mapped value is not
in ``ACTOR_CLASSES``, the graph builder silently falls through to the
default interaction radius and the ``other`` size prior -- no error is
raised, and the resulting graph is subtly wrong. This test catches the
misconfiguration at CI time instead.

See also ``tests/test_builder_config.py`` for the pair-radius side of
the same configuration surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from graf.graph.edges import ACTOR_CLASSES

REPO = Path(__file__).resolve().parents[1]
DETECTION_CONFIGS = sorted((REPO / "configs" / "detection").glob("*.yaml"))


@pytest.mark.parametrize(
    "config_path", DETECTION_CONFIGS, ids=lambda p: p.name
)
def test_class_map_values_are_in_actor_classes(config_path: Path) -> None:
    """Every value in ``class_map`` must be a member of ACTOR_CLASSES."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    class_map = cfg.get("class_map")
    if class_map is None:
        pytest.skip(f"{config_path.name} does not define a class_map")

    unknown = {
        src: dst for src, dst in class_map.items() if dst not in ACTOR_CLASSES
    }
    assert not unknown, (
        f"{config_path.name}: class_map values not in ACTOR_CLASSES "
        f"({sorted(ACTOR_CLASSES)}): {unknown}"
    )


@pytest.mark.parametrize(
    "config_path", DETECTION_CONFIGS, ids=lambda p: p.name
)
def test_config_declares_weights(config_path: Path) -> None:
    """Every detection config must point at a model file or name."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    assert cfg.get("weights"), f"{config_path.name}: missing 'weights'"
