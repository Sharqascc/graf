import pytest
from pathlib import Path

from graf.graph.builders import GraphBuilder, load_pair_radii, DEFAULT_INTERACTION_RADII


def test_load_pair_radii_from_yaml(tmp_path):
    yaml_content = """
pair_radii:
  - [car, pedestrian, 4.5]
  - [car, bus, 8.5]
"""
    config_file = tmp_path / "interaction.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")
    radii = load_pair_radii(config_file)
    assert radii[("car", "pedestrian")] == 4.5
    assert radii[("car", "bus")] == 8.5


def test_load_pair_radii_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pair_radii(tmp_path / "missing.yaml")


def test_load_pair_radii_invalid_entry(tmp_path):
    yaml_content = """
pair_radii:
  - [car, pedestrian]
"""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_pair_radii(config_file)


def test_graph_builder_uses_config_path(tmp_path):
    yaml_content = """
pair_radii:
  - [car, pedestrian, 10.0]
"""
    config_file = tmp_path / "interaction.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")
    builder = GraphBuilder(config_path=config_file)
    assert builder.pair_radii[("car", "pedestrian")] == 10.0
    # Other pairs not in config should not exist
    assert ("car", "car") not in builder.pair_radii


def test_graph_builder_fallback_default():
    builder = GraphBuilder()
    assert builder.pair_radii == DEFAULT_INTERACTION_RADII
