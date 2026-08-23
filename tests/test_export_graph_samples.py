import json

from graf.utils.export_graph_samples import export_graph_samples


def test_export_graph_samples_creates_file(tmp_path):
    out_path = export_graph_samples(tmp_path)
    assert out_path.exists()
    assert out_path.suffix == ".json"
    assert out_path.name == "sample_graph_pyg.json"

def test_export_graph_samples_valid_json(tmp_path):
    out_path = export_graph_samples(tmp_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert len(payload) > 0

def test_export_graph_samples_reproducible_structure(tmp_path):
    out_path1 = export_graph_samples(tmp_path / "run1")
    out_path2 = export_graph_samples(tmp_path / "run2")
    payload1 = json.loads(out_path1.read_text(encoding="utf-8"))
    payload2 = json.loads(out_path2.read_text(encoding="utf-8"))
    assert payload1.keys() == payload2.keys()
