from graf.graph.builders import build_graph_for_frame
from graf.graph.pyg_export import to_pyg_dict


def sample_graph():
    records = [
        {"track_id": 10, "frame_id": 7, "actor_class": "car", "x": 0.0, "y": 0.0, "vx": 2.0, "vy": 0.0},
        {"track_id": 25, "frame_id": 7, "actor_class": "two_wheeler", "x": 3.0, "y": 4.0, "vx": 1.0, "vy": 0.5},
        {"track_id": 99, "frame_id": 7, "actor_class": "pedestrian", "x": 20.0, "y": 0.0, "vx": 0.2, "vy": 0.0},
    ]
    return build_graph_for_frame(records, radius=6.0, actor_classes=["car", "two_wheeler", "pedestrian"])


def test_to_pyg_dict_shapes_and_metadata():
    payload = to_pyg_dict(sample_graph())
    assert payload["num_nodes"] == 3
    assert payload["frame_id"] == 7
    assert payload["track_ids"] == [10, 25, 99]
    assert len(payload["x"]) == 3
    assert len(payload["x"][0]) == 12
    assert len(payload["edge_index"]) == 2
    assert len(payload["edge_index"][0]) == 1
    assert len(payload["edge_attr"]) == 1


def test_to_pyg_dict_remaps_edge_indices():
    payload = to_pyg_dict(sample_graph())
    assert payload["edge_index"] == [[0], [1]]
