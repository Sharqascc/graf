from graf.graph.builders import build_graph_for_frame, build_interaction_graph
from graf.graph.features import graph_to_tensors

def sample_records():
    return [
        {"track_id": 1, "frame_id": 10, "actor_class": "car", "x": 0.0, "y": 0.0, "vx": 2.0, "vy": 0.0},
        {"track_id": 2, "frame_id": 10, "actor_class": "two_wheeler", "x": 3.0, "y": 4.0, "vx": 1.0, "vy": 0.0},
        {"track_id": 3, "frame_id": 10, "actor_class": "pedestrian", "x": 30.0, "y": 0.0, "vx": 0.5, "vy": 0.0},
        {"track_id": 1, "frame_id": 11, "actor_class": "car", "x": 1.0, "y": 0.0, "vx": 2.0, "vy": 0.0},
        {"track_id": 2, "frame_id": 11, "actor_class": "two_wheeler", "x": 4.0, "y": 4.0, "vx": 1.0, "vy": 0.0},
    ]

def test_build_graph_for_frame_radius_filter():
    graph = build_graph_for_frame(sample_records()[:3], radius=6.0, actor_classes=["car", "two_wheeler", "pedestrian"])
    assert graph["num_nodes"] == 3
    assert graph["num_edges"] == 1
    assert graph["edges"][0]["src"] == 1
    assert graph["edges"][0]["dst"] == 2

def test_build_interaction_graph_groups_frames():
    graphs = build_interaction_graph(sample_records(), radius=6.0)
    assert len(graphs) == 2
    assert graphs[0]["frame_id"] == 10
    assert graphs[1]["frame_id"] == 11

def test_graph_to_tensors_shapes():
    graph = build_graph_for_frame(sample_records()[:3], radius=6.0, actor_classes=["car", "two_wheeler", "pedestrian"])
    tensors = graph_to_tensors(graph)
    assert len(tensors["x"]) == 3
    assert len(tensors["edge_index"]) == 1
    assert len(tensors["edge_attr"]) == 1
    assert len(tensors["x"][0]) == 12
