import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import pytest

# Homography
from graf.calibration.homography import _to_array, fit_homography, project_points, invert_homography, world_to_image
# Graph edges
from graf.graph.edges import build_edge_feature, reverse_edge_feature, safe_float, infer_actor_class
# Builders
from graf.graph.builders import GraphBuilder, compute_feature_stats, build_graph_for_frame, build_pyg_graph_for_frame, _legacy_group_by_frame, _legacy_trim_graph_edges
# Features
from graf.graph.features import extract_edge_distance, extract_edge_kinematics, extract_edge_angles, extract_edge_risk_features, validate_feature_layout, filter_subgraph_by_class, filter_subgraph_by_node_mask, _empty_like
# Nodes
from graf.graph.nodes import get_class_one_hot, get_spatial_positions, get_actor_class_names, clone_with_updated_node_feature, clone_with_updated_node_feature_block, validate_node_layout, node_feature_dim
# PyG export
from graf.graph.pyg_export import package_tensor_graph, load_graph_sample, save_graph_sample, to_pyg_dict, to_pyg_data
# Temporal
from graf.graph.temporal import build_temporal_window_graph
# Models
from graf.models.baselines import GraphFeatureExtractor, MajorityClassBaseline, LogisticRegressionBaseline, RandomForestBaseline, MLPBaseline, _to_numpy, _safe_stats
from graf.models.gcn_risk import build_model, has_torch_geometric
# SSM
from graf.ssm.pet import PETResult, compute_pet_from_conflict_zone, PETCalculator
from graf.ssm.ttc import TTCResult, compute_ttc_constant_velocity
# Utils
from graf.utils.io import write_json, write_jsonl, write_text, snapshot_environment
from graf.utils.seeds import set_global_seed
from graf.utils.pipeline_status import count_real_files, stage_status_icon

def test_homography_to_array_invalid_shape():
    with pytest.raises(ValueError):
        _to_array([[1,2,3]])

def test_fit_homography_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_homography([(0,0),(1,1),(2,2),(3,3)], [(0,0),(1,1)])

def test_fit_homography_insufficient_points():
    with pytest.raises(ValueError):
        fit_homography([(0,0),(1,1),(2,2)], [(0,0),(1,1),(2,2)])

def test_world_to_image_roundtrip():
    H = np.array([[1,0,2],[0,1,3],[0,0,1]], dtype=np.float64)
    world = [(1.0, 1.0), (2.0, 3.0)]
    img = world_to_image(H, world)
    back = project_points(H, [tuple(p) for p in img])
    assert np.allclose(back, world)

def test_reverse_edge_feature():
    feat = {'dx':1,'dy':2,'distance':3,'dvx':4,'dvy':5,'relative_speed':6,'closing_speed':7,'bearing_sin':8,'bearing_cos':9,'rel_heading_sin':10,'rel_heading_cos':11,'ttc':12,'same_class':0,'size_src':1.5,'size_dst':2.5}
    rev = reverse_edge_feature(feat)
    assert rev['dx'] == -1 and rev['dvx'] == -4 and rev['size_src'] == 2.5

def test_safe_float_none():
    assert safe_float(None) == 0.0

def test_infer_actor_class_fallback():
    assert infer_actor_class({'actor_class':'bus'}) == 'bus'
    assert infer_actor_class({}) == 'other'

def test_builder_find_candidate_pairs_no_kdtree():
    builder = GraphBuilder(use_kdtree=False)
    pos = np.array([[0,0],[1,0],[10,10]])
    pairs = builder._find_candidate_pairs(pos)
    assert len(pairs) == 3

def test_builder_max_search_radius_velocity_adaptive():
    builder = GraphBuilder(use_velocity_adaptive_radius=True, velocity_radius_scale=0.5)
    assert builder._max_search_radius() > 0

def test_builder_normalize_array_empty():
    arr = np.empty((0,3), dtype=np.float32)
    out = GraphBuilder._normalize_array(arr, np.array([0]), np.array([1]))
    assert out.size == 0

def test_compute_feature_stats_no_data():
    stats = compute_feature_stats([Data()])
    assert stats.node_mean is None

def test_build_graph_for_frame_empty():
    result = build_graph_for_frame([], radius=6.0)
    assert result['num_nodes'] == 0

def test_build_pyg_graph_for_frame():
    actors = [{'track_id':1,'actor_class':'car','x_m':0,'y_m':0,'vx':1,'vy':0}, {'track_id':2,'actor_class':'pedestrian','x_m':1,'y_m':0,'vx':0,'vy':0}]
    data = build_pyg_graph_for_frame(actors, radius=10.0, frame_id=1, video_id='v1')
    assert data.num_nodes == 2

def test_legacy_trim_graph_edges_fallback():
    graph = {'edges': [{'source':1,'target':1}]}
    out = _legacy_trim_graph_edges(graph)
    assert out['edges'] == [{'source':1,'target':1}]

def test_extract_edge_distance_empty():
    data = Data(edge_attr=torch.empty(0, 15))
    out = extract_edge_distance(data)
    assert out.shape == (0,1)

def test_extract_edge_kinematics_empty():
    data = Data(edge_attr=torch.empty(0, 15))
    out = extract_edge_kinematics(data)
    assert out.shape == (0,8)

def test_extract_edge_angles_empty():
    data = Data(edge_attr=torch.empty(0, 15))
    out = extract_edge_angles(data)
    assert out.shape == (0,4)

def test_extract_edge_risk_features_empty():
    data = Data(edge_attr=torch.empty(0, 15))
    out = extract_edge_risk_features(data)
    assert out.shape == (0,7)

def test_validate_feature_layout_missing_x():
    data = Data(edge_index=torch.empty((2,0), dtype=torch.long), edge_attr=torch.empty((0,15)))
    with pytest.raises(ValueError, match='Data.x is missing'):
        validate_feature_layout(data)

def test_get_class_one_hot_empty():
    data = Data(x=torch.empty((0, node_feature_dim())))
    one_hot = get_class_one_hot(data)
    from graf.graph.nodes import ACTOR_CLASSES
    assert one_hot.shape == (0, len(ACTOR_CLASSES))

def test_get_spatial_positions_fallback():
    data = Data(x=torch.rand(3, node_feature_dim()), pos=None)
    pos = get_spatial_positions(data)
    assert pos.shape == (3,2)

def test_clone_with_updated_node_feature_block_wrong_shape():
    data = Data(x=torch.rand(3, node_feature_dim()))
    with pytest.raises(ValueError):
        clone_with_updated_node_feature_block(data, ['x','y'], torch.ones(2,2))

def test_pyg_export_package_with_y():
    x = torch.rand(2,5); edge_index = torch.tensor([[0,1],[1,0]]); edge_attr = torch.rand(2,3)
    data = package_tensor_graph(x, edge_index, edge_attr, y=torch.tensor([0]))
    assert data.y is not None

def test_pyg_export_load_graph_sample_missing():
    with pytest.raises(FileNotFoundError):
        load_graph_sample('/tmp/nonexistent.pt')

def test_pyg_export_to_pyg_dict_empty_edges():
    payload = to_pyg_dict({'nodes': [], 'edges': []})
    assert payload['edge_index'] == [[], []]

def test_temporal_window_empty_frames_raises():
    with pytest.raises(ValueError):
        build_temporal_window_graph([])

def test_temporal_window_multi_video_raises():
    d1 = Data(x=torch.rand(2,5), edge_index=torch.empty((2,0), dtype=torch.long), edge_attr=torch.empty((0,15)), pos=torch.rand(2,2), track_ids=torch.tensor([0,1]), video_id='v1', frame_id=0)
    d2 = Data(x=torch.rand(2,5), edge_index=torch.empty((2,0), dtype=torch.long), edge_attr=torch.empty((0,15)), pos=torch.rand(2,2), track_ids=torch.tensor([0,1]), video_id='v2', frame_id=1)
    with pytest.raises(ValueError, match='same video_id'):
        build_temporal_window_graph([d1, d2])

def test_baseline_extractor_unsupported():
    with pytest.raises(TypeError):
        GraphFeatureExtractor.transform('invalid')

def test_baseline_to_numpy_tensor():
    tensor = torch.tensor([1.0,2.0])
    arr = _to_numpy(tensor)
    assert arr.shape == (2,)

def test_baseline_safe_stats_empty():
    assert _safe_stats(np.array([])) == [0.0,0.0,0.0,0.0]

def test_gcn_build_model_and_forward():
    if not has_torch_geometric:
        return
    data = Data(x=torch.rand(3,10), edge_index=torch.tensor([[0,1,2],[1,2,0]]), edge_attr=torch.rand(3,15))
    model = build_model(in_channels=10, hidden_channels=16)
    model.eval()
    with torch.no_grad():
        out = model(data)
    assert out.shape == (1,)

def test_pet_zone_overlap():
    traj1 = np.array([[0,0],[1,0]]); traj2 = np.array([[0,0],[1,0]])
    t1 = np.array([0,1]); t2 = np.array([0.5,1.5])
    res = compute_pet_from_conflict_zone(traj1, traj2, t1, t2, np.array([0.5,0]), zone_radius=1.0)
    assert res.status == 'zone_overlap'

def test_pet_one_never_enter():
    traj1 = np.array([[0,0]]); traj2 = np.array([[10,10]])
    res = compute_pet_from_conflict_zone(traj1, traj2, np.array([0]), np.array([0]), np.array([0,0]))
    assert res.status == 'one_or_both_never_enter_zone'

def test_ttc_no_collision_min_sep():
    res = compute_ttc_constant_velocity(np.array([0,0]), np.array([1,0]), np.array([1,10]), np.array([-1,0]))
    assert res.ttc_seconds == float('inf')

def test_io_snapshot_environment(tmp_path):
    snapshot_environment('.', tmp_path)
    assert (tmp_path/'git_commit.txt').exists()

def test_seed_set_global_seed():
    set_global_seed(42)
    first = np.random.rand()
    set_global_seed(42)
    second = np.random.rand()
    assert first == second

def test_pipeline_status_count_real_files(tmp_path):
    (tmp_path/'file.txt').write_text('x')
    assert count_real_files(tmp_path) == 1