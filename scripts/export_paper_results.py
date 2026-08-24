import argparse
import sys
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.data.data as pyg_data
import torch_geometric.data.storage as pyg_storage
from torch_geometric.data import Data
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from graf.data.graph_dataset import SpatioTemporalWindowDataset
from graf.calibration.homography import project_points

torch.serialization.add_safe_globals([
    Data,
    pyg_data.DataEdgeAttr,
    pyg_data.DataTensorAttr,
    pyg_storage.GlobalStorage,
    pyg_storage.BaseStorage,
    pyg_storage.NodeStorage,
    pyg_storage.EdgeStorage,
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs_dir', required=True)
    parser.add_argument('--frames_dir', required=True)
    parser.add_argument('--homography_config', required=True)
    parser.add_argument('--output_dir', default='outputs/figures')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=2)
    args = parser.parse_args()

    with open(args.homography_config) as f:
        H = np.array(yaml.safe_load(f)['H'], dtype=np.float64)

    window_ds = SpatioTemporalWindowDataset(
        graph_dir=args.graphs_dir,
        window_size=args.window_size,
        stride=args.stride,
    )
    if len(window_ds) == 0:
        print('No windows found')
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: spatial graph on first frame
    first_frame = Path(args.frames_dir) / '000000.jpg'
    if first_frame.exists():
        img = cv2.imread(str(first_frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        graph_path = Path(args.graphs_dir) / 'graph_f000000.pt'
        if graph_path.exists():
            graph = torch.load(graph_path, map_location='cpu', weights_only=True)
            node_positions = graph.pos.numpy() * 20.0
            edge_index = graph.edge_index.numpy()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img)
            for src, dst in edge_index.T:
                x1, y1 = node_positions[src]
                x2, y2 = node_positions[dst]
                ax.plot([x1, x2], [y1, y2], 'g-', alpha=0.6, linewidth=0.8)
            ax.scatter(node_positions[:, 0], node_positions[:, 1], c='red', s=30, zorder=5)
            ax.set_title('First frame interaction graph')
            ax.axis('off')
            fig.savefig(out_dir / 'spatial_graph_frame0.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print('Saved spatial graph figure')

    # Figure 2: temporal window preview
    if len(window_ds) > 0:
        sample = window_ds[0]
        frames = sample.frame_ids.tolist()[:3]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, fid in zip(axes, frames):
            fp = Path(args.frames_dir) / f'{fid:06d}.jpg'
            if fp.exists():
                im = cv2.imread(str(fp))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                ax.imshow(im)
            mask = (sample.node_frame_id == fid)
            node_indices = mask.nonzero(as_tuple=True)[0].numpy()
            if len(node_indices):
                pos = sample.pos[node_indices].numpy() * 20.0
                ax.scatter(pos[:, 0], pos[:, 1], c='red', s=30, zorder=5)
            ax.set_title(f'Frame {fid}')
            ax.axis('off')
        fig.savefig(out_dir / 'temporal_window_preview.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('Saved temporal window preview')

if __name__ == '__main__':
    main()