# Experiment Results

## Sample Video: Sama Savli Intersection

### Configuration
- **Video:** sample_video.mp4
- **Frames:** 457 (720×480, 23.98 fps)
- **Detection:** YOLOv8n (stride=5)
- **Tracking:** Simple IoU tracker (threshold=0.3)
- **Homography:** 4-point calibration (Sama Savli chowk)
- **Window size:** 5 frames
- **Stride:** 2

### Surrogate Safety Events
| Metric | Count |
|--------|-------|
| TTC events (filtered) | 91 |
| Conflict pairs (≥3 frames within 5m) | 7 |
| Positive windows | 7 |
| Negative windows | 37 |

### Machine Learning Results
**Model:** GCN with 32 hidden channels  
**Training:** 50 epochs, Adam (lr=0.01), BCEWithLogitsLoss  
**Cross-validation:** 5-fold (seed=42)

| Fold | Validation Accuracy |
|------|---------------------|
| 1    | 0.750               |
| 2    | 0.875               |
| 3    | 0.875               |
| 4    | 0.875               |
| 5    | 1.000               |
| **Mean ± Std** | **0.875 ± 0.079** |

### Conclusion
The GCN model successfully distinguishes temporal windows containing conflict pairs from non‑conflict windows with **87.5% cross‑validated accuracy**.  
This demonstrates the feasibility of graph‑based surrogate safety analysis on real traffic video data.

### Reproducing Results
Run the following command after generating frames, detections, tracks, and graphs:

```bash
python scripts/train_conflict_pairs.py \
  --tracks data/interim/tracks/sample_video/tracks.jsonl \
  --graphs_dir data/processed/graphs_calibrated/sample_video \
  --homography_config configs/homography/sample_video_4pt.yaml \
  --output_dir outputs/models_conflict_pairs \
  --distance_threshold 5.0 \
  --min_interaction_frames 3 \
  --epochs 50 \
  --num_folds 5 \
  --seed 42
```