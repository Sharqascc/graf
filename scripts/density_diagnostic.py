"""Density-stratified diagnostic for the window-level TTC label."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

import numpy as np

from graf.data.graph_dataset import SpatioTemporalWindowDataset
from graf.models.baselines import (
    GraphFeatureExtractor,
    RandomForestBaseline,
    SingleFeatureBaseline,
)
from graf.training.conflict_pairs import (
    add_world_coords,
    filter_tracks,
    load_tracks,
)
from scripts.evaluate_vntraffic import _auc
from scripts.label_strategies import apply_strategy


def per_window_density(tracks_df, window_ds):
    out = []
    for i in range(len(window_ds)):
        fids = window_ds[i].frame_ids.tolist()
        sub = tracks_df[tracks_df["frame_idx"].isin(fids)]
        pf = sub.groupby("frame_idx")["track_id"].nunique()
        out.append(float(pf.mean()) if len(pf) else 0.0)
    return np.asarray(out)


def random_folds(n, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = []
    size = n // k
    for i in range(k):
        s = i * size
        e = (i + 1) * size if i < k - 1 else n
        val = idx[s:e]
        folds.append((np.setdiff1d(idx, val), val))
    return folds


def evaluate_bin(X, y, k, seed):
    n, n_pos = len(y), int(y.sum())
    if n_pos < 2 or n_pos > n - 2 or n < 2 * k:
        return {"n": n, "n_pos": n_pos, "skipped": True}
    folds = random_folds(n, k, seed)

    def run(factory):
        aucs = []
        for tr, va in folds:
            if len(np.unique(y[tr])) < 2:
                continue
            m = factory()
            m.fit(X[tr], y[tr])
            p = (
                m.predict_proba(X[va])[:, 1]
                if hasattr(m, "predict_proba")
                else m.predict(X[va])
            )
            aucs.append(_auc(np.asarray(p), y[va]))
        return {"mean_auc": float(np.mean(aucs)) if aucs else None, "fold_aucs": aucs}

    return {
        "n": n,
        "n_pos": n_pos,
        "positive_rate": n_pos / n,
        "rf": run(RandomForestBaseline),
        "single_feature": run(
            lambda: SingleFeatureBaseline(feature_name="edge_attr_nonzero_frac")
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", required=True)
    p.add_argument("--graphs_dir", required=True)
    p.add_argument("--homography_config", default=None)
    p.add_argument("--pixels_per_meter", type=float, default=None)
    p.add_argument("--label_strategy", default="sustained")
    p.add_argument("--run_length", type=int, default=5)
    p.add_argument("--ttc_threshold_seconds", type=float, default=1.5)
    p.add_argument("--ttc_distance_threshold", type=float, default=3.0)
    p.add_argument("--ttc_dcpa_threshold", type=float, default=1.5)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="docs/research")
    a = p.parse_args(argv)

    tracks_df = filter_tracks(load_tracks(a.tracks))
    if a.homography_config:
        import yaml

        with Path(a.homography_config).open() as f:
            H = np.array(yaml.safe_load(f)["H"], dtype=np.float64)
    elif a.pixels_per_meter:
        ppm = a.pixels_per_meter
        H = np.array([[1 / ppm, 0, 0], [0, 1 / ppm, 0], [0, 0, 1]])
    else:
        raise SystemExit("need --homography_config or --pixels_per_meter")
    tracks_df = add_world_coords(tracks_df, H, fps=30.0)

    wds = SpatioTemporalWindowDataset(graph_dir=a.graphs_dir, window_size=5, stride=2)
    labels = apply_strategy(
        a.label_strategy,
        tracks_df,
        wds,
        ttc_threshold_seconds=a.ttc_threshold_seconds,
        distance_threshold=a.ttc_distance_threshold,
        dcpa_threshold=a.ttc_dcpa_threshold,
        run_length=a.run_length,
    )
    y = np.asarray(labels, dtype=int)

    dens = per_window_density(tracks_df, wds)
    q1, q2 = np.quantile(dens, [1 / 3, 2 / 3])
    tercile = np.where(dens <= q1, 0, np.where(dens <= q2, 1, 2))

    X = np.asarray(GraphFeatureExtractor().transform([wds[i] for i in range(len(wds))]))

    results = []
    for t, name in enumerate(["low", "medium", "high"]):
        m = tercile == t
        r = evaluate_bin(X[m], y[m], a.num_folds, a.seed)
        r["tercile"] = name
        results.append(r)

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "density_results.json").write_text(
        json.dumps(
            {
                "overall": {
                    "n_windows": len(y),
                    "n_positive": int(y.sum()),
                    "positive_rate": float(y.mean()),
                    "density_q1": float(q1),
                    "density_q2": float(q2),
                },
                "per_tercile": results,
            },
            indent=2,
        )
    )

    md = [
        "# Density diagnostic\n",
        f"Windows: {len(y)}, positive: {int(y.sum())} ({y.mean():.3f})",
        f"Density q1={q1:.2f}, q2={q2:.2f}\n",
        "| tercile | n | n_pos | pos_rate | RF AUC | single AUC |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.get("skipped"):
            md.append(f"| {r['tercile']} | {r['n']} | {r['n_pos']} | - | skip | skip |")
            continue
        rf = r["rf"]["mean_auc"]
        sf = r["single_feature"]["mean_auc"]
        rf_s = f"{rf:.3f}" if rf is not None else "n/a"
        sf_s = f"{sf:.3f}" if sf is not None else "n/a"
        md.append(
            f"| {r['tercile']} | {r['n']} | {r['n_pos']} | "
            f"{r['positive_rate']:.3f} | {rf_s} | {sf_s} |"
        )
    (out / "density_results.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
