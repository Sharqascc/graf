"""Compare the GCN risk model against tabular baselines on identical folds.

Runs 5-fold cross-validation with the same label definition, window
construction, and fold assignment across a menu of models:

    majority   majority-class predictor (lower bound)
    logreg     logistic regression on GraphFeatureExtractor features
    rf         random forest on the same features
    mlp        scikit-learn MLP on the same features
    gcn        the GNN risk model used elsewhere in the pipeline

Reports per-fold and pooled accuracy / F1 / AUC for each model, plus
the majority baseline rate as a reference. Output is JSON plus a
markdown-style summary table suitable for the paper's results section.

The point is not to make any one model look good; it is to answer the
research question: does the GNN architecture add anything over a
classical classifier given the same input features?

Reuses build_labels / blocked_folds / random_folds / train_one_fold /
_auc / _f1 from scripts.evaluate_vntraffic so the GCN path here is the
same one used in evaluate_vntraffic.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Make both `graf.*` and `scripts.*` importable when this file is
# run as a subprocess (CI, pipeline_status.py). pytest.ini's
# pythonpath setting does not affect subprocesses.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import yaml

from graf.models.baselines import (
    GraphFeatureExtractor,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
    MLPBaseline,
    RandomForestBaseline,
)
from graf.training.conflict_pairs import (
    add_world_coords,
    filter_tracks,
    load_tracks,
)
from scripts.evaluate_vntraffic import (
    _auc,
    _f1,
    blocked_folds,
    build_labels,
    random_folds,
    train_one_fold,
)
from scripts.label_strategies import STRATEGIES, apply_strategy

ALL_MODELS = ("majority", "logreg", "rf", "mlp", "gcn")


def _make_model(name: str, seed: int) -> Any:
    if name == "majority":
        return MajorityClassBaseline()
    if name == "logreg":
        return LogisticRegressionBaseline(random_state=seed)
    if name == "rf":
        return RandomForestBaseline(random_state=seed)
    if name == "mlp":
        return MLPBaseline(random_state=seed)
    raise ValueError(f"unknown tabular model: {name!r}")


def _features_for_windows(window_ds, indices) -> np.ndarray:
    """Flatten a list of PyG window Data objects to a (N, D) matrix."""
    rows: list[np.ndarray] = []
    for i in indices:
        v = GraphFeatureExtractor.transform(window_ds[int(i)])
        if v.ndim == 2 and v.shape[0] == 1:
            rows.append(v[0])
        else:
            rows.append(v.reshape(-1))
    return np.asarray(rows, dtype=np.float32)


def train_tabular_fold(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one sklearn-compatible baseline, return (val_scores, val_labels)."""
    model = _make_model(name, seed)
    model.fit(X_train, y_train)

    # sklearn models return (N, 2) proba with column 1 = positive class
    try:
        proba = model.predict_proba(X_val)
        if proba.ndim == 2 and proba.shape[1] == 2:
            scores = proba[:, 1]
        elif proba.ndim == 2 and proba.shape[1] == 1:
            scores = proba[:, 0]
        else:
            scores = proba.reshape(-1)
    except Exception:
        # Fallback: hard predictions as 0/1 scores
        scores = np.asarray(model.predict(X_val), dtype=np.float32)

    return np.asarray(scores, dtype=np.float64), np.asarray(y_train)  # placeholder


def _negative_count(labels) -> int:
    """Count negatives (label == 0) in a val fold's label set."""
    arr = np.asarray(labels)
    return int((arr == 0).sum())


def _fold_auc_summary(fold_auc: list[float]) -> dict:
    """Summarise per-fold AUCs against a null of 0.5.

    Reports mean, std, and a one-sample two-sided Wilcoxon signed-rank
    p-value against the majority-AUC null (0.5). With n=5 folds the
    minimum achievable p is ~0.0625, so a non-significant result here
    is a limit of the fold count, not evidence of no signal.
    """
    valid = [a for a in fold_auc if a == a]
    n = len(valid)
    if n == 0:
        return {
            "n_valid_folds": 0,
            "mean_auc": float("nan"),
            "std_auc": float("nan"),
            "wilcoxon_p_vs_0_5": None,
        }
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    p_val = None
    if n >= 5 and any(a != 0.5 for a in valid):
        try:
            from scipy.stats import wilcoxon

            _, p = wilcoxon(np.asarray(valid) - 0.5, alternative="two-sided")
            p_val = float(p)
        except Exception:
            p_val = None
    return {
        "n_valid_folds": n,
        "mean_auc": mean,
        "std_auc": std,
        "wilcoxon_p_vs_0_5": p_val,
    }


def evaluate_model(
    name: str,
    window_ds,
    labels: list[int],
    folds: list[np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight: float | None,
    seed: int,
) -> dict[str, Any]:
    """Train and evaluate one model across all folds. Returns metrics + per-fold arrays."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    fold_acc, fold_f1, fold_auc, fold_maj = [], [], [], []
    fold_neg_count: list[int] = []
    pooled_scores: list[float] = []
    pooled_labels: list[int] = []

    if name == "gcn":
        for k, val_idx in enumerate(folds):
            train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
            scores, val_labels = train_one_fold(
                window_ds,
                labels,
                train_idx,
                val_idx,
                epochs,
                batch_size,
                lr,
                pos_weight,
                seed + k,
            )
            preds = (scores > 0.5).astype(int)
            fold_acc.append(float((preds == val_labels).mean()))
            fold_f1.append(_f1(preds, val_labels))
            fold_auc.append(_auc(scores, val_labels))
            fold_maj.append(float(max(val_labels.mean(), 1 - val_labels.mean())))
            fold_neg_count.append(_negative_count(val_labels))
            pooled_scores.extend(scores.tolist())
            pooled_labels.extend(val_labels.tolist())
    else:
        # Tabular path: flatten features once
        all_features = _features_for_windows(window_ds, list(range(len(window_ds))))
        for k, val_idx in enumerate(folds):
            train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
            X_train = all_features[train_idx]
            y_train = labels_arr[train_idx]
            X_val = all_features[val_idx]
            y_val = labels_arr[val_idx]

            model = _make_model(name, seed + k)
            model.fit(X_train, y_train)
            try:
                proba = model.predict_proba(X_val)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    scores = proba[:, 1]
                else:
                    scores = proba.reshape(-1)
            except Exception:
                scores = np.asarray(model.predict(X_val), dtype=np.float64)
            scores = np.asarray(scores, dtype=np.float64)

            preds = (scores > 0.5).astype(int)
            fold_acc.append(float((preds == y_val).mean()))
            fold_f1.append(_f1(preds, y_val))
            fold_auc.append(_auc(scores, y_val))
            fold_maj.append(float(max(y_val.mean(), 1 - y_val.mean())))
            fold_neg_count.append(_negative_count(y_val))
            pooled_scores.extend(scores.tolist())
            pooled_labels.extend(y_val.tolist())

    pooled_scores_arr = np.asarray(pooled_scores)
    pooled_labels_arr = np.asarray(pooled_labels)
    pooled_preds = (pooled_scores_arr > 0.5).astype(int)
    pooled_acc = float((pooled_preds == pooled_labels_arr).mean())

    auc_stats = _fold_auc_summary(fold_auc)

    return {
        "model": name,
        "fold_acc": fold_acc,
        "fold_f1": fold_f1,
        "fold_auc": fold_auc,
        "fold_majority": fold_maj,
        "fold_negative_count": fold_neg_count,
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "mean_f1": float(np.mean(fold_f1)),
        "mean_auc": auc_stats["mean_auc"],
        "std_auc": auc_stats["std_auc"],
        "n_valid_folds_auc": auc_stats["n_valid_folds"],
        "wilcoxon_p_vs_0_5": auc_stats["wilcoxon_p_vs_0_5"],
        "pooled_accuracy": pooled_acc,
        # Retained only for debugging. Pooling scores across folds trained
        # on different data is not a valid AUC: each model produces
        # scores on its own calibration scale. Do not report this.
        "pooled_auc_DEBUG_DO_NOT_REPORT": float(
            _auc(pooled_scores_arr, pooled_labels_arr)
        ),
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tracks", required=True)
    p.add_argument("--graphs_dir", required=True)
    p.add_argument("--homography_config", required=True)
    p.add_argument("--output_dir", default="outputs/baseline_comparison")
    p.add_argument("--window_size", type=int, default=5)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--distance_threshold", type=float, default=5.0)
    p.add_argument("--min_interaction_frames", type=int, default=3)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--pos_weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["blocked", "random"], default="blocked")
    p.add_argument("--label-source", choices=["ttc", "proximity"], default="ttc")
    p.add_argument("--ttc-threshold-seconds", type=float, default=1.5)
    p.add_argument("--ttc-distance-threshold", type=float, default=3.0)
    p.add_argument(
        "--label-strategy",
        choices=list(STRATEGIES),
        default="any",
        help=(
            "Window-level label rule. 'any' is the declared-setup default; "
            "the others target the scale-saturation problem documented in "
            "docs/paper/baselines_vntraffic_all81.md."
        ),
    )
    p.add_argument(
        "--fraction-threshold",
        type=float,
        default=0.2,
        help="pair_fraction strategy: minimum critical-pair fraction.",
    )
    p.add_argument(
        "--run-length",
        type=int,
        default=3,
        help="sustained strategy: minimum consecutive-frame run.",
    )
    p.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help="Comma-separated subset of: " + ", ".join(ALL_MODELS),
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in requested:
        if m not in ALL_MODELS:
            raise SystemExit(f"unknown model: {m!r} (choose from {ALL_MODELS})")

    window_ds, labels = build_labels(
        args.tracks,
        args.graphs_dir,
        args.homography_config,
        args.window_size,
        args.stride,
        args.distance_threshold,
        args.min_interaction_frames,
        args.fps,
        label_source=args.label_source,
        ttc_threshold_seconds=args.ttc_threshold_seconds,
        ttc_distance_threshold=args.ttc_distance_threshold,
    )

    # Non-default label strategies recompute labels from the raw tracks.
    if args.label_strategy != "any":
        print(f"Recomputing labels with strategy={args.label_strategy}")
        tracks_df = filter_tracks(load_tracks(args.tracks))
        with open(args.homography_config) as _f:
            H = np.array(yaml.safe_load(_f)["H"], dtype=np.float64)
        tracks_df = add_world_coords(tracks_df, H, fps=args.fps)
        labels = apply_strategy(
            args.label_strategy,
            tracks_df,
            window_ds,
            ttc_threshold_seconds=args.ttc_threshold_seconds,
            distance_threshold=args.ttc_distance_threshold,
            fraction_threshold=args.fraction_threshold,
            run_length=args.run_length,
        )
    n = len(window_ds)
    if n == 0:
        raise SystemExit("No windows found — check graphs_dir.")
    n_pos = int(sum(labels))
    majority = max(n_pos, n - n_pos) / n
    print(f"Windows: {n}  positives: {n_pos}  majority: {majority:.3f}")

    # A single-class label set makes every model trivially 100% or 0%;
    # sklearn also raises inside logistic-regression fit. Report the
    # degenerate distribution and exit cleanly rather than crashing.
    if n_pos == 0 or n_pos == n:
        print(
            f"\nLabel set is degenerate: {n_pos}/{n} positives. "
            "No model can be trained or evaluated. This is itself a "
            "finding worth recording — widen the threshold, tighten "
            "the window rule, or add negatives before rerunning."
        )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        degenerate_payload = {
            "setup": {
                "tracks": args.tracks,
                "graphs_dir": args.graphs_dir,
                "num_windows": n,
                "label_source": args.label_source,
                "label_strategy": args.label_strategy,
                "ttc_threshold_seconds": args.ttc_threshold_seconds,
                "ttc_distance_threshold": args.ttc_distance_threshold,
                "split": args.split,
            },
            "labels": {
                "num_windows": n,
                "num_positive": n_pos,
                "num_negative": n - n_pos,
                "majority_baseline": majority,
            },
            "results": [],
            "degenerate": True,
        }
        (out_dir / "comparison.json").write_text(
            json.dumps(degenerate_payload, indent=2) + "\n"
        )
        print(f"Wrote {out_dir / 'comparison.json'}")
        return 0

    frame_ids_per_window = [window_ds[i].frame_ids.tolist() for i in range(n)]
    if args.split == "blocked":
        folds = blocked_folds(frame_ids_per_window, args.num_folds)
    else:
        folds = random_folds(n, args.num_folds, args.seed)

    results = []
    for name in requested:
        t0 = time.time()
        r = evaluate_model(
            name,
            window_ds,
            labels,
            folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            pos_weight=args.pos_weight,
            seed=args.seed,
        )
        r["runtime_seconds"] = round(time.time() - t0, 2)
        results.append(r)
        print(
            f"  {name:10s}  "
            f"acc={r['mean_accuracy']:.3f}±{r['std_accuracy']:.3f}  "
            f"f1={r['mean_f1']:.3f}  "
            f"auc={r['mean_auc']:.3f}±{r['std_auc']:.3f}  "
            f"neg/fold={r['fold_negative_count']}  "
            f"({r['runtime_seconds']}s)"
        )

    # Markdown table
    print()
    print(
        "| model | mean acc ± std | mean F1 | mean AUC ± std "
        "| neg/fold | pooled acc | runtime (s) |"
    )
    print("|---|---|---|---|---|---|---|")
    for r in results:
        neg_per_fold = "/".join(str(n) for n in r["fold_negative_count"])
        print(
            f"| {r['model']} | "
            f"{r['mean_accuracy']:.3f} ± {r['std_accuracy']:.3f} | "
            f"{r['mean_f1']:.3f} | "
            f"{r['mean_auc']:.3f} ± {r['std_auc']:.3f} | "
            f"{neg_per_fold} | "
            f"{r['pooled_accuracy']:.3f} | "
            f"{r['runtime_seconds']} |"
        )
    print()
    print("Note on AUC: pooled AUC is intentionally not reported. With")
    print("n=5 folds the minimum achievable Wilcoxon p vs AUC=0.5 is")
    print("~0.0625, so a non-significant result here is a limit of the")
    print("fold count, not evidence of no signal. Per-fold negatives are")
    print("shown so a reader can see the label distribution behind the")
    print("variance.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "setup": {
            "tracks": args.tracks,
            "graphs_dir": args.graphs_dir,
            "window_size": args.window_size,
            "stride": args.stride,
            "label_source": args.label_source,
            "ttc_threshold_seconds": args.ttc_threshold_seconds,
            "ttc_distance_threshold": args.ttc_distance_threshold,
            "split": args.split,
            "num_folds": args.num_folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "pos_weight": args.pos_weight,
            "seed": args.seed,
        },
        "labels": {
            "num_windows": n,
            "num_positive": n_pos,
            "num_negative": n - n_pos,
            "majority_baseline": majority,
        },
        "results": results,
    }
    (out / "comparison.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
