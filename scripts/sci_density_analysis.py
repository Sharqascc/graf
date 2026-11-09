"""End-to-end density analysis for bbox-only tracks (SCI-style)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

SIZE_PRIORS = {
    "car": (2.0, 4.5),
    "two_wheeler": (0.8, 2.0),
    "auto_rickshaw": (1.5, 2.5),
    "truck": (2.5, 9.0),
    "bus": (2.5, 10.0),
    "pedestrian": (0.6, 1.7),
    "bicycle": (0.7, 1.8),
    "other": (1.5, 2.5),
}
FPS = 30.0
MAX_SPEED_MPS = 40.0
WIN_FRAMES = 150
STRIDE_FRAMES = 2
LEAK_FREE_FEATURES = [
    "mean_speed",
    "std_speed",
    "max_speed",
    "frac_car",
    "frac_two_wheeler",
    "frac_auto_rickshaw",
    "frac_truck",
    "frac_bus",
]


def load_tracks(path):
    df = pd.read_json(path, lines=True)
    df[["x1", "y1", "x2", "y2"]] = pd.DataFrame(
        df["bbox_xyxy"].tolist(), index=df.index
    )
    df["w"] = df["x2"] - df["x1"]
    df["h"] = df["y2"] - df["y1"]
    df["cx_px"] = (df["x1"] + df["x2"]) / 2
    df["cy_px"] = (df["y1"] + df["y2"]) / 2
    return df


def estimate_pixels_per_meter(df):
    vals = []
    for cls, (w_m, h_m) in SIZE_PRIORS.items():
        sub = df[df["class_name"] == cls]
        if len(sub) < 10:
            continue
        vals.append(sub["w"].median() / w_m)
        vals.append(sub["h"].median() / h_m)
    if not vals:
        raise ValueError("cannot estimate scale from tracks")
    return float(np.median(vals))


def bbox_to_world(df, ppm):
    df = df.copy()
    df["x_m"] = df["cx_px"] / ppm
    df["y_m"] = df["cy_px"] / ppm
    return df


def add_velocities(df, fps=FPS):
    df = df.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)
    g = df.groupby("track_id")
    df["dframes"] = g["frame_idx"].diff()
    dx = g["x_m"].diff()
    dy = g["y_m"].diff()
    df["dx_m"] = dx.where(df["dframes"] == 1)
    df["dy_m"] = dy.where(df["dframes"] == 1)
    df["vx"] = df["dx_m"] * fps
    df["vy"] = df["dy_m"] * fps
    df["speed"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    return df


def filter_implausible(df, max_speed=MAX_SPEED_MPS):
    return df[df["speed"].fillna(0) <= max_speed].copy()


def scan_frame(frame_df, ttc_thresh, dist_thresh, closing_thresh, dcpa_thresh):
    recs = frame_df[["x_m", "y_m", "vx", "vy", "track_id"]].dropna().to_numpy()
    crit = []
    best_ttc = float("inf")
    n_pairs = 0
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            rp = b[:2] - a[:2]
            rv = b[2:4] - a[2:4]
            if np.linalg.norm(rp) >= dist_thresh:
                continue
            n_pairs += 1
            cl = -float(np.dot(rp, rv))
            rs = float(np.dot(rv, rv))
            if rs <= 1e-9 or cl <= closing_thresh:
                continue
            ttc = cl / rs
            if not (0 < ttc <= ttc_thresh):
                continue
            if np.linalg.norm(rp + ttc * rv) >= dcpa_thresh:
                continue
            crit.append((int(a[4]), int(b[4])))
            best_ttc = min(best_ttc, ttc)
    return crit, best_ttc, n_pairs


def scan_window(frame_ids, df):
    out = {}
    for fid in frame_ids:
        sub = df[df["frame_idx"] == fid]
        out[int(fid)] = scan_frame(sub, 1.5, 3.0, 0.5, 1.5)
    return out


def label_variants(per_frame):
    any_crit = int(any(crit for crit, _, _ in per_frame.values()))
    sus = {}
    for rl in (5, 30):
        best = cur = 0
        for fid in sorted(per_frame):
            crit, _, _ = per_frame[fid]
            cur = cur + 1 if crit else 0
            best = max(best, cur)
        sus[rl] = int(best >= rl)
    gm = min((bt for _, bt, _ in per_frame.values()), default=float("inf"))
    mttc = {th: int(gm <= th) for th in (0.5, 1.0)}
    bf = 0.0
    for crit, _, np_ in per_frame.values():
        if np_ >= 1:
            bf = max(bf, len(crit) / np_)
    pf = {th: int(bf >= th) for th in (0.05, 0.10)}
    pr = 0
    run = {}
    for fid in sorted(per_frame):
        crit, _, _ = per_frame[fid]
        pairs = set(crit)
        for p in list(run):
            if p not in pairs:
                del run[p]
        for p in pairs:
            run[p] = run.get(p, 0) + 1
            pr = max(pr, run[p])
    return {
        "any_1.5": any_crit,
        "sustained_r5": sus[5],
        "sustained_r30": sus[30],
        "min_ttc_0.5": mttc[0.5],
        "min_ttc_1.0": mttc[1.0],
        "pair_frac_0.05": pf[0.05],
        "pair_frac_0.10": pf[0.10],
        "persistent_r3": int(pr >= 3),
    }


def window_features(frame_ids, df):
    sub = df[df["frame_idx"].isin(frame_ids)]
    if len(sub) < 10:
        return None
    pf = sub.groupby("frame_idx")["track_id"].nunique()
    sp = sub["speed"].dropna()
    total = len(sub)
    return {
        "mean_density": float(pf.mean()),
        "std_density": float(pf.std()) if len(pf) > 1 else 0.0,
        "mean_speed": float(sp.mean()) if len(sp) else 0.0,
        "std_speed": float(sp.std()) if len(sp) > 1 else 0.0,
        "max_speed": float(sp.max()) if len(sp) else 0.0,
        "frac_car": float((sub["class_name"] == "car").sum() / total),
        "frac_two_wheeler": float((sub["class_name"] == "two_wheeler").sum() / total),
        "frac_auto_rickshaw": float(
            (sub["class_name"] == "auto_rickshaw").sum() / total
        ),
        "frac_truck": float((sub["class_name"] == "truck").sum() / total),
        "frac_bus": float((sub["class_name"] == "bus").sum() / total),
    }


def blocked_cv_auc(X, y, n_folds=5, purge=75, seed=42):
    n = len(y)
    if n < 50 or y.sum() < 10 or y.sum() > n - 10:
        return None
    fold_size = n // n_folds
    aucs = []
    for k in range(n_folds):
        lo = k * fold_size
        hi = (k + 1) * fold_size if k < n_folds - 1 else n
        tr = np.ones(n, dtype=bool)
        tr[max(0, lo - purge) : min(n, hi + purge)] = False
        va = np.zeros(n, dtype=bool)
        va[lo:hi] = True
        if tr.sum() < 20 or va.sum() < 20:
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        m = RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=seed, n_jobs=-1
        )
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
    return float(np.mean(aucs)) if aucs else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--pixels_per_meter", type=float, default=None)
    ap.add_argument("--output_dir", default="outputs/sci_density")
    ap.add_argument("--win_frames", type=int, default=WIN_FRAMES)
    ap.add_argument("--stride_frames", type=int, default=STRIDE_FRAMES)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    df = load_tracks(a.tracks)
    print(f"loaded: {len(df)} rows, {df['track_id'].nunique()} tracks")

    if a.pixels_per_meter is None:
        ppm = estimate_pixels_per_meter(df)
        print(f"estimated scale: {ppm:.2f} px/m")
    else:
        ppm = a.pixels_per_meter
        print(f"scale override: {ppm:.2f} px/m")

    df = bbox_to_world(df, ppm)
    df = add_velocities(df)
    df = filter_implausible(df)
    print(f"after filter: {len(df)} rows")

    fmin = df["frame_idx"].min()
    fmax = df["frame_idx"].max()
    starts = list(range(fmin, fmax - a.win_frames + 1, a.stride_frames))
    print(f"windows: {len(starts)} (win={a.win_frames}, stride={a.stride_frames})")

    rows = []
    for w in starts:
        fids = list(range(w, w + a.win_frames))
        feat = window_features(fids, df)
        if feat is None:
            continue
        pfr = scan_window(fids, df)
        feat["window_start"] = w
        feat.update(label_variants(pfr))
        rows.append(feat)

    W = pd.DataFrame(rows)
    print(f"computed: {len(W)} windows")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    W.to_csv(out / "window_features.csv", index=False)
    print(f"saved: {out / 'window_features.csv'}")

    label_cols = [
        c
        for c in W.columns
        if c not in ("window_start", "mean_density", "std_density")
        and c not in LEAK_FREE_FEATURES
    ]
    print("")
    print("=== label summary ===")
    print(f"{'variant':<20} {'positives':>10} {'rate':>8}")
    for c in sorted(label_cols):
        rate = W[c].mean()
        print(f"{c:<20} {W[c].sum():>10} {rate:>8.3f}")

    try:
        W["tercile"] = pd.qcut(
            W["mean_density"], 3, labels=["low", "med", "high"], duplicates="drop"
        )
        tercile_ok = True
    except (ValueError, IndexError):
        tercile_ok = False
        print("")
        print("(tercile analysis skipped: insufficient density variation)")

    if tercile_ok:
        print("")
        print("=== within-tercile RF (leak-free features) ===")

    for c in sorted(label_cols) if tercile_ok else []:
        y = W[c].to_numpy()
        if y.sum() < 10 or y.sum() > len(y) - 10:
            print(f"  {c}: skipped (degenerate {int(y.sum())}/{len(y)})")
            continue
        within = []
        for t in ("low", "med", "high"):
            sub = W[W["tercile"] == t]
            if len(sub) < 30:
                continue
            yy = sub[c].to_numpy()
            if yy.sum() < 5 or yy.sum() > len(yy) - 5:
                continue
            auc = blocked_cv_auc(
                sub[LEAK_FREE_FEATURES].to_numpy(), yy, purge=75, seed=a.seed
            )
            if auc is not None:
                within.append(auc)
        dens = blocked_cv_auc(W[["mean_density"]].to_numpy(), y, purge=75, seed=a.seed)
        avg_s = f"{np.mean(within):.3f}" if within else "n/a"
        dd_s = f"{dens:.3f}" if dens is not None else "n/a"
        print(f"  {c}: within={avg_s}, density-only={dd_s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
