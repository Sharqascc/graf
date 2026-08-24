
import argparse
import yaml
import numpy as np
import cv2
from pathlib import Path


def latlon_to_local(latlon_points, origin_latlon=None):
    R = 6371000.0
    if origin_latlon is None:
        origin_latlon = latlon_points[0]
    lat0, lon0 = origin_latlon
    lat0_rad = np.deg2rad(lat0)
    world = []
    for lat, lon in latlon_points:
        y = (lat - lat0) * (np.pi / 180.0) * R
        x = (lon - lon0) * (np.pi / 180.0) * R * np.cos(lat0_rad)
        world.append([x, y])
    return np.array(world, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Estimate homography using RANSAC from calibration points")
    parser.add_argument("--input_config", required=True, help="YAML with calibration_points_pixel and calibration_points_world or calibration_points_latlon")
    parser.add_argument("--output_config", default=None, help="Output YAML (default: overwrite input)")
    parser.add_argument("--ransac_threshold", type=float, default=1.0, help="RANSAC reprojection threshold in pixels? or world units?")
    args = parser.parse_args()

    in_path = Path(args.input_config)
    with open(in_path) as f:
        cfg = yaml.safe_load(f)

    if "calibration_points_pixel" not in cfg or "calibration_points_world" not in cfg:
        raise ValueError("Config must contain calibration_points_pixel and calibration_points_world")

    pixel = np.array(cfg["calibration_points_pixel"], dtype=np.float64)
    world = np.array(cfg["calibration_points_world"], dtype=np.float64)

    # Use RANSAC to find robust homography
    H, mask = cv2.findHomography(
        pixel,
        world,
        method=cv2.RANSAC,
        ransacReprojThreshold=args.ransac_threshold,
        maxIters=2000,
        confidence=0.995,
    )
    if H is None:
        raise RuntimeError("RANSAC homography estimation failed")

    # Compute reprojection error for inliers
    reproj_world = cv2.perspectiveTransform(pixel.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(reproj_world - world, axis=1)
    inlier_mask = mask.ravel().astype(bool)
    inlier_errors = errors[inlier_mask] if np.any(inlier_mask) else errors
    mean_inlier_error = float(np.mean(inlier_errors))
    max_inlier_error = float(np.max(inlier_errors)) if len(inlier_errors) else 0.0

    print(f"RANSAC results:")
    print(f"  Inliers: {inlier_mask.sum()}/{len(pixel)}")
    print(f"  Mean inlier reprojection error: {mean_inlier_error:.3f} m")
    print(f"  Max inlier reprojection error: {max_inlier_error:.3f} m")
    print(f"  H matrix:\n{H}")

    # Update config
    out_cfg = dict(cfg)
    out_cfg["H"] = H.tolist()
    out_cfg["ransac_reproj_threshold"] = args.ransac_threshold
    out_cfg["calibration_inlier_mask"] = inlier_mask.astype(int).tolist()
    out_cfg["mean_inlier_error_m"] = mean_inlier_error
    out_cfg["max_inlier_error_m"] = max_inlier_error

    out_path = Path(args.output_config) if args.output_config else in_path
    with open(out_path, "w") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False)
    print(f"Saved refined homography to {out_path}")

if __name__ == "__main__":
    main()
