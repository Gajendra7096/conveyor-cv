from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from common.video import ensure_directory, read_frame, save_image, video_info
from module_b_calibration.camera_models import estimate_speed, load_calibration
from module_c_motion.lucas_kanade import (
    fit_affine_flow,
    lucas_kanade_flow,
    median_flow_vector,
    visualize_flow,
)


def _box(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    return tuple(parts)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _sparse_flow(previous: np.ndarray, next_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if points is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty(0, bool)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        next_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    valid = status.ravel().astype(bool) & np.all(
        np.isfinite(tracked), axis=2
    ).ravel()
    return points.reshape(-1, 2)[valid], tracked.reshape(-1, 2)[valid], valid


def _draw_sparse(image: np.ndarray, points: np.ndarray, tracked: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    for point, destination in zip(points, tracked):
        start = tuple(int(round(value)) for value in point)
        end = tuple(int(round(value)) for value in destination)
        cv2.arrowedLine(canvas, start, end, (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(canvas, start, 2, (0, 255, 0), -1)
    return canvas


def _draw_affine(image: np.ndarray, result: dict[str, object]) -> np.ndarray:
    canvas = image.copy()
    points = np.asarray(result["points"])
    destinations = np.asarray(result["destinations"])
    inliers = np.asarray(result["inliers"], dtype=bool)
    for index, (point, destination) in enumerate(zip(points, destinations)):
        color = (0, 255, 0) if inliers[index] else (0, 0, 255)
        cv2.arrowedLine(
            canvas,
            tuple(int(round(value)) for value in point),
            tuple(int(round(value)) for value in destination),
            color,
            1,
            cv2.LINE_AA,
            tipLength=0.3,
        )
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dense and sparse optical flow.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--next-frame", type=int)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--roi", type=_box)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--cm-per-pixel", type=float)
    parser.add_argument("--output", type=Path, default=Path("outputs/optical_flow"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = ensure_directory(args.output)
    total_frames, fps, _ = video_info(args.video)
    next_index = args.next_frame if args.next_frame is not None else args.frame + 1
    if args.frame < 0 or next_index < 0 or next_index >= total_frames:
        raise ValueError(f"Frame pair must be within 0..{total_frames - 1}")
    previous = read_frame(args.video, args.frame)
    next_frame = read_frame(args.video, next_index)
    flow = lucas_kanade_flow(previous, next_frame, window=args.window)
    affine = fit_affine_flow(flow, region=args.roi)
    sparse_points, sparse_tracked, _ = _sparse_flow(previous, next_frame)
    save_image(output / "lucas_kanade_flow.png", visualize_flow(previous, flow))
    save_image(output / "sparse_flow.png", _draw_sparse(previous, sparse_points, sparse_tracked))
    save_image(output / "affine_ransac_flow.png", _draw_affine(previous, affine))
    homography = None
    if args.calibration:
        homography = load_calibration(args.calibration).plane_homography
    center = (previous.shape[1] / 2.0, previous.shape[0] / 2.0)
    vector = median_flow_vector(flow, region=args.roi)
    seconds = max(1.0 / fps, 1e-6) * (next_index - args.frame)
    speed = estimate_speed(
        center,
        (float(vector[0]), float(vector[1])),
        seconds,
        plane_homography=homography,
        cm_per_pixel=args.cm_per_pixel,
    )
    farneback = cv2.calcOpticalFlowFarneback(
        _to_gray(previous),
        _to_gray(next_frame),
        None,
        0.5,
        3,
        21,
        3,
        5,
        1.2,
        0,
    )
    cross_check = median_flow_vector(farneback, region=args.roi)
    metrics = {
        "frame": args.frame,
        "next_frame": next_index,
        "fps": fps,
        "median_pixel_flow": vector.tolist(),
        "estimated_speed_cm_s": speed,
        "sparse_features": int(len(sparse_points)),
        "affine_matrix": np.asarray(affine["matrix"]).tolist(),
        "affine_inliers": int(np.asarray(affine["inliers"], dtype=bool).sum()),
        "farneback_cross_check": cross_check.tolist(),
    }
    with (output / "flow_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Estimated belt speed: {speed:.3f} cm/s")
    print(f"Saved optical-flow outputs to {output}")

    print("\n--- Dense vs. sparse optical flow (discussion) ---")
    print(
        "Dense Lucas–Kanade (differential, brightness constancy, local window):"
        " uses every pixel's gradient in the window to solve the 2x2 system, so"
        " it produces a flow vector at almost every pixel. On textureless box faces"
        " (uniform color, small gradient magnitude in all directions) the system"
        " is ill-posed (aperture problem): the window contains no edge or corner"
        " that constrains both components, so many pixels fall below the"
        " eigenvalue threshold and return zero or noisy flow."
    )
    print(
        "Sparse LK on corner features (cv2.goodFeaturesToTrack + calcOpticalFlowPyrLK):"
        " only tracks points that have strong 2D texture (corners), where the"
        " auto-correlation is well-conditioned. Feature tracks are reliable where"
        " features exist (box edges/corners), but there are no tracks on large"
        " textureless surfaces. This is exactly the aperture/observability trade-off:"
        " dense gives coverage, sparse gives selectivity."
    )
    print(
        "Affine 6-parameter model fitted by RANSAC: the flow over a rigid planar"
        " region is well approximated by an affine (or perspective) transform."
        " Fitting to the dense field and rejecting outliers (RANSAC) isolates the"
        " coherent motion of the box and downweights inconsistent vectors"
        " (e.g. background, shadows, or the zero-flow regions created by the"
        " aperture problem on textureless patches)."
    )


if __name__ == "__main__":
    main()
