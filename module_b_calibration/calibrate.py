from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from scipy.linalg import rq

from module_b_calibration.camera_models import CalibrationData, save_calibration


def _find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    detector = getattr(cv2, "findChessboardCornersSB", None)
    if detector is not None:
        found, corners = detector(gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE)
        if found:
            return corners.astype(np.float32)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    return cv2.cornerSubPix(
        gray,
        corners.astype(np.float32),
        (11, 11),
        (-1, -1),
        criteria,
    ).astype(np.float32)


def _decompose_projection(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(projection, dtype=np.float64)
    intrinsic, rotation = rq(matrix[:, :3])
    signs = np.sign(np.diag(intrinsic))
    signs[signs == 0] = 1.0
    sign_matrix = np.diag(signs)
    intrinsic = intrinsic @ sign_matrix
    rotation = sign_matrix @ rotation
    if np.linalg.det(rotation) < 0:
        rotation = -rotation
        intrinsic = -intrinsic
    translation = np.linalg.solve(intrinsic, matrix[:, 3])
    return intrinsic, rotation, translation


def _reprojection_error(
    points: list[np.ndarray],
    object_points: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
) -> float:
    errors: list[float] = []
    for image_points, world_points, rvec, tvec in zip(points, object_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            world_points, rvec, tvec, camera_matrix, distortion
        )
        observed = np.asarray(image_points).reshape(-1, 2)
        predicted = np.asarray(projected).reshape(-1, 2)
        errors.extend(np.linalg.norm(predicted - observed, axis=1).tolist())
    return float(np.mean(errors)) if errors else float("inf")


def _board_normal_spread(tvecs: list[np.ndarray]) -> float:
    """Largest angle in degrees between the optical axes of the captured views.

    A near-zero value means every board view was taken from the same direction, so
    focal length and the lens distortion terms are not identifiable from the data.
    """
    normals = []
    for tvec in tvecs:
        norm = float(np.linalg.norm(tvec))
        if norm > 1e-9:
            normals.append(np.asarray(tvec, dtype=np.float64).reshape(3) / norm)
    if len(normals) < 2:
        return 0.0
    spread = 0.0
    for index, first in enumerate(normals):
        for second in normals[index + 1 :]:
            cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
            spread = max(spread, float(np.degrees(np.arccos(cosine))))
    return spread


def calibrate_images(
    images: list[np.ndarray],
    pattern_size: tuple[int, int],
    square_size_cm: float,
) -> CalibrationData:
    if len(images) < 3:
        raise ValueError("At least three checkerboard images are required")
    height, width = images[0].shape[:2]
    if any(image.shape[:2] != (height, width) for image in images):
        raise ValueError("All calibration images must have the same resolution")
    object_template = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    object_template[:, :2] = np.mgrid[
        0 : pattern_size[0], 0 : pattern_size[1]
    ].T.reshape(-1, 2)
    object_template *= np.float32(square_size_cm)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = _find_corners(gray, pattern_size)
        if corners is None:
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
    if len(object_points) < 3:
        raise ValueError(
            f"Checkerboard detected in only {len(object_points)} images; expected at least 3"
        )
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        (width, height),
        None,
        None,
    )
    plane_homography, homography_inliers = cv2.findHomography(
        image_points[0], object_points[0][:, :2], cv2.RANSAC, 3.0
    )
    if plane_homography is None or homography_inliers is None:
        raise ValueError("Could not fit a plane homography to the first board view")
    inlier_ratio = float(np.mean(homography_inliers))
    if inlier_ratio < 0.9:
        print(
            f"WARNING: plane homography kept only {inlier_ratio * 100:.0f}% of corners "
            f"in view 0; check the first checkerboard image"
        )
    error = _reprojection_error(
        image_points, object_points, camera_matrix, distortion, rvecs, tvecs
    )
    spread = _board_normal_spread(tvecs)
    if spread < 10.0:
        print(
            f"WARNING: the captured board views differ by only {spread:.1f} deg. "
            "Focal length and distortion are not identifiable from near-identical "
            "view directions; retake the board tilted roughly 20-45 deg in several "
            "directions."
        )
    return CalibrationData(
        K=camera_matrix,
        distortion=distortion,
        rvecs=rvecs,
        tvecs=tvecs,
        image_size=(width, height),
        pattern_size=pattern_size,
        square_size_cm=square_size_cm,
        reprojection_error=error,
        plane_homography=plane_homography,
    )


def _load_images(source: Path) -> list[np.ndarray]:
    if source.is_dir():
        paths = sorted(
            Path(path)
            for path in glob.glob(str(source / "*.jpg"))
            + glob.glob(str(source / "*.jpeg"))
            + glob.glob(str(source / "*.png"))
        )
    else:
        paths = [source]
    images: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is not None:
            images.append(image)
    if not images:
        raise ValueError(f"No readable calibration images found at {source}")
    return images


def _save_undistortion_sample(data: CalibrationData, image: np.ndarray, output: Path) -> None:
    before = image.copy()
    after = data.undistort(image)
    cv2.putText(before, "before", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(after, "after", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.concatenate((before, after), axis=1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate a camera from checkerboard images.")
    parser.add_argument("--images", required=True, type=Path, help="Image file or directory.")
    parser.add_argument("--cols", type=int, required=True, help="Number of inner corners across.")
    parser.add_argument("--rows", type=int, required=True, help="Number of inner corners down.")
    parser.add_argument("--square-size", type=float, required=True, help="Checker square size in cm.")
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration.npz"))
    parser.add_argument("--undistort-output", type=Path, default=Path("outputs/undistortion.png"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images = _load_images(args.images)
    data = calibrate_images(images, (args.cols, args.rows), args.square_size)
    save_calibration(args.output, data)
    _save_undistortion_sample(data, images[0], args.undistort_output)
    print(f"Saved calibration to {args.output}")
    print(f"RMS calibration error: {data.reprojection_error:.4f} px")
    print(
        f"Board view spread: {_board_normal_spread(data.tvecs):.1f} deg "
        "(focal length needs roughly 20-45 deg of tilt)"
    )
    print(
        "NOTE: plane_homography is fitted to the FIRST checkerboard view only. "
        "It is valid for objects lying on that same plane. Move the board to the "
        "belt plane before capturing, or capture it in the belt position, or "
        "re-run calibration on a board image taken at the belt plane."
    )
    print("K:")
    print(data.K)
    print("Distortion:", data.distortion.ravel())
    projection = data.projection_matrices[0]
    intrinsic, rotation, translation = _decompose_projection(projection)
    print("Projection decomposition K:", intrinsic)
    print("Projection decomposition R:", rotation)
    print("Projection decomposition t:", translation.ravel())
    print(f"Saved undistortion sample to {args.undistort_output}")


if __name__ == "__main__":
    main()
