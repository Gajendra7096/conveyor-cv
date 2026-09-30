from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CalibrationData:
    K: np.ndarray
    distortion: np.ndarray
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    image_size: tuple[int, int]
    pattern_size: tuple[int, int]
    square_size_cm: float
    reprojection_error: float
    plane_homography: np.ndarray | None = None

    @property
    def projection_matrices(self) -> list[np.ndarray]:
        matrices: list[np.ndarray] = []
        for rvec, tvec in zip(self.rvecs, self.tvecs):
            rotation, _ = cv2.Rodrigues(rvec)
            extrinsic = np.hstack((rotation, tvec.reshape(3, 1)))
            matrices.append(self.K @ extrinsic)
        return matrices

    def undistort(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(image, self.K, self.distortion)

    def map_image_to_plane(self, points: np.ndarray) -> np.ndarray:
        if self.plane_homography is None:
            raise ValueError("Calibration file does not contain a plane homography")
        source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(source, self.plane_homography).reshape(-1, 2)


def save_calibration(path: str | Path, data: CalibrationData) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        K=data.K,
        distortion=data.distortion,
        rvecs=np.asarray(data.rvecs, dtype=object),
        tvecs=np.asarray(data.tvecs, dtype=object),
        image_size=np.asarray(data.image_size, dtype=np.int32),
        pattern_size=np.asarray(data.pattern_size, dtype=np.int32),
        square_size_cm=np.asarray(data.square_size_cm),
        reprojection_error=np.asarray(data.reprojection_error),
        plane_homography=(
            data.plane_homography
            if data.plane_homography is not None
            else np.empty((0, 0), dtype=np.float64)
        ),
    )
    return output


def load_calibration(path: str | Path) -> CalibrationData:
    with np.load(path, allow_pickle=True) as values:
        plane_h = values["plane_homography"]
        return CalibrationData(
            K=values["K"].astype(np.float64),
            distortion=values["distortion"].astype(np.float64),
            rvecs=[value.astype(np.float64) for value in values["rvecs"]],
            tvecs=[value.astype(np.float64) for value in values["tvecs"]],
            image_size=tuple(int(value) for value in values["image_size"]),
            pattern_size=tuple(int(value) for value in values["pattern_size"]),
            square_size_cm=float(values["square_size_cm"]),
            reprojection_error=float(values["reprojection_error"]),
            plane_homography=(
                plane_h.astype(np.float64) if plane_h.size else None
            ),
        )


def _homography_scale(homography: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    matrix = np.asarray(homography, dtype=np.float64)
    center = np.array([[[point[0], point[1]]]], dtype=np.float32)
    x_point = np.array([[[point[0] + 1.0, point[1]]]], dtype=np.float32)
    y_point = np.array([[[point[0], point[1] + 1.0]]], dtype=np.float32)
    center_world = cv2.perspectiveTransform(center, matrix)[0, 0]
    x_world = cv2.perspectiveTransform(x_point, matrix)[0, 0]
    y_world = cv2.perspectiveTransform(y_point, matrix)[0, 0]
    return (
        float(np.linalg.norm(x_world - center_world)),
        float(np.linalg.norm(y_world - center_world)),
    )


def _bbox_corners(box: tuple[float, float, float, float]) -> np.ndarray:
    x, y, width, height = box
    return np.array(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float32,
    )


def estimate_box_size(
    reference_box: tuple[float, float, float, float],
    reference_size_cm: tuple[float, float],
    unknown_box: tuple[float, float, float, float],
    plane_homography: np.ndarray | None = None,
) -> dict[str, tuple[float, float]]:
    ref_x, ref_y, ref_width, ref_height = reference_box
    unknown_x, unknown_y, unknown_width, unknown_height = unknown_box
    ref_real_width, ref_real_height = reference_size_cm
    ortho_width = unknown_width * ref_real_width / max(1.0, ref_width)
    ortho_height = unknown_height * ref_real_height / max(1.0, ref_height)
    scale = 0.5 * (
        ref_real_width / max(1.0, ref_width) + ref_real_height / max(1.0, ref_height)
    )
    weak = (unknown_width * scale, unknown_height * scale)
    affine = (ortho_width, ortho_height)
    perspective = (ortho_width, ortho_height)
    if plane_homography is not None:
        center = (unknown_x + unknown_width / 2.0, unknown_y + unknown_height / 2.0)
        affine_scale_x, affine_scale_y = _homography_scale(plane_homography, center)
        affine = (unknown_width * affine_scale_x, unknown_height * affine_scale_y)
        corners = _bbox_corners(unknown_box).reshape(-1, 1, 2)
        world = cv2.perspectiveTransform(corners, np.asarray(plane_homography, dtype=np.float64)).reshape(-1, 2)
        top = float(np.linalg.norm(world[1] - world[0]))
        bottom = float(np.linalg.norm(world[2] - world[3]))
        left = float(np.linalg.norm(world[3] - world[0]))
        right = float(np.linalg.norm(world[2] - world[1]))
        perspective = (0.5 * (top + bottom), 0.5 * (left + right))
    return {
        "orthographic": (float(ortho_width), float(ortho_height)),
        "weak_perspective": (float(weak[0]), float(weak[1])),
        "affine": (float(affine[0]), float(affine[1])),
        "full_perspective": (float(perspective[0]), float(perspective[1])),
    }


def estimate_speed(
    pixel_point: tuple[float, float],
    pixel_vector: tuple[float, float],
    seconds: float,
    plane_homography: np.ndarray | None = None,
    cm_per_pixel: float | None = None,
) -> float:
    if seconds <= 0:
        return 0.0
    if plane_homography is not None:
        points = np.array(
            [pixel_point, (pixel_point[0] + pixel_vector[0], pixel_point[1] + pixel_vector[1])],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        world = cv2.perspectiveTransform(
            points, np.asarray(plane_homography, dtype=np.float64)
        ).reshape(-1, 2)
        return float(np.linalg.norm(world[1] - world[0]) / seconds)
    if cm_per_pixel is None:
        return float("nan")
    return float(np.linalg.norm(pixel_vector) * cm_per_pixel / seconds)
