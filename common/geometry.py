from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def resize_max_side(image: np.ndarray, max_side: int = 640) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale == 1.0:
        return image, 1.0
    resized = cv2_resize(image, width, height, scale)
    return resized, scale


def cv2_resize(image: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
    import cv2

    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def bbox_corners(box: tuple[float, float, float, float]) -> np.ndarray:
    x, y, width, height = box
    return np.array(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float32,
    )


def point_scale_from_homography(homography: np.ndarray, point: tuple[float, float]) -> float:
    matrix = np.asarray(homography, dtype=np.float64)
    source = np.array([[[point[0], point[1], 1.0]]], dtype=np.float64)
    mapped = cv2_perspective_transform(source, matrix)[0, 0]
    dx = 1.0
    px, py = mapped[:2]
    x_map = cv2_perspective_transform(
        np.array([[[point[0] + dx, point[1], 1.0]]], dtype=np.float64), matrix
    )[0, 0, :2]
    return float(math.hypot(x_map[0] - px, x_map[1] - py) / dx)


def cv2_perspective_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.perspectiveTransform(points.astype(np.float32), matrix.astype(np.float64))


def median_vector_magnitude(vectors: np.ndarray) -> float:
    values = np.asarray(vectors, dtype=np.float64).reshape(-1, 2)
    if values.size == 0:
        return 0.0
    return float(np.median(np.linalg.norm(values, axis=1)))


def finite_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return array[np.all(np.isfinite(array), axis=1)]


def normalized(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(array)
    return array if norm == 0 else array / norm


def distances(points: Iterable[tuple[float, float]], target: tuple[float, float]) -> np.ndarray:
    values = np.asarray(list(points), dtype=np.float64)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.linalg.norm(values - np.asarray(target), axis=1)
