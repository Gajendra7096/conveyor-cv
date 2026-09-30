from __future__ import annotations

import cv2
import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _box_sum(image: np.ndarray, window: int) -> np.ndarray:
    return cv2.boxFilter(
        image,
        ddepth=cv2.CV_64F,
        ksize=(window, window),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )


def _dense_level(
    previous_gray: np.ndarray,
    next_gray: np.ndarray,
    window: int,
    min_eigenvalue: float,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the dense 2x2 Lucas-Kanade system on one image level."""
    gradient_x = cv2.Sobel(previous_gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gradient_y = cv2.Sobel(previous_gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    if initial is None:
        residual = next_gray - previous_gray
    else:
        height, width = previous_gray.shape
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        warped = cv2.remap(
            next_gray,
            (grid_x + initial[..., 0]).astype(np.float32),
            (grid_y + initial[..., 1]).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        residual = warped - previous_gray
    xx = _box_sum(gradient_x * gradient_x, window)
    xy = _box_sum(gradient_x * gradient_y, window)
    yy = _box_sum(gradient_y * gradient_y, window)
    bx = -_box_sum(gradient_x * residual, window)
    by = -_box_sum(gradient_y * residual, window)
    determinant = xx * yy - xy * xy
    trace = xx + yy
    minimum = min_eigenvalue * max(1.0, float(np.max(trace)) ** 2)
    valid = (determinant > minimum) & (trace > min_eigenvalue)
    safe_determinant = np.where(np.abs(determinant) < 1e-12, 1e-12, determinant)
    flow_x = (bx * yy - by * xy) / safe_determinant
    flow_y = (xx * by - xy * bx) / safe_determinant
    flow = np.stack((flow_x, flow_y), axis=-1).astype(np.float32)
    flow[~valid] = 0.0
    return flow


def _zero_border(flow: np.ndarray, window: int) -> np.ndarray:
    border = window // 2 + 1
    flow[:border] = 0
    flow[-border:] = 0
    flow[:, :border] = 0
    flow[:, -border:] = 0
    return flow


def lucas_kanade_flow(
    previous: np.ndarray,
    next_frame: np.ndarray,
    window: int = 15,
    min_eigenvalue: float = 1e-4,
    levels: int | None = None,
    max_levels: int = 5,
) -> np.ndarray:
    """Dense Lucas-Kanade flow with optional coarse-to-fine refinement.

    A single-level solve is only valid for small inter-frame motion, because
    the brightness-constancy assumption is linearised around zero displacement.
    Each extra pyramid level halves the motion seen by that level, which
    extends the usable range well beyond one window width.
    """
    previous_gray = _gray(previous)
    next_gray = _gray(next_frame)
    if previous_gray.shape != next_gray.shape:
        raise ValueError("Consecutive frames must have the same dimensions")
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer of at least 3")
    height, width = previous_gray.shape[:2]
    if levels is None:
        levels = 1
        while levels < max_levels and min(height, width) // (2**levels) >= window:
            levels += 1
    else:
        levels = max(1, int(levels))
        while levels > 1 and min(height, width) // (2 ** (levels - 1)) < window:
            levels -= 1

    previous_pyramid = [previous_gray]
    next_pyramid = [next_gray]
    for _ in range(levels - 1):
        previous_pyramid.append(
            cv2.pyrDown(previous_pyramid[-1], dstsize=(previous_pyramid[-1].shape[1] // 2, previous_pyramid[-1].shape[0] // 2))
        )
        next_pyramid.append(
            cv2.pyrDown(next_pyramid[-1], dstsize=(next_pyramid[-1].shape[1] // 2, next_pyramid[-1].shape[0] // 2))
        )

    flow: np.ndarray | None = None
    for level in range(levels - 1, -1, -1):
        current = cv2.GaussianBlur(previous_pyramid[level].astype(np.float32), (3, 3), 0)
        following = cv2.GaussianBlur(next_pyramid[level].astype(np.float32), (3, 3), 0)
        if flow is None:
            flow = _dense_level(current, following, window, min_eigenvalue)
        else:
            upsampled = cv2.resize(flow * 2.0, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_LINEAR)
            increment = _dense_level(current, following, window, min_eigenvalue, upsampled)
            flow = (upsampled + increment).astype(np.float32)
    assert flow is not None
    return _zero_border(flow, window)


def visualize_flow(
    image: np.ndarray,
    flow: np.ndarray,
    step: int = 18,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    canvas = image.copy()
    height, width = flow.shape[:2]
    for y in range(step, height - step, step):
        for x in range(step, width - step, step):
            vector = flow[y, x]
            if not np.all(np.isfinite(vector)):
                continue
            end_x = int(round(x + vector[0]))
            end_y = int(round(y + vector[1]))
            if not (0 <= end_x < width and 0 <= end_y < height):
                continue
            cv2.arrowedLine(canvas, (x, y), (end_x, end_y), color, 1, cv2.LINE_AA, tipLength=0.3)
    return canvas


def fit_affine_flow(
    flow: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
    ransac_threshold: float = 1.5,
    max_points: int = 2500,
) -> dict[str, object]:
    height, width = flow.shape[:2]
    if region is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        rx, ry, rw, rh = region
        x1 = max(0, min(width - 1, int(rx)))
        y1 = max(0, min(height - 1, int(ry)))
        x2 = max(x1 + 1, min(width, int(rx + rw)))
        y2 = max(y1 + 1, min(height, int(ry + rh)))
    coordinates = np.zeros((y2 - y1, x2 - x1, 2), dtype=np.float32)
    coordinates[..., 0] = np.arange(x1, x2, dtype=np.float32)[None, :]
    coordinates[..., 1] = np.arange(y1, y2, dtype=np.float32)[:, None]
    vectors = flow[y1:y2, x1:x2].copy()
    valid = np.linalg.norm(vectors, axis=-1) > 0.05
    points = coordinates[valid]
    destinations = points + vectors[valid]
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
        points = points[indices]
        destinations = destinations[indices]
    if len(points) < 6:
        return {
            "matrix": np.eye(2, 3, dtype=np.float64),
            "inliers": np.zeros(len(points), dtype=bool),
            "points": points,
            "destinations": destinations,
        }
    matrix, inliers = cv2.estimateAffine2D(
        points,
        destinations,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.995,
    )
    if matrix is None:
        return {
            "matrix": np.eye(2, 3, dtype=np.float64),
            "inliers": np.ones(len(points), dtype=bool),
            "points": points,
            "destinations": destinations,
        }
    inlier_mask = inliers.ravel().astype(bool)
    if inlier_mask.sum() >= 6:
        refined, _ = cv2.estimateAffine2D(
            points[inlier_mask],
            destinations[inlier_mask],
            method=cv2.LMEDS,
        )
        if refined is not None:
            matrix = refined
    return {
        "matrix": matrix,
        "inliers": inlier_mask,
        "points": points,
        "destinations": destinations,
    }


def median_flow_vector(flow: np.ndarray, region: tuple[int, int, int, int] | None = None) -> np.ndarray:
    height, width = flow.shape[:2]
    if region is None:
        values = flow.reshape(-1, 2)
    else:
        x, y, box_width, box_height = region
        x1 = max(0, min(width - 1, int(x)))
        y1 = max(0, min(height - 1, int(y)))
        x2 = max(x1 + 1, min(width, int(x + box_width)))
        y2 = max(y1 + 1, min(height, int(y + box_height)))
        values = flow[y1:y2, x1:x2].reshape(-1, 2)
    values = values[np.all(np.isfinite(values), axis=1)]
    values = values[np.linalg.norm(values, axis=1) > 0.05]
    if len(values) == 0:
        return np.zeros(2, dtype=np.float64)
    return np.median(values, axis=0)
