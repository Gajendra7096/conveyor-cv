from __future__ import annotations

import time

import cv2
import numpy as np

from common.types import BoxDetection

METHOD_NAMES = ("split_merge", "watershed", "felzenszwalb", "mean_shift")


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _resize_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if labels.shape[:2] == shape:
        return labels
    return cv2.resize(
        labels.astype(np.int32),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def _binary_foreground(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate(
        [binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]]
    )
    if float(np.mean(border)) > 127.0:
        binary = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def watershed_segmentation(image: np.ndarray) -> np.ndarray:
    gray = _gray(image)
    binary = _binary_foreground(gray)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if float(distance.max()) <= 0:
        return np.zeros(gray.shape, dtype=np.int32)
    markers = np.zeros(gray.shape, dtype=np.int32)
    background = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    markers[background == 0] = 1
    threshold = max(2.0, float(distance.max()) * 0.45)
    sure_foreground = (distance > threshold) & (binary > 0)
    count, component_labels = cv2.connectedComponents(
        sure_foreground.astype(np.uint8), 8
    )
    for label in range(1, count):
        markers[component_labels == label] = label + 1
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(rgb, markers)
    markers[markers <= 1] = 0
    return markers.astype(np.int32)


def _region_mean_variance(
    sums: np.ndarray,
    squares: np.ndarray,
    rect: tuple[int, int, int, int],
) -> tuple[float, float]:
    x, y, width, height = rect
    total = float(sums[y + height, x + width] - sums[y, x + width] - sums[y + height, x] + sums[y, x])
    square_total = float(
        squares[y + height, x + width]
        - squares[y, x + width]
        - squares[y + height, x]
        + squares[y, x]
    )
    count = max(1, width * height)
    mean = total / count
    variance = max(0.0, square_total / count - mean * mean)
    return mean, variance


def split_merge_segmentation(
    image: np.ndarray,
    variance_threshold: float = 180.0,
    merge_tolerance: float = 18.0,
    min_side: int = 16,
) -> np.ndarray:
    gray = _gray(image)
    height, width = gray.shape
    values = gray.astype(np.float64)
    sums = cv2.integral(values)
    squares = cv2.integral(values * values)
    leaves: list[tuple[int, int, int, int, float]] = []
    stack = [(0, 0, width, height)]
    while stack:
        x, y, region_width, region_height = stack.pop()
        mean, variance = _region_mean_variance(sums, squares, (x, y, region_width, region_height))
        can_split = region_width >= min_side * 2 or region_height >= min_side * 2
        if variance > variance_threshold and can_split:
            if region_width >= region_height and region_width >= min_side * 2:
                middle = x + region_width // 2
                stack.append((x, y, middle - x, region_height))
                stack.append((middle, y, x + region_width - middle, region_height))
            elif region_height >= min_side * 2:
                middle = y + region_height // 2
                stack.append((x, y, region_width, middle - y))
                stack.append((x, middle, region_width, y + region_height - middle))
            else:
                leaves.append((x, y, region_width, region_height, mean))
        else:
            leaves.append((x, y, region_width, region_height, mean))

    parent = list(range(len(leaves)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    lookup: dict[tuple[int, int], int] = {}
    for index, (x, y, _, _, _) in enumerate(leaves):
        lookup[(x, y)] = index
    for index, (x, y, region_width, region_height, _) in enumerate(leaves):
        neighbors = ((x + region_width, y), (x, y + region_height))
        for neighbor in neighbors:
            other = lookup.get(neighbor)
            if other is not None and abs(leaves[index][4] - leaves[other][4]) <= merge_tolerance:
                union(index, other)

    labels = np.zeros((height, width), dtype=np.int32)
    for index, (x, y, region_width, region_height, _) in enumerate(leaves):
        label = find(index)
        labels[y : y + region_height, x : x + region_width] = label + 1
    return labels


def felzenszwalb_segmentation(image: np.ndarray) -> np.ndarray:
    from skimage.segmentation import felzenszwalb

    gray = _gray(image)
    scale = min(1.0, 640.0 / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (int(gray.shape[1] * scale), int(gray.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    labels = felzenszwalb(gray, scale=1.0, sigma=0.8, min_size=80).astype(np.int32)
    return _resize_labels(labels, image.shape[:2])


def _nearest_center_labels(
    features: np.ndarray,
    centers: np.ndarray,
    chunk: int = 20000,
) -> np.ndarray:
    """Assign each row to the nearest center without materialising all distances.

    A full (rows x clusters x dims) array is what made mean shift allocate hundreds
    of megabytes on a single HD frame.
    """
    labels = np.empty(features.shape[0], dtype=np.int32)
    squared = np.sum(centers * centers, axis=1)[None, :]
    for start in range(0, features.shape[0], chunk):
        block = features[start : start + chunk]
        distances = squared - 2.0 * (block @ centers.T)
        labels[start : start + chunk] = np.argmin(distances, axis=1)
    return labels


def mean_shift_segmentation(
    image: np.ndarray,
    max_samples: int = 3500,
    bandwidth: float | None = None,
) -> np.ndarray:
    from sklearn.cluster import MeanShift

    small = image
    height, width = image.shape[:2]
    scale = min(1.0, 480.0 / max(height, width))
    if scale < 1.0:
        small = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    pixels = small.reshape(-1, 3).astype(np.float32) / 255.0
    yy, xx = np.indices(small.shape[:2], dtype=np.float32)
    position_weight = 2.0
    features = np.column_stack(
        (
            pixels,
            (xx.reshape(-1) / max(1, small.shape[1])) * position_weight,
            (yy.reshape(-1) / max(1, small.shape[0])) * position_weight,
        )
    )
    if features.shape[0] > max_samples:
        rng = np.random.default_rng(7)
        selected = rng.choice(features.shape[0], max_samples, replace=False)
        sample = features[selected]
    else:
        sample = features
    if bandwidth is None:
        bandwidth = 0.12
    clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=80)
    clusterer.fit(sample)
    centers = clusterer.cluster_centers_.astype(np.float32)
    labels = _nearest_center_labels(features, centers)
    return _resize_labels(labels.reshape(small.shape[:2]), (height, width))


def segment_frame(image: np.ndarray, method: str) -> np.ndarray:
    if method == "split_merge":
        return split_merge_segmentation(image)
    if method == "watershed":
        return watershed_segmentation(image)
    if method == "felzenszwalb":
        return felzenszwalb_segmentation(image)
    if method == "mean_shift":
        return mean_shift_segmentation(image)
    raise ValueError(f"Unknown segmentation method: {method}")


def _label_mask(labels: np.ndarray, label: int) -> np.ndarray:
    return (labels == label).astype(np.uint8) * 255


def labels_to_boxes(
    labels: np.ndarray,
    min_area: int = 250,
    max_boxes: int = 30,
) -> list[BoxDetection]:
    candidates: list[BoxDetection] = []
    for label in np.unique(labels):
        if label <= 0:
            continue
        mask = _label_mask(labels, int(label))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < 12 or box_height < 12:
                continue
            candidates.append(
                BoxDetection(
                    x=int(x),
                    y=int(y),
                    width=int(box_width),
                    height=int(box_height),
                    score=float(area),
                )
            )
    candidates.sort(key=lambda box: box.area * box.score, reverse=True)
    return candidates[:max_boxes]


def _colorize(labels: np.ndarray) -> np.ndarray:
    maximum = max(1, int(labels.max()))
    colored = cv2.applyColorMap(
        ((labels * 37) % 256).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    colored[labels <= 0] = (25, 25, 25)
    if maximum <= 1:
        return colored
    return colored


def production_detections(image: np.ndarray, min_area: int = 350) -> list[BoxDetection]:
    gray = _gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if float(np.mean(border)) > 127.0:
        binary = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[BoxDetection] = []
    height, width = gray.shape
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 20 or box_height < 20:
            continue
        if box_width / max(1, box_height) > 8 or box_height / max(1, box_width) > 8:
            continue
        detections.append(
            BoxDetection(
                x=max(0, int(x)),
                y=max(0, int(y)),
                width=min(width, int(box_width)),
                height=min(height, int(box_height)),
                score=area,
            )
        )
    detections.sort(key=lambda box: box.x)
    return detections


def reject_static_detections(
    detections: list[BoxDetection],
    previous: list[BoxDetection],
    static_distance: float = 2.0,
) -> list[BoxDetection]:
    """Drop detections sitting exactly where one sat in the previous frame.

    Otsu plus RETR_EXTERNAL accepts any large dark blob, so belt strips, roller
    gaps and shadows become permanent false positives. A real box on a moving
    belt shifts by several pixels between frames, so a near-zero displacement is
    a strong static-background cue. Detections that are small or that were absent
    from the previous frame are always kept.
    """
    if not previous or not detections:
        return list(detections)
    centers = np.asarray([box.center for box in previous], dtype=np.float64)
    keep: list[BoxDetection] = []
    for detection in detections:
        distances = np.linalg.norm(centers - np.asarray(detection.center, dtype=np.float64), axis=1)
        if float(distances.min()) > static_distance:
            keep.append(detection)
    return keep


def select_tracking_detections(
    detections: list[BoxDetection],
    previous: list[BoxDetection],
    *,
    keep_static: bool = False,
    has_previous_frame: bool = True,
) -> list[BoxDetection]:
    """Decide which detections the tracker should be given this frame.

    Both entry points (pipeline.py and kalman_tracker.py) need this decision, and
    they had drifted apart while P4 was being fixed, so it lives here once.

    `previous` must be the RAW detections from the previous frame, not the
    filtered ones. A detection rejected as static has to still be present next
    frame for the comparison, or it is re-admitted and tracked instead.

    The first frame is withheld entirely. With nothing to compare against, a
    static background region and a real box are indistinguishable, and feeding it
    to the tracker creates a track that then coasts on its own velocity for the
    rest of its confirmation window. The cost is that a stationary object which
    starts moving is picked up one frame late.
    """
    if keep_static:
        return list(detections)
    if not has_previous_frame:
        return []
    return reject_static_detections(detections, previous)


def compare_methods(
    image: np.ndarray,
    methods: tuple[str, ...] = METHOD_NAMES,
    return_labels: bool = False,
) -> tuple[np.ndarray, dict[str, float]] | tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    panels: list[np.ndarray] = []
    timings: dict[str, float] = {}
    label_maps: dict[str, np.ndarray] = {}
    for method in methods:
        start = time.perf_counter()
        labels = segment_frame(image, method)
        timings[method] = time.perf_counter() - start
        label_maps[method] = labels
        panel = _colorize(_resize_labels(labels, image.shape[:2])).copy()
        cv2.putText(panel, method, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    columns = len(panels)
    rows = (len(panels) + columns - 1) // columns
    height, width = panels[0].shape[:2]
    grid = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        grid[row * height : (row + 1) * height, column * width : (column + 1) * width] = panel
    if return_labels:
        return grid, timings, label_maps
    return grid, timings
