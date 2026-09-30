from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def video_info(path: str | Path) -> tuple[int, float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or not np.isfinite(fps):
        fps = 30.0
    if width <= 0 or height <= 0:
        raise ValueError(f"Video has invalid dimensions: {path}")
    return total_frames, fps, (width, height)


def iter_video_frames(
    path: str | Path,
    stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    frame_index = 0
    yielded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                yield frame_index, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()


def read_frame(path: str | Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Unable to read frame {frame_index} from {path}")
    return frame


def sampled_indices(
    total_frames: int,
    count: int = 5,
    start: int = 0,
    end: int | None = None,
) -> list[int]:
    if total_frames < 1 or count < 1:
        return []
    last = total_frames - 1 if end is None else min(total_frames - 1, end)
    first = max(0, min(start, last))
    if last <= first:
        return [first]
    return [int(round(value)) for value in np.linspace(first, last, count)]


def write_video(
    path: str | Path,
    frames: list[np.ndarray],
    fps: float,
    size: tuple[int, int] | None = None,
) -> Path:
    output = Path(path)
    ensure_directory(output.parent)
    if not frames:
        raise ValueError("Cannot write a video without frames")
    height, width = frames[0].shape[:2]
    if size is not None and tuple(size) != (width, height):
        raise ValueError("Video frame size does not match requested size")
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(fps)),
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Unable to open video writer for {output}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return output


def save_image(path: str | Path, image: np.ndarray) -> Path:
    output = Path(path)
    ensure_directory(output.parent)
    if not cv2.imwrite(str(output), image):
        raise ValueError(f"Unable to write image: {output}")
    return output
