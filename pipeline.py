from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from common.video import ensure_directory, save_image, video_info
from module_a_segmentation.segmentation import production_detections, select_tracking_detections
from module_b_calibration.camera_models import estimate_speed, load_calibration
from module_c_motion.lucas_kanade import fit_affine_flow, lucas_kanade_flow, median_flow_vector
from module_d_tracking.kalman import MultiBoxTracker
from module_e_recognition.recognize import RecognitionEngine, load_crops

COLORS = (
    (0, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (200, 200, 200),
)


@dataclass
class PipelineConfig:
    video: Path
    calibration: Path | None = None
    recognition_root: Path | None = None
    cm_per_pixel: float | None = None
    output: Path = Path("outputs/pipeline")
    max_frames: int = 300
    stride: int = 1
    flow_every: int = 5
    max_boxes: int = 12
    keep_static: bool = False
    belt_velocity: tuple[float, float] | None = None
    write_video: bool = True


@dataclass
class PipelineResult:
    output: Path
    frames_processed: int
    rows: list[dict[str, object]] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)
    flow_preview: Path | None = None
    annotated_video: Path | None = None
    results_csv: Path | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "output": str(self.output),
            "frames_processed": self.frames_processed,
            "records": len(self.rows),
            "calibrated": bool(self.summary.get("calibrated")),
            "recognition_enabled": bool(self.summary.get("recognition_enabled")),
            "rows": self.rows,
            "artifacts": {
                "flow_preview": str(self.flow_preview) if self.flow_preview else None,
                "annotated_video": str(self.annotated_video) if self.annotated_video else None,
                "results_csv": str(self.results_csv) if self.results_csv else None,
            },
        }


@dataclass
class PipelineProgress:
    processed: int
    max_frames: int
    frame_index: int
    detections: int
    tracks: int
    latest_speed: float
    preview: np.ndarray | None = None


def _world_size(homography: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, width, height = box
    corners = np.array(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    return (
        float(np.mean([np.linalg.norm(mapped[1] - mapped[0]), np.linalg.norm(mapped[2] - mapped[3])])),
        float(np.mean([np.linalg.norm(mapped[3] - mapped[0]), np.linalg.norm(mapped[2] - mapped[1])])),
    )


def _fallback_size(box: tuple[int, int, int, int], cm_per_pixel: float | None) -> tuple[float, float]:
    if cm_per_pixel is None:
        return float("nan"), float("nan")
    return box[2] * cm_per_pixel, box[3] * cm_per_pixel


def _crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = box
    x1 = max(0, min(frame.shape[1] - 1, x))
    y1 = max(0, min(frame.shape[0] - 1, y))
    x2 = max(x1 + 1, min(frame.shape[1], x + width))
    y2 = max(y1 + 1, min(frame.shape[0], y + height))
    return frame[y1:y2, x1:x2]


def _draw(frame: np.ndarray, detections, tracks, rows: list[dict[str, object]]) -> np.ndarray:
    canvas = frame.copy()
    for detection in detections:
        x, y = detection.x, detection.y
        cv2.rectangle(canvas, (x, y), (detection.x2, detection.y2), (160, 160, 160), 1)
        center = tuple(int(round(value)) for value in detection.center)
        cv2.circle(canvas, center, 4, (160, 160, 160), -1)
    for track in tracks:
        color = COLORS[(track.track_id - 1) % len(COLORS)]
        point = tuple(int(round(value)) for value in track.position)
        cv2.circle(canvas, point, 7, color, 2)
        cv2.putText(canvas, f"ID {track.track_id}", (point[0] + 9, point[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"boxes={len(detections)} tracks={len(tracks)}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if rows:
        latest_speed = rows[-1]["speed_cm_s"]
        if isinstance(latest_speed, float) and not np.isfinite(latest_speed):
            text = "speed=n/a cm/s"
        else:
            text = f"speed={latest_speed} cm/s"
        cv2.putText(canvas, text, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _nearest_detection(track, detections) -> tuple[int, float] | None:
    if not detections:
        return None
    positions = np.asarray([detection.center for detection in detections], dtype=np.float64)
    distances = np.linalg.norm(positions - np.asarray(track.position), axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = np.clip(magnitude / max(1.0, float(np.percentile(magnitude, 99))), 0, 1)
    hue = (angle * 180.0 / np.pi / 2.0).astype(np.uint8)
    saturation = np.full(hue.shape, 255, dtype=np.uint8)
    value = (magnitude * 255.0).astype(np.uint8)
    return cv2.cvtColor(np.stack((hue, saturation, value), axis=-1), cv2.COLOR_HSV2BGR)


def run_pipeline(
    config: PipelineConfig,
    on_progress: Callable[[PipelineProgress], None] | None = None,
) -> PipelineResult:
    if config.flow_every < 1 or config.stride < 1:
        raise ValueError("flow_every and stride must be positive")
    started = time.perf_counter()
    output = ensure_directory(config.output)
    total_frames, fps, frame_size = video_info(config.video)
    del total_frames
    capture = cv2.VideoCapture(str(config.video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {config.video}")
    calibration = load_calibration(config.calibration) if config.calibration else None
    homography = calibration.plane_homography if calibration else None
    recognizer = (
        RecognitionEngine(load_crops(config.recognition_root)) if config.recognition_root else None
    )
    tracker = MultiBoxTracker(
        association_distance=max(60.0, 0.15 * max(frame_size)),
        process_noise=2.0,
        measurement_noise=9.0,
        seed_distance=max(180.0, 0.45 * max(frame_size)),
        initial_velocity=config.belt_velocity,
    )
    rows: list[dict[str, object]] = []
    frames: list[np.ndarray] = []
    previous_frame: np.ndarray | None = None
    previous_index: int | None = None
    previous_detections: list = []
    latest_speed = float("nan")
    processed = 0
    flow_preview: Path | None = None
    try:
        while processed < config.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if frame_index % config.stride:
                continue
            raw_detections = production_detections(frame)
            detections = select_tracking_detections(
                raw_detections,
                previous_detections,
                keep_static=config.keep_static,
                has_previous_frame=previous_frame is not None,
            )
            detections = sorted(detections, key=lambda box: box.area, reverse=True)[: config.max_boxes]
            previous_detections = raw_detections
            measurements = [detection.center for detection in detections]
            tracks = tracker.step(measurements, 1.0 / max(fps, 1.0) * config.stride)
            if (
                previous_frame is not None
                and previous_index is not None
                and (processed - 1) % config.flow_every == 0
            ):
                flow = lucas_kanade_flow(previous_frame, frame)
                translation = median_flow_vector(flow)
                if not np.all(np.isfinite(translation)) or not np.any(translation):
                    matrix = np.asarray(fit_affine_flow(flow)["matrix"], dtype=np.float64)
                    if matrix.shape == (2, 3):
                        translation = matrix[:, 2]
                seconds = max(1e-6, (frame_index - previous_index) / max(fps, 1.0))
                center = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)
                latest_speed = estimate_speed(
                    center,
                    (float(translation[0]), float(translation[1])),
                    seconds,
                    plane_homography=homography,
                    cm_per_pixel=config.cm_per_pixel,
                )
                if processed == 1:
                    flow_preview = output / "flow_preview.png"
                    save_image(flow_preview, flow_to_color(flow))
            for track in tracks:
                nearest = _nearest_detection(track, detections)
                if nearest is None or nearest[1] > 100.0:
                    size_width, size_height = float("nan"), float("nan")
                    label = "unclassified"
                    detection = None
                else:
                    detection = detections[nearest[0]]
                    box = (detection.x, detection.y, detection.width, detection.height)
                    if homography is not None:
                        size_width, size_height = _world_size(homography, box)
                    else:
                        size_width, size_height = _fallback_size(box, config.cm_per_pixel)
                    if recognizer is not None:
                        prediction = recognizer.predict(_crop(frame, box))
                        label = str(prediction["label"])
                    else:
                        label = "unclassified"
                rows.append(
                    {
                        "frame": frame_index,
                        "track_id": track.track_id,
                        "size_width_cm": size_width,
                        "size_height_cm": size_height,
                        "speed_cm_s": latest_speed if np.isfinite(latest_speed) else "",
                        "type": label,
                        "raw_x": detection.center[0] if detection else None,
                        "raw_y": detection.center[1] if detection else None,
                        "filtered_x": float(track.position[0]),
                        "filtered_y": float(track.position[1]),
                    }
                )
                speed_text = f"{latest_speed:.3f}" if np.isfinite(latest_speed) else "n/a"
                size_text = (
                    f"{size_width:.2f}x{size_height:.2f}"
                    if np.isfinite(size_width) and np.isfinite(size_height)
                    else "n/a"
                )
                print(
                    f"ID={track.track_id} size_cm={size_text} "
                    f"speed_cm_s={speed_text} type={label}"
                )
            annotated = _draw(frame, detections, tracks, rows)
            if config.write_video:
                frames.append(annotated)
            previous_frame = frame
            previous_index = frame_index
            processed += 1
            if on_progress is not None:
                on_progress(
                    PipelineProgress(
                        processed=processed,
                        max_frames=config.max_frames,
                        frame_index=frame_index,
                        detections=len(detections),
                        tracks=len(tracks),
                        latest_speed=latest_speed,
                        preview=annotated,
                    )
                )
    finally:
        capture.release()
    annotated_video: Path | None = None
    if frames:
        annotated_video = output / "annotated.mp4"
        writer = cv2.VideoWriter(
            str(annotated_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, fps / max(1, config.stride)),
            (frames[0].shape[1], frames[0].shape[0]),
        )
        if writer.isOpened():
            for frame in frames:
                writer.write(frame)
        writer.release()
    results_csv: Path | None = None
    if rows:
        results_csv = output / "pipeline_results.csv"
        with results_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "frames_processed": processed,
        "records": len(rows),
        "calibrated": homography is not None,
        "recognition_enabled": recognizer is not None,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    with (output / "pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"Saved pipeline outputs to {output} "
        f"({summary['elapsed_seconds']:.2f}s for {processed} frames)"
    )
    return PipelineResult(
        output=output,
        frames_processed=processed,
        rows=rows,
        summary=summary,
        flow_preview=flow_preview,
        annotated_video=annotated_video,
        results_csv=results_csv,
    )


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        video=args.video,
        calibration=args.calibration,
        recognition_root=args.recognition_root,
        cm_per_pixel=args.cm_per_pixel,
        output=args.output,
        max_frames=args.max_frames,
        stride=args.stride,
        flow_every=args.flow_every,
        max_boxes=args.max_boxes,
        keep_static=args.keep_static,
        belt_velocity=tuple(args.belt_velocity) if args.belt_velocity else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete conveyor inspection pipeline.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--recognition-root", type=Path)
    parser.add_argument("--cm-per-pixel", type=float)
    parser.add_argument("--output", type=Path, default=Path("outputs/pipeline"))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--flow-every", type=int, default=5)
    parser.add_argument("--max-boxes", type=int, default=12)
    parser.add_argument(
        "--keep-static",
        action="store_true",
        help="Keep detections that do not move between frames (belt strips, shadows).",
    )
    parser.add_argument(
        "--belt-velocity",
        nargs=2,
        type=float,
        metavar=("VX", "VY"),
        help="Known belt velocity in px/s, used to seed the very first tracks.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.flow_every < 1 or args.stride < 1:
        raise SystemExit("--flow-every and --stride must be positive")
    run_pipeline(config_from_args(args))


if __name__ == "__main__":
    main()