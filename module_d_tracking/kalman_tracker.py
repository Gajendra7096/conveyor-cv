from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from common.video import ensure_directory, video_info, write_video
from module_a_segmentation.segmentation import production_detections, select_tracking_detections
from module_d_tracking.kalman import MultiBoxTracker

COLORS = (
    (0, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (200, 200, 200),
)


def _plot_trajectory(rows: list[dict[str, object]], output: Path, dropout_start: int, dropout_length: int) -> None:
    import matplotlib.pyplot as plt

    first_id = rows[0]["track_id"] if rows else None
    selected = [row for row in rows if row["track_id"] == first_id]
    figure, axis = plt.subplots(figsize=(10, 5))
    if selected:
        raw_x = [float(row["raw_x"]) for row in selected if row["raw_x"] is not None]
        raw_y = [float(row["raw_y"]) for row in selected if row["raw_x"] is not None]
        filtered_x = [float(row["filtered_x"]) for row in selected]
        filtered_y = [float(row["filtered_y"]) for row in selected]
        if raw_x:
            axis.scatter(raw_x, raw_y, c="tab:orange", label="raw measurements", s=22)
        axis.plot(filtered_x, filtered_y, c="tab:blue", label="Kalman trajectory", linewidth=2)
        axis.set_xlabel("x pixels")
        axis.set_ylabel("y pixels")
        axis.set_title(f"Track {first_id}; simulated dropout {dropout_start}..{dropout_start + dropout_length - 1}")
        axis.legend()
        axis.grid(alpha=0.25)
        axis.invert_yaxis()
        figure.tight_layout()
        figure.savefig(output, dpi=150)
    plt.close(figure)


def _noisy(
    centers: list[tuple[float, float]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Add the measurement noise a real segmenter produces.

    The Kalman filter only earns its place when the measurements are noisy; on
    noise-free synthetic detections predict, measure and correct coincide, which
    hides the whole point of the module. The generator is seeded, so a run is
    reproducible.
    """
    if args.measurement_noise <= 0:
        return list(centers)
    offsets = rng.normal(0.0, args.measurement_noise, size=(len(centers), 2))
    return [
        (float(x) + float(dx), float(y) + float(dy))
        for (x, y), (dx, dy) in zip(centers, offsets)
    ]


def _print_cycle(step) -> None:
    """Print one predict -> measure -> update cycle in full.

    The brief asks for the three stages to be visible rather than implied, so all of
    them are shown: the extrapolation, the raw noisy segmentation measurement, and
    the corrected posterior. During a simulated dropout the measurement line is
    absent and the posterior equals the prediction, which is the observability
    demonstration - the linear motion model carries the state alone.
    """
    if step.measurement is None:
        measurement = "     no measurement (coasting on prediction)"
    else:
        measurement = f"({step.measurement[0]:8.2f}, {step.measurement[1]:8.2f})"
    print(
        f"    ID {step.track_id}  predict ({step.predicted[0]:8.2f}, {step.predicted[1]:8.2f})"
        f"  measure {measurement}"
        f"  correct ({step.corrected[0]:8.2f}, {step.corrected[1]:8.2f})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track box centroids with a custom Kalman filter.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/kalman"))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dropout-start", type=int, default=5)
    parser.add_argument("--dropout-length", type=int, default=4)
    parser.add_argument(
        "--belt-velocity",
        nargs=2,
        type=float,
        metavar=("VX", "VY"),
        help="Known belt velocity in px/s, used to seed the very first tracks.",
    )
    parser.add_argument(
        "--keep-static",
        action="store_true",
        help="Keep detections that do not move between frames (static background).",
    )
    parser.add_argument(
        "--cycle-frames",
        type=int,
        default=12,
        help=(
            "How many leading frames print the explicit predict/measure/update "
            "cycle. The brief asks for at least 10 consecutive frames."
        ),
    )
    parser.add_argument(
        "--measurement-noise",
        type=float,
        default=0.0,
        help=(
            "Gaussian sigma in pixels added to each segmentation measurement, to "
            "reproduce the noise a real segmenter produces. Deterministic per seed."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = ensure_directory(args.output)
    rng = np.random.default_rng(args.seed)
    total_frames, fps, frame_size = video_info(args.video)
    del total_frames
    tracker = MultiBoxTracker(
        association_distance=max(50.0, 0.15 * max(frame_size)),
        process_noise=2.0,
        measurement_noise=9.0,
        seed_distance=max(150.0, 0.45 * max(frame_size)),
        initial_velocity=tuple(args.belt_velocity) if args.belt_velocity else None,
    )
    frames: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    processed = 0
    frame_index = 0
    previous_detections: list = []
    has_previous_frame = False
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {args.video}")
    try:
        while processed < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            current_index = frame_index
            frame_index += 1
            if current_index % max(1, args.stride):
                continue
            raw_detections = production_detections(frame)
            detections = select_tracking_detections(
                raw_detections,
                previous_detections,
                keep_static=args.keep_static,
                has_previous_frame=has_previous_frame,
            )
            previous_detections = raw_detections
            has_previous_frame = True
            raw_centers = [detection.center for detection in detections]
            dropout = args.dropout_start <= processed < args.dropout_start + args.dropout_length
            measurements = [] if dropout else _noisy(raw_centers, args, rng)
            tracks = tracker.step(measurements, 1.0 / max(fps, 1.0) * max(1, args.stride))
            steps_by_id = {step.track_id: step for step in tracker.last_step}
            if processed < max(0, args.cycle_frames) and tracker.last_step:
                label = "simulated dropout" if dropout else "tracking"
                print(f"  frame {current_index} ({label}), dt = 1/{fps:.2f} s")
                for step in tracker.last_step:
                    _print_cycle(step)
            for track in tracks:
                matching_detection = None
                if not dropout:
                    matching_detection = next(
                        (
                            detection
                            for detection in detections
                            if np.linalg.norm(
                                np.asarray(detection.center) - track.position
                            )
                            < 80.0
                        ),
                        None,
                    )
                step = steps_by_id.get(track.track_id)
                rows.append(
                    {
                        "frame": current_index,
                        "processed_frame": processed,
                        "track_id": track.track_id,
                        "predicted_x": step.predicted[0] if step else None,
                        "predicted_y": step.predicted[1] if step else None,
                        "raw_x": matching_detection.center[0] if matching_detection else None,
                        "raw_y": matching_detection.center[1] if matching_detection else None,
                        "filtered_x": float(track.position[0]),
                        "filtered_y": float(track.position[1]),
                        "coasted": bool(step.coasted) if step else None,
                        "dropout": dropout,
                    }
                )
            canvas = frame.copy()
            for detection in detections:
                x, y = int(detection.x), int(detection.y)
                cv2.rectangle(
                    canvas,
                    (x, y),
                    (x + detection.width, y + detection.height),
                    (160, 160, 160),
                    1,
                )
                center = (
                    int(round(detection.center[0])),
                    int(round(detection.center[1])),
                )
                cv2.circle(canvas, center, 4, (160, 160, 160), -1)
            for track in tracks:
                color = COLORS[(track.track_id - 1) % len(COLORS)]
                point = tuple(int(round(value)) for value in track.position)
                cv2.circle(canvas, point, 6, color, 2)
                cv2.putText(
                    canvas,
                    f"ID {track.track_id}",
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                canvas,
                "simulated dropout" if dropout else "tracking",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if dropout else (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            frames.append(canvas)
            processed += 1
    finally:
        capture.release()
    if frames:
        write_video(output / "tracking_overlay.mp4", frames, fps / max(1, args.stride))
    if rows:
        with (output / "tracking_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        _plot_trajectory(rows, output / "tracking_plot.png", args.dropout_start, args.dropout_length)
    with (output / "tracking_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"frames_processed": processed, "tracks_created": tracker.next_id - 1, "dropout_start": args.dropout_start, "dropout_length": args.dropout_length}, handle, indent=2)
    print(f"Processed {processed} frames and saved tracking outputs to {output}")


if __name__ == "__main__":
    main()
