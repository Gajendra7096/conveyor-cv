"""Create a tiny synthetic clip for local smoke tests, not final submission footage."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/smoke.mp4"))
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--layout",
        choices=("separated", "touching"),
        default="separated",
        help=(
            "'separated' is two well-spaced boxes. 'touching' overlaps them by 5 px "
            "horizontally with a 25 px vertical offset, so their silhouettes merge "
            "into one blob with a concave step. That is the case the brief asks "
            "about, and it is measurably harder for every method."
        ),
    )
    parser.add_argument("--speed", type=int, default=8, help="Pixels moved per frame.")
    return parser


def _boxes(
    layout: str,
    index: int,
    speed: int,
) -> list[tuple[int, int, int, int, tuple[int, int, int]]]:
    if layout == "touching":
        first = 50 + index * speed
        return [
            (first, 115, 90, 120, (35, 80, 220)),
            (first + 85, 140, 115, 95, (40, 155, 65)),
        ]
    return [
        (50 + index * speed, 115, 90, 120, (35, 80, 220)),
        (310 + index * speed, 90, 115, 130, (40, 155, 65)),
    ]


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Unable to create {args.output}")
    for index in range(args.frames):
        frame = np.full((args.height, args.width, 3), 210, dtype=np.uint8)
        cv2.rectangle(frame, (0, 270), (args.width, args.height), (70, 70, 70), -1)
        for (x, y, box_width, box_height, color) in _boxes(args.layout, index, args.speed):
            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), color, -1)
        cv2.putText(
            frame,
            f"synthetic smoke test ({args.layout})",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        writer.write(frame)
    writer.release()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
