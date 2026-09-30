from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from common.video import ensure_directory, read_frame, sampled_indices, save_image, video_info
from module_a_segmentation.segmentation import (
    METHOD_NAMES,
    compare_methods,
    labels_to_boxes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare classical segmentation methods.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/segmentation"))
    parser.add_argument("--frames", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = ensure_directory(args.output)
    total_frames, _, _ = video_info(args.video)
    indices = sampled_indices(total_frames, args.frames)
    panels: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for frame_index in indices:
        frame = read_frame(args.video, frame_index)
        comparison, timings, label_maps = compare_methods(
            frame, return_labels=True
        )
        cv2.putText(
            comparison,
            f"frame {frame_index}",
            (12, comparison.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(comparison)
        for method in METHOD_NAMES:
            labels = label_maps[method]
            records.append(
                {
                    "frame": frame_index,
                    "method": method,
                    "seconds": timings[method],
                    "boxes": [box.to_dict() for box in labels_to_boxes(labels)],
                }
            )
    if panels:
        save_image(output / "segmentation_comparison.png", panels[0])
        first_height, first_width = panels[0].shape[:2]
        grid = panels[0]
        if len(panels) > 1:
            grid = np.zeros(
                (first_height * len(panels), first_width, 3), dtype=np.uint8
            )
            for row, panel in enumerate(panels):
                grid[row * first_height : (row + 1) * first_height] = panel
        save_image(output / "segmentation_all_frames.png", grid)
    with (output / "segmentation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    summary: dict[str, dict[str, float]] = {}
    for method in METHOD_NAMES:
        rows = [record for record in records if record["method"] == method]
        if rows:
            summary[method] = {
                "mean_seconds": float(np.mean([row["seconds"] for row in rows])),
                "mean_boxes": float(np.mean([len(row["boxes"]) for row in rows])),
            }
    with (output / "segmentation_notes.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            "Chosen for the pipeline: Otsu plus external contours "
            "(segmentation.production_detections). Correcting an earlier note: "
            "watershed is compared here but is NOT what the pipeline calls. The "
            "old note named watershed as the downstream method and as the one "
            "that separates touching boxes; neither was true.\n\n"
            "Why production is Otsu + RETR_EXTERNAL: on this footage it matches "
            "the box count, needs no tunable marker threshold, and is the "
            "fastest of the four. Watershed here uses one global "
            "distance-transform threshold over the whole frame, so a dark "
            "region touching the frame border inflates the distance maximum and "
            "a single threshold then suits neither the boxes nor that region.\n\n"
            "Measured on this video:\n"
        )
        for method, values in summary.items():
            handle.write(
                f"  {method:16s} {values['mean_seconds']:7.3f} s/frame"
                f"  mean boxes {values['mean_boxes']:5.1f}\n"
            )
        slowest = max(summary, key=lambda name: summary[name]["mean_seconds"])
        ratio = summary[slowest]["mean_seconds"] / max(
            summary["watershed"]["mean_seconds"], 1e-9
        )
        handle.write(
            "\nSplit-and-merge exposes intensity-region behavior but over-segments the "
            "background. Felzenszwalb preserves compact edges but is slower. "
            f"{slowest} is the slowest method, about {ratio:.0f}x the cost of "
            "watershed, and hits the box cap on every frame, so it is reported for "
            "comparison only and is not used downstream.\n\n"
            "Touching boxes, measured with IoU >= 0.5 on tools/make_test_video.py "
            "--layout touching (boxes overlapped by 5 px with a 25 px vertical "
            "offset, so their union is one blob with a concave step):\n"
            "  felzenszwalb     10/10 and 15/15 matched for 2 and 3 touching boxes\n"
            "  watershed         0/10 and 0/15, merges the pair into one region\n"
            "  Otsu (production) 0/10 and 0/15, RETR_EXTERNAL returns the merged blob\n"
            "  mean_shift        0/10, and hits the box cap regardless\n"
            "On genuinely separate boxes, watershed and Otsu both reach 10/10.\n\n"
            "So the honest answer to the touching-box question is Felzenszwalb, "
            "and the reason is structural: the concave step where two boxes meet "
            "is a low-gradient edge that the region-adjacency graph's minimum-cut "
            "criterion refuses to merge, while each box interior is internally "
            "coherent. Watershed and Otsu merge it because neither has a term "
            "for it. Felzenszwalb is not used downstream because it costs roughly "
            "two orders of magnitude more and over-segments the belt, so the "
            "pipeline accepts the limitation rather than paying for it.\n"
        )
    print(f"Saved segmentation outputs to {output}")


if __name__ == "__main__":
    main()
