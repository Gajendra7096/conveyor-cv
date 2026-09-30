from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from module_b_calibration.camera_models import estimate_box_size, load_calibration


def _box(value: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError("A box must be x,y,width,height")
    return parts[0], parts[1], parts[2], parts[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate box dimensions under four camera models.")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--reference-box", type=_box, required=True)
    parser.add_argument("--reference-size", type=float, nargs=2, required=True, metavar=("WIDTH_CM", "HEIGHT_CM"))
    parser.add_argument("--unknown-box", type=_box, required=True)
    parser.add_argument(
        "--true-size",
        type=float,
        nargs=2,
        metavar=("WIDTH_CM", "HEIGHT_CM"),
        help=(
            "Known real size of the unknown box. When given, each of the four "
            "camera-model estimates is scored against it, so the recommendation is "
            "measured rather than asserted."
        ),
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _score(
    estimates: dict[str, tuple[float, float]],
    truth: tuple[float, float] | None,
) -> list[str]:
    """Report the four estimates, and score them against truth when it is known."""
    width_lines = []
    for name, (width, height) in estimates.items():
        line = f"  {name:18s} {width:8.3f} cm x {height:8.3f} cm"
        if truth is not None:
            width_error = (width - truth[0]) / max(1e-9, truth[0]) * 100.0
            height_error = (height - truth[1]) / max(1e-9, truth[1]) * 100.0
            line += f"   width {width_error:+6.2f}%   height {height_error:+6.2f}%"
        width_lines.append(line)
    print("\nReal-world size of the unknown box under each camera model:")
    print("\n".join(width_lines))
    if truth is None:
        print(
            "\nPass --true-size WIDTH_CM HEIGHT_CM to score all four models against a "
            "measured size."
        )
        return []
    print(f"\nTruth: {truth[0]:.3f} cm x {truth[1]:.3f} cm. Ranked by mean absolute error:")
    scored = sorted(
        estimates.items(),
        key=lambda item: (
            abs(item[1][0] - truth[0]) / max(1e-9, truth[0])
            + abs(item[1][1] - truth[1]) / max(1e-9, truth[1])
        ),
    )
    for name, (width, height) in scored:
        error = 0.5 * (
            abs(width - truth[0]) / max(1e-9, truth[0]) + abs(height - truth[1]) / max(1e-9, truth[1])
        ) * 100.0
        print(f"  {name:18s} mean abs error {error:6.2f}%")
    best = scored[0][0]
    print(
        f"\nMost accurate for THIS capture: {best}. Which model wins is a property of"
        "\nthe geometry, not a fixed ordering. A short focal length (wide field of view)"
        "\nor a close camera makes strong perspective, so full perspective is the only"
        "\nmodel that stays correct. A long focal length (narrow field of view) or a far"
        "\ncamera flattens the projection towards orthographic, and then the orthographic"
        "\nestimate is already within a few percent while the perspective models add"
        "\nparameters they cannot constrain. When a plane homography is available it is"
        "\nderived from the calibration rather than from a reference box, so it does not"
        "\ninherit the reference box's placement error, and that is the estimate to trust."
    )
    return [name for name, _ in scored]


def main() -> None:
    args = build_parser().parse_args()
    homography = None
    if args.calibration:
        calibration = load_calibration(args.calibration)
        homography = calibration.plane_homography
    estimates = estimate_box_size(
        args.reference_box,
        tuple(args.reference_size),
        args.unknown_box,
        homography,
    )
    if args.image:
        image = cv2.imread(str(args.image))
        if image is not None:
            for name, box in (("reference", args.reference_box), ("unknown", args.unknown_box)):
                x, y, width, height = (int(value) for value in box)
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0) if name == "reference" else (0, 128, 255), 2)
                cv2.putText(image, name, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            args.output = args.output or args.image.with_name(args.image.stem + "_measured.png")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output), image)
    print(json.dumps(estimates, indent=2))
    _score(estimates, tuple(args.true_size) if args.true_size else None)


if __name__ == "__main__":
    main()
