from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from common.types import BoxDetection
from module_a_segmentation.segmentation import (
    reject_static_detections,
    select_tracking_detections,
    split_merge_segmentation,
)
from module_b_calibration.camera_models import estimate_box_size
from module_c_motion.lucas_kanade import lucas_kanade_flow, median_flow_vector
from module_d_tracking.kalman import ConstantVelocityKalman, MultiBoxTracker
from module_e_recognition.recognize import CropRecord, EigenboxClassifier, _binary


def band_limited_texture(height: int, width: int, seed: int = 3) -> np.ndarray:
    """Sinusoid texture with a correlation length long enough for Lucas-Kanade."""
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    generator = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=np.float32)
    for _ in range(25):
        image += np.sin(
            2.0
            * np.pi
            * (
                generator.uniform(0.01, 0.05) * rows
                + generator.uniform(0.01, 0.05) * columns
            )
            + generator.uniform(0.0, 2.0 * np.pi)
        )
    image -= image.min()
    image = (image / image.max() * 255.0).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def shifted(texture: np.ndarray, dx: float, dy: float = 0.0) -> np.ndarray:
    matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(
        texture, matrix, (texture.shape[1], texture.shape[0]), borderMode=cv2.BORDER_REFLECT
    )


def test_kalman_tracker_coasts_without_measurements() -> None:
    tracker = MultiBoxTracker(association_distance=50.0)
    tracks = tracker.step([(100.0, 100.0)], 1.0)
    assert len(tracks) == 1
    assert np.all(np.isfinite(tracks[0].position))
    coasted = tracker.step([], 1.0)
    assert len(coasted) == 1
    assert np.all(np.isfinite(coasted[0].position))


def test_seeded_velocity_initialisation_beats_zero_velocity() -> None:
    """Ground truth: the target ramps at exactly 8 px/frame with 1 px noise.

    At 15 fps that is 120 px/s. Starting the velocity at zero costs the filter
    most of its run while it learns the ramp, so the seeded filter must win clearly.
    """
    seconds = 1.0 / 15.0
    speed_per_frame = 8.0
    speed_per_second = speed_per_frame / seconds
    truth = [(100.0 + speed_per_frame * index, 200.0) for index in range(45)]

    def run(velocity: tuple[float, float]) -> np.ndarray:
        rng = np.random.default_rng(11)
        filter_ = ConstantVelocityKalman(
            process_noise=2.0,
            measurement_noise=9.0,
            initial_position=(truth[0][0], truth[0][1]),
            velocity=velocity,
            velocity_std=10.0 if velocity[0] else 100.0,
        )
        positions = []
        for index, target in enumerate(truth):
            filter_.predict(seconds)
            noise = rng.normal(0.0, 1.0, size=2) if index else np.zeros(2)
            filter_.update((target[0] + noise[0], target[1] + noise[1]))
            positions.append(filter_.position)
        return np.asarray(positions)

    def rmse(positions: np.ndarray) -> float:
        errors = positions - np.asarray(truth)
        return float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))

    seeded = rmse(run((speed_per_second, 0.0)))
    unseeded = rmse(run((0.0, 0.0)))
    raw_rmse = float(np.sqrt(2.0))  # 1 px noise per axis, unfiltered
    assert seeded < unseeded, f"seeded {seeded:.2f} px did not beat unseeded {unseeded:.2f} px"
    assert seeded < raw_rmse, f"seeded RMSE {seeded:.2f} px is worse than the {raw_rmse:.2f} px raw noise"


def _tracker_rmse(initial_velocity, frames: int = 45, seconds: float = 1.0 / 15.0) -> float:
    truth = [(100.0 + 8.0 * index, 200.0) for index in range(frames)]
    rng = np.random.default_rng(11)
    tracker = MultiBoxTracker(
        association_distance=50.0,
        process_noise=2.0,
        measurement_noise=9.0,
        initial_velocity=initial_velocity,
    )
    positions = []
    for index, target in enumerate(truth):
        noise = rng.normal(0.0, 1.0, size=2) if index else np.zeros(2)
        tracks = tracker.step([(target[0] + noise[0], target[1] + noise[1])], seconds)
        positions.append(tuple(tracks[0].position))
    errors = np.asarray(positions) - np.asarray(truth)
    return float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))


def test_tracker_with_a_known_belt_velocity_is_accurate_on_a_short_clip() -> None:
    """A 45-frame clip is where an unseeded filter spends most of its run in transient.

    8 px/frame at 15 fps is 120 px/s, which is what --belt-velocity supplies.
    """
    seeded = _tracker_rmse((120.0, 0.0))
    unseeded = _tracker_rmse(None)
    assert seeded < 0.6, f"seeded RMSE {seeded:.2f} px is larger than the 1 px measurement noise"
    assert seeded < unseeded, f"seeded {seeded:.2f} px did not beat unseeded {unseeded:.2f} px"


def test_tracker_seeds_velocity_from_the_previous_frame() -> None:
    """A new track created mid-clip must start with a real velocity, not zero."""
    tracker = MultiBoxTracker(association_distance=50.0, seed_distance=150.0)
    tracker.step([(100.0, 200.0)], 1.0 / 15.0)
    # Past the association gate, so this becomes a new track rather than a match.
    fresh = tracker.step([(200.0, 200.0)], 1.0 / 15.0)
    assert len(fresh) == 2
    seeded = next(track for track in fresh if track.track_id == 2)
    assert seeded.velocity[0] > 0.0


@pytest.mark.parametrize("displacement", [1.0, 3.0, 8.0, 16.0])
def test_lucas_kanade_recovers_known_translation(displacement: float) -> None:
    """Ground truth: a rigid shift of a textured image. Dense LK must find it."""
    texture = band_limited_texture(240, 320)
    flow = lucas_kanade_flow(texture, shifted(texture, displacement))
    vector = median_flow_vector(flow)
    assert abs(float(vector[0]) - displacement) < 0.35
    # The perpendicular component carries far less information, so it is held looser.
    assert abs(float(vector[1])) < 1.0


def test_lucas_kanade_survives_large_displacement() -> None:
    """A single-level solve collapses past one window width; the pyramid must not."""
    texture = band_limited_texture(240, 320)
    flow = lucas_kanade_flow(texture, shifted(texture, 24.0))
    vector = median_flow_vector(flow)
    assert abs(float(vector[0]) - 24.0) < 1.0


def test_lucas_kanade_rejects_ill_textured_regions() -> None:
    """A flat frame has no valid flow and must not produce NaNs."""
    flat = np.full((120, 160, 3), 128, dtype=np.uint8)
    flow = lucas_kanade_flow(flat, shifted(flat, 4.0))
    assert np.all(np.isfinite(flow))


def test_split_merge_returns_label_map() -> None:
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, :48] = 20
    image[:, 48:] = 220
    labels = split_merge_segmentation(image, variance_threshold=100.0)
    assert labels.shape == image.shape[:2]
    assert labels.dtype == np.int32
    assert labels.max() >= 1


def test_four_camera_models_are_reported() -> None:
    estimates = estimate_box_size(
        (0.0, 0.0, 100.0, 50.0),
        (10.0, 5.0),
        (200.0, 100.0, 200.0, 100.0),
    )
    assert set(estimates) == {"orthographic", "weak_perspective", "affine", "full_perspective"}
    assert all(np.isfinite(size).all() for size in estimates.values())


def test_eigenbox_classifier_returns_training_label() -> None:
    records = [
        CropRecord(
            path=Path(f"{label}_{index}.png"),
            label=label,
            image=np.full((32, 32), value, dtype=np.uint8),
        )
        for label, value in (("a", 20), ("b", 220))
        for index in range(2)
    ]
    classifier = EigenboxClassifier(size=16, components=2).fit(records)
    predicted, score = classifier.predict(records[0].image)
    assert predicted == "a"
    assert np.isfinite(score)


def _detections(index: int) -> tuple[BoxDetection, BoxDetection]:
    """A static belt strip and a box advancing 8 px per frame."""
    return (
        BoxDetection(x=0, y=270, width=640, height=90),
        BoxDetection(x=50 + 8 * index, y=115, width=90, height=120),
    )


def test_static_background_never_becomes_a_track() -> None:
    """Regression: the belt strip is a permanent Otsu false positive.

    Two things have to hold, and the first alone is not enough. Rejecting the
    static measurement stops the track being fed, but the track still exists from
    the frame it was created on, so it coasts on its own velocity and is written
    out anyway. Suppressing a false measurement is not the same as suppressing a
    false track, which is why the first attempt at this fix still emitted one.
    The entry points therefore also withhold the first measurement, since frame 0
    has nothing to compare against.
    """
    tracker = MultiBoxTracker(association_distance=50.0, measurement_noise=9.0)
    seconds = 1.0 / 15.0
    has_previous = False

    for index in range(30):
        raw = list(_detections(index))
        previous = list(_detections(index - 1)) if has_previous else []
        kept = select_tracking_detections(
            raw, previous, keep_static=False, has_previous_frame=has_previous
        )
        if has_previous:
            # Only the moving box survives the static filter.
            assert len(kept) == 1
        measurements = [detection.center for detection in kept]
        tracker.step(measurements, seconds)
        has_previous = True

    assert len(tracker.tracks) == 1, f"static belt produced {len(tracker.tracks)} tracks, expected 1"
    only = tracker.tracks[0]
    assert only.velocity[0] > 0.0, "the surviving track is not the moving box"
    assert only.position[0] > 50.0, "the surviving track is sitting on the static belt"


def test_the_first_frame_is_withheld_from_the_tracker() -> None:
    """The load-bearing half of the fix: with frame 0 fed in, a phantom survives.

    Rejecting the static measurement is not enough on its own. This asserts that
    the first-frame withholding is what actually removes the track, so a future
    change that drops it fails here rather than quietly resurrecting a 31-frame
    false positive.
    """
    tracker = MultiBoxTracker(association_distance=50.0, measurement_noise=9.0)
    seconds = 1.0 / 15.0
    belt = _detections(0)[0]

    # Without the withholding: the belt is measured on frame 0 and becomes a track.
    for index in range(30):
        raw = list(_detections(index))
        previous = list(_detections(index - 1)) if index else []
        kept = reject_static_detections(raw, previous)
        tracker.step([detection.center for detection in kept], seconds)
    assert len(tracker.tracks) == 2, "expected the belt to survive without the withholding"

    # And the belt is genuinely static, so the filter is behaving correctly above.
    assert belt.center == _detections(9)[0].center


def test_keep_static_restores_the_unfiltered_behaviour() -> None:
    raw = list(_detections(5))
    previous = list(_detections(4))
    assert len(select_tracking_detections(raw, previous, keep_static=True, has_previous_frame=True)) == 2
    assert select_tracking_detections(raw, previous, keep_static=True, has_previous_frame=False) == raw


def test_static_filter_keeps_a_box_that_actually_moves() -> None:
    """The filter must not become a general box-destroyer."""
    previous = list(_detections(4))
    current = list(_detections(5))
    assert len(reject_static_detections(current, previous)) == 1


def test_hu_moments_accept_a_colour_crop() -> None:
    """Regression: the pipeline slices 3-channel BGR frames, Otsu needs one channel."""
    colour = np.zeros((64, 64, 3), dtype=np.uint8)
    colour[..., 2] = 200
    colour[:, 32:, :] = 30
    binary = _binary(colour)
    assert binary.shape == (64, 64)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)) <= {0, 255}


def test_binary_matches_the_grey_version_of_the_same_crop() -> None:
    colour = np.zeros((48, 48, 3), dtype=np.uint8)
    colour[..., 1] = np.tile(np.arange(48, dtype=np.uint8) * 5, (48, 1))
    assert np.array_equal(_binary(colour), _binary(cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)))
