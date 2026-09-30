from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ConstantVelocityKalman:
    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 4.0,
        initial_position: tuple[float, float] | None = None,
        velocity: tuple[float, float] = (0.0, 0.0),
        velocity_std: float = 10.0,
        position_std: float = 5.0,
    ) -> None:
        self.state = np.zeros(4, dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64) * 100.0
        self.transition = np.eye(4, dtype=np.float64)
        self.observation = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )
        self.process_covariance = np.eye(4, dtype=np.float64) * float(process_noise)
        self.measurement_covariance = np.eye(2, dtype=np.float64) * float(measurement_noise)
        if initial_position is not None:
            self.initialize(initial_position, velocity, velocity_std, position_std)

    def initialize(
        self,
        position: tuple[float, float],
        velocity: tuple[float, float] = (0.0, 0.0),
        velocity_std: float = 10.0,
        position_std: float = 5.0,
    ) -> None:
        self.state[:] = [
            position[0],
            position[1],
            float(velocity[0]),
            float(velocity[1]),
        ]
        self.covariance = np.diag(
            [position_std, position_std, velocity_std, velocity_std]
        ).astype(np.float64)

    def predict(self, seconds: float) -> np.ndarray:
        dt = max(0.0, float(seconds))
        self.transition[0, 2] = dt
        self.transition[1, 3] = dt
        self.state = self.transition @ self.state
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T
            + self.process_covariance * max(dt, 1e-3)
        )
        return self.state.copy()

    def seed_velocity(self, velocity: tuple[float, float], velocity_std: float) -> None:
        self.state[2] = float(velocity[0])
        self.state[3] = float(velocity[1])
        std = max(1e-3, float(velocity_std))
        self.covariance[2, 2] = std * std
        self.covariance[3, 3] = std * std

    def update(self, measurement: tuple[float, float]) -> np.ndarray:
        observation = np.asarray(measurement, dtype=np.float64)
        innovation = observation - self.observation @ self.state
        innovation_covariance = (
            self.observation @ self.covariance @ self.observation.T
            + self.measurement_covariance
        )
        gain = (
            self.covariance
            @ self.observation.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.state = self.state + gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        residual = identity - gain @ self.observation
        self.covariance = (
            residual @ self.covariance @ residual.T
            + gain @ self.measurement_covariance @ gain.T
        )
        return self.state.copy()

    @property
    def position(self) -> np.ndarray:
        return self.state[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[2:].copy()


@dataclass
class KalmanStep:
    """One predict -> measure -> update cycle for a single track.

    The brief asks for the filter's three stages to be visible rather than implied,
    so the tracker records them instead of collapsing them into a new state.
    """

    track_id: int
    predicted: tuple[float, float]
    measurement: tuple[float, float] | None
    corrected: tuple[float, float]
    coasted: bool


@dataclass
class KalmanTrack:
    track_id: int
    kalman: ConstantVelocityKalman
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)

    def predict(self, seconds: float) -> np.ndarray:
        self.age += 1
        self.time_since_update += 1
        return self.kalman.predict(seconds)

    def update(self, measurement: tuple[float, float]) -> None:
        self.kalman.update(measurement)
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.kalman.position)

    @property
    def position(self) -> np.ndarray:
        return self.kalman.position

    @property
    def velocity(self) -> np.ndarray:
        return self.kalman.velocity


class MultiBoxTracker:
    def __init__(
        self,
        max_coasting_frames: int = 30,
        association_distance: float = 80.0,
        process_noise: float = 1.0,
        measurement_noise: float = 4.0,
        seed_distance: float | None = None,
        initial_velocity: tuple[float, float] | None = None,
    ) -> None:
        self.max_coasting_frames = max_coasting_frames
        self.association_distance = association_distance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.seed_distance = (
            float(association_distance) if seed_distance is None else float(seed_distance)
        )
        # A velocity seeded from two noisy positions inherits roughly sqrt(2) times
        # the position noise, expressed per second.
        self.velocity_std = float(np.sqrt(2.0 * measurement_noise))
        self.initial_velocity = initial_velocity
        self.tracks: list[KalmanTrack] = []
        self.next_id = 1
        self.previous_measurements: list[tuple[float, float]] = []
        self.last_step: list[KalmanStep] = []

    def _seed_velocity(
        self,
        position: tuple[float, float],
        seconds: float,
    ) -> tuple[tuple[float, float] | None, float]:
        """Estimate a new track velocity from the previous frame's nearest detection.

        The gate is deliberately wider than the association gate: a new track is
        created precisely because the object broke association, so an object moving
        several association widths per frame still gets a usable initial velocity.
        A caller-supplied `initial_velocity` covers the very first frame, where no
        previous measurement exists yet.
        """
        if not self.previous_measurements or seconds <= 0.0:
            return None, self.velocity_std
        previous = np.asarray(self.previous_measurements, dtype=np.float64)
        distances = np.linalg.norm(previous - np.asarray(position, dtype=np.float64), axis=1)
        index = int(np.argmin(distances))
        if distances[index] > self.seed_distance:
            return None, self.velocity_std
        velocity = (
            (float(position[0]) - float(previous[index][0])) / seconds,
            (float(position[1]) - float(previous[index][1])) / seconds,
        )
        velocity_std = self.velocity_std / seconds
        return velocity, velocity_std

    def step(
        self,
        measurements: list[tuple[float, float]],
        seconds: float,
    ) -> list[KalmanTrack]:
        for track in self.tracks:
            track.predict(seconds)
        predicted: dict[int, tuple[float, float]] = {
            track.track_id: (float(track.position[0]), float(track.position[1]))
            for track in self.tracks
        }
        unmatched = list(range(len(measurements)))
        pairs: list[tuple[int, int]] = []
        if self.tracks and unmatched:
            predicted_array = np.asarray([track.position for track in self.tracks])
            observed = np.asarray(measurements)
            costs = np.linalg.norm(predicted_array[:, None, :] - observed[None, :, :], axis=2)
            while costs.size:
                track_index, measurement_index = np.unravel_index(
                    int(np.argmin(costs)), costs.shape
                )
                if costs[track_index, measurement_index] > self.association_distance:
                    break
                pairs.append((int(track_index), int(measurement_index)))
                costs[track_index, :] = np.inf
                costs[:, measurement_index] = np.inf
        matched_measurements = {measurement_index for _, measurement_index in pairs}
        applied: dict[int, tuple[float, float]] = {}
        for track_index, measurement_index in pairs:
            track = self.tracks[track_index]
            track.update(measurements[measurement_index])
            applied[track.track_id] = (
                float(measurements[measurement_index][0]),
                float(measurements[measurement_index][1]),
            )
        for measurement_index in unmatched:
            if measurement_index in matched_measurements:
                continue
            x, y = measurements[measurement_index]
            velocity, velocity_std = self._seed_velocity((x, y), seconds)
            if velocity is None and self.initial_velocity is not None:
                velocity = self.initial_velocity
            self.tracks.append(
                KalmanTrack(
                    track_id=self.next_id,
                    kalman=ConstantVelocityKalman(
                        process_noise=self.process_noise,
                        measurement_noise=self.measurement_noise,
                        initial_position=(x, y),
                        velocity=velocity if velocity is not None else (0.0, 0.0),
                        velocity_std=velocity_std,
                    ),
                    age=1,
                    hits=1,
                )
            )
            # A new track has no prior state to extrapolate from, so its cycle starts
            # at the measurement it was born on rather than showing a fake prediction.
            applied[self.next_id] = (float(x), float(y))
            self.next_id += 1
        self.tracks = [
            track
            for track in self.tracks
            if track.time_since_update <= self.max_coasting_frames
        ]
        self.last_step = [
            KalmanStep(
                track_id=track.track_id,
                predicted=predicted.get(
                    track.track_id,
                    applied.get(track.track_id, (float(track.position[0]), float(track.position[1]))),
                ),
                measurement=applied.get(track.track_id),
                corrected=(float(track.position[0]), float(track.position[1])),
                coasted=track.track_id not in applied,
            )
            for track in self.tracks
        ]
        self.previous_measurements = list(measurements)
        return list(self.tracks)
