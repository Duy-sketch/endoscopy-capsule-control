"""RF-localization measurement model for the capsule."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from endoscopy_capsule_control.config import LocalizationConfig


@dataclass(frozen=True)
class LocalizationMeasurement:
    """Measurement delivered to the controller at one control update."""

    position_lcs_m: np.ndarray
    theta_y_rad: float
    position_noise_m: np.ndarray
    theta_y_noise_rad: float
    position_residual_m: np.ndarray
    theta_y_residual_rad: float


class RFLocalizationModel:
    """Noisy, optionally delayed capsule localization model.

    The measurement is generated in the physical DEMA local frame.  The
    controller may then transform it to world coordinates using its
    reported robot kinematics.  This separation lets AUBO pose error and
    RF localization error be tested independently.
    """

    def __init__(
        self,
        *,
        config: LocalizationConfig,
        rng: np.random.Generator,
        noise_enabled: bool,
    ) -> None:
        self.config = config
        self.rng = rng
        self.noise_enabled = bool(noise_enabled)

        if self.config.position_noise_std_m < 0.0:
            raise ValueError("position_noise_std_m must be non-negative")
        if self.config.theta_y_noise_std_rad < 0.0:
            raise ValueError("theta_y_noise_std_rad must be non-negative")
        if self.config.delay_samples < 0:
            raise ValueError("delay_samples must be non-negative")

        self._buffer = deque(maxlen=self.config.delay_samples + 1)

    def reset(self) -> None:
        self._buffer.clear()

    @staticmethod
    def _angle_difference(a: float, b: float) -> float:
        d = float(a) - float(b)
        return float(np.arctan2(np.sin(d), np.cos(d)))

    def sample(
        self,
        *,
        true_position_lcs_m: np.ndarray,
        true_theta_y_rad: float,
    ) -> LocalizationMeasurement:
        true_position = np.asarray(true_position_lcs_m, dtype=float).reshape(3)
        true_theta = float(true_theta_y_rad)

        if self.noise_enabled:
            position_noise = self.rng.normal(
                0.0,
                self.config.position_noise_std_m,
                size=3,
            )
            theta_noise = float(
                self.rng.normal(0.0, self.config.theta_y_noise_std_rad)
            )
        else:
            position_noise = np.zeros(3, dtype=float)
            theta_noise = 0.0

        newest = (
            true_position + position_noise,
            true_theta + theta_noise,
            position_noise.copy(),
            theta_noise,
        )
        self._buffer.append(newest)

        measured_position, measured_theta, delayed_noise, delayed_theta_noise = (
            self._buffer[0]
        )

        measured_position = np.asarray(measured_position, dtype=float).copy()
        measured_theta = float(measured_theta)

        return LocalizationMeasurement(
            position_lcs_m=measured_position,
            theta_y_rad=measured_theta,
            position_noise_m=np.asarray(delayed_noise, dtype=float).copy(),
            theta_y_noise_rad=float(delayed_theta_noise),
            position_residual_m=measured_position - true_position,
            theta_y_residual_rad=self._angle_difference(
                measured_theta,
                true_theta,
            ),
        )
