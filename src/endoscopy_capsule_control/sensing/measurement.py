"""
Measurement model for the DEMA-MCE hover simulation.

The sensor model is intentionally separated from the physical plant.

True MuJoCo state:
    -> remains untouched

Controller measurement:
    -> Gaussian position noise
    -> Gaussian theta_y noise
    -> discrete sample delay

The measurement residual is defined relative to the CURRENT true state,
so it contains both sensor noise and delay effects.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Measurement:
    """
    One measurement delivered to the controller.

    position_lcs
        Delayed noisy capsule position in DEMA LCS [m].

    theta_y
        Delayed noisy magnetic-moment tilt [rad].

    position_noise_lcs
        Noise originally added to the delayed position sample [m].

    theta_y_noise
        Noise originally added to the delayed orientation sample [rad].

    position_residual_lcs
        Delayed measurement - current true position [m].

    theta_y_residual
        Delayed measured theta_y - current true theta_y [rad].
    """

    position_lcs: np.ndarray
    theta_y: float

    position_noise_lcs: np.ndarray
    theta_y_noise: float

    position_residual_lcs: np.ndarray
    theta_y_residual: float


class MeasurementModel:
    """
    Gaussian-noise + discrete-delay measurement model.

    Noise is sampled whenever sample() is called. In the hover
    simulation this occurs at the controller rate (100 Hz).

    Parameters
    ----------
    position_noise_std
        Gaussian position standard deviation [m] per local axis.

    theta_y_noise_std
        Gaussian theta_y standard deviation [rad].

    delay_samples
        Measurement delay measured in controller samples.

        0 -> no delay
        1 -> one control-sample delay
        etc.

    seed
        NumPy random seed for reproducible simulations.
    """

    def __init__(
        self,
        *,
        position_noise_std: float = 0.0,
        theta_y_noise_std: float = 0.0,
        delay_samples: int = 0,
        seed: int = 1,
    ):
        if position_noise_std < 0.0:
            raise ValueError(
                "position_noise_std must be non-negative."
            )

        if theta_y_noise_std < 0.0:
            raise ValueError(
                "theta_y_noise_std must be non-negative."
            )

        if delay_samples < 0:
            raise ValueError(
                "delay_samples must be non-negative."
            )

        self.position_noise_std = float(
            position_noise_std
        )

        self.theta_y_noise_std = float(
            theta_y_noise_std
        )

        self.delay_samples = int(
            delay_samples
        )

        self.seed = int(
            seed
        )

        self._rng = np.random.default_rng(
            self.seed
        )

        self._buffer = deque(
            maxlen=self.delay_samples + 1
        )

    def reset(self) -> None:
        """
        Reset random generator and delay buffer.
        """

        self._rng = np.random.default_rng(
            self.seed
        )

        self._buffer.clear()

    @staticmethod
    def _angle_difference(
        angle_a: float,
        angle_b: float,
    ) -> float:
        """
        Wrapped angle difference angle_a - angle_b.
        """

        delta = (
            float(angle_a)
            - float(angle_b)
        )

        return float(
            np.arctan2(
                np.sin(delta),
                np.cos(delta),
            )
        )

    def sample(
        self,
        *,
        true_position_lcs: np.ndarray,
        true_theta_y: float,
    ) -> Measurement:
        """
        Generate one sensor sample and return the delayed measurement.

        The delay buffer stores noisy samples.

        With delay_samples = 1:

            k = 0:
                only one sample exists, so that sample is returned.

            k >= 1:
                sample k is appended,
                sample k-1 is delivered to the controller.

        This avoids inventing sensor history before simulation start.
        """

        true_position = np.asarray(
            true_position_lcs,
            dtype=float,
        ).reshape(3)

        true_theta = float(
            true_theta_y
        )

        # ----------------------------------------------------
        # Generate newest noisy sample
        # ----------------------------------------------------

        position_noise = self._rng.normal(
            loc=0.0,
            scale=self.position_noise_std,
            size=3,
        )

        theta_noise = float(
            self._rng.normal(
                loc=0.0,
                scale=self.theta_y_noise_std,
            )
        )

        noisy_position = (
            true_position
            + position_noise
        )

        noisy_theta = (
            true_theta
            + theta_noise
        )

        self._buffer.append(
            (
                noisy_position.copy(),
                float(noisy_theta),
                position_noise.copy(),
                float(theta_noise),
            )
        )

        # ----------------------------------------------------
        # Oldest available sample is delivered
        # ----------------------------------------------------

        (
            measured_position,
            measured_theta,
            delayed_position_noise,
            delayed_theta_noise,
        ) = self._buffer[0]

        measured_position = np.asarray(
            measured_position,
            dtype=float,
        ).copy()

        measured_theta = float(
            measured_theta
        )

        # ----------------------------------------------------
        # Residual against CURRENT true state
        #
        # Therefore this contains both:
        #
        #   sensor noise
        #   +
        #   delay effect
        # ----------------------------------------------------

        position_residual = (
            measured_position
            - true_position
        )

        theta_residual = (
            self._angle_difference(
                measured_theta,
                true_theta,
            )
        )

        return Measurement(
            position_lcs=(
                measured_position
            ),
            theta_y=(
                measured_theta
            ),
            position_noise_lcs=(
                np.asarray(
                    delayed_position_noise,
                    dtype=float,
                ).copy()
            ),
            theta_y_noise=float(
                delayed_theta_noise
            ),
            position_residual_lcs=(
                position_residual
            ),
            theta_y_residual=(
                theta_residual
            ),
        )