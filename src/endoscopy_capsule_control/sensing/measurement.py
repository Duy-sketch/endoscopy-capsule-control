"""
Measurement model for capsule localization.

The model supports:
- Gaussian position noise
- Gaussian theta_y noise
- discrete measurement delay

Noise is sampled once per controller update.

For delay_samples = 1:
    the controller receives the previous measurement sample.

For delay_samples = 0:
    the controller receives the newest measurement immediately.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Measurement:
    """
    Measurement returned to the controller.
    """

    position_local: np.ndarray
    theta_y: float

    position_noise: np.ndarray
    theta_y_noise: float

    position_residual: np.ndarray
    theta_y_residual: float


class MeasurementModel:
    """
    Noisy and delayed capsule measurement model.
    """

    def __init__(
        self,
        position_noise_std: float,
        theta_y_noise_std: float,
        delay_samples: int = 0,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        position_noise_std
            Gaussian position-noise standard deviation [m]
            applied independently to local X, Y and Z.

        theta_y_noise_std
            Gaussian theta_y noise standard deviation [rad].

        delay_samples
            Number of controller samples of measurement delay.

            0 -> no delay
            1 -> one control-sample delay

        seed
            Random-number-generator seed.
        """
        self.position_noise_std = float(
            position_noise_std
        )

        self.theta_y_noise_std = float(
            theta_y_noise_std
        )

        self.delay_samples = int(
            delay_samples
        )

        if self.position_noise_std < 0.0:
            raise ValueError(
                "position_noise_std cannot be negative."
            )

        if self.theta_y_noise_std < 0.0:
            raise ValueError(
                "theta_y_noise_std cannot be negative."
            )

        if self.delay_samples < 0:
            raise ValueError(
                "delay_samples cannot be negative."
            )

        self.rng = np.random.default_rng(
            seed
        )

        self._buffer = deque(
            maxlen=self.delay_samples + 1
        )


    def reset(
        self,
        position_local: np.ndarray,
        theta_y: float,
    ) -> Measurement:
        """
        Initialize the measurement buffer.

        The first sample is generated from the supplied true state.
        """
        self._buffer.clear()

        sample = self._create_noisy_sample(
            position_local=position_local,
            theta_y=theta_y,
        )

        self._buffer.append(
            sample
        )

        return self._build_output(
            delayed_sample=sample,
            current_true_position=position_local,
            current_true_theta=theta_y,
        )


    def update(
        self,
        position_local: np.ndarray,
        theta_y: float,
    ) -> Measurement:
        """
        Generate a new sensor sample and return the delayed
        measurement seen by the controller.
        """
        position_local = np.asarray(
            position_local,
            dtype=float,
        ).reshape(3)

        theta_y = float(
            theta_y
        )

        newest_sample = (
            self._create_noisy_sample(
                position_local=position_local,
                theta_y=theta_y,
            )
        )

        self._buffer.append(
            newest_sample
        )

        delayed_sample = (
            self._buffer[0]
        )

        return self._build_output(
            delayed_sample=delayed_sample,
            current_true_position=position_local,
            current_true_theta=theta_y,
        )


    def _create_noisy_sample(
        self,
        position_local: np.ndarray,
        theta_y: float,
    ):
        """
        Generate one noisy measurement sample.
        """
        position_local = np.asarray(
            position_local,
            dtype=float,
        ).reshape(3)

        theta_y = float(
            theta_y
        )

        position_noise = self.rng.normal(
            loc=0.0,
            scale=self.position_noise_std,
            size=3,
        )

        theta_y_noise = float(
            self.rng.normal(
                loc=0.0,
                scale=self.theta_y_noise_std,
            )
        )

        measured_position = (
            position_local
            + position_noise
        )

        measured_theta_y = (
            theta_y
            + theta_y_noise
        )

        return (
            measured_position,
            measured_theta_y,
            position_noise,
            theta_y_noise,
        )


    def _build_output(
        self,
        delayed_sample,
        current_true_position,
        current_true_theta,
    ) -> Measurement:
        """
        Convert one buffered sample into the controller output.
        """
        (
            measured_position,
            measured_theta_y,
            position_noise,
            theta_y_noise,
        ) = delayed_sample

        current_true_position = np.asarray(
            current_true_position,
            dtype=float,
        ).reshape(3)

        current_true_theta = float(
            current_true_theta
        )

        # Includes both sensor noise and delay error.
        position_residual = (
            measured_position
            - current_true_position
        )

        theta_y_residual = (
            measured_theta_y
            - current_true_theta
        )

        return Measurement(
            position_local=np.asarray(
                measured_position,
                dtype=float,
            ).copy(),
            theta_y=float(
                measured_theta_y
            ),
            position_noise=np.asarray(
                position_noise,
                dtype=float,
            ).copy(),
            theta_y_noise=float(
                theta_y_noise
            ),
            position_residual=np.asarray(
                position_residual,
                dtype=float,
            ).copy(),
            theta_y_residual=float(
                theta_y_residual
            ),
        )