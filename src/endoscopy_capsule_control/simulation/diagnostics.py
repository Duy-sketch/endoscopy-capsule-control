"""
Diagnostics for hover simulations.

The diagnostics in this module operate on TRUE plant states.

They are intentionally separate from the measurement statistics:
sensor residuals describe the sensing system, while post-settling
metrics describe actual closed-loop capsule performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PostSettlingMetrics:
    """
    True-state statistics after the settling period.
    """

    settling_time: float
    samples: int

    mean_x: float
    mean_z_error: float
    mean_theta_y: float

    rms_x: float
    rms_z_error: float
    rms_theta_y: float

    std_x: float
    std_z_error: float
    std_theta_y: float


class PostSettlingDiagnostics:
    """
    Online statistics for:

        x
        z_error = z_ref - z
        theta_y

    Samples before settling_time are ignored.

    The full time history does not need to be stored.
    """

    def __init__(
        self,
        *,
        settling_time: float,
    ):
        if settling_time < 0.0:
            raise ValueError(
                "settling_time must be non-negative."
            )

        self.settling_time = float(
            settling_time
        )

        self.samples = 0

        self._sum = np.zeros(
            3,
            dtype=float,
        )

        self._sum_sq = np.zeros(
            3,
            dtype=float,
        )

    def reset(self) -> None:
        self.samples = 0

        self._sum[:] = 0.0
        self._sum_sq[:] = 0.0

    def update(
        self,
        *,
        time: float,
        x: float,
        z_error: float,
        theta_y: float,
    ) -> None:
        """
        Add one TRUE-state sample if the settling period has passed.
        """

        if (
            float(time)
            + 1e-12
            < self.settling_time
        ):
            return

        values = np.array(
            [
                float(x),
                float(z_error),
                float(theta_y),
            ],
            dtype=float,
        )

        self._sum += values

        self._sum_sq += (
            values**2
        )

        self.samples += 1

    def result(
        self,
    ) -> PostSettlingMetrics:
        """
        Build final statistics.
        """

        if self.samples <= 0:
            raise RuntimeError(
                "No post-settling samples were collected. "
                "Reduce settling_time or increase simulation_time."
            )

        n = float(
            self.samples
        )

        mean = (
            self._sum
            / n
        )

        mean_sq = (
            self._sum_sq
            / n
        )

        rms = np.sqrt(
            mean_sq
        )

        variance = np.maximum(
            mean_sq
            - mean**2,
            0.0,
        )

        std = np.sqrt(
            variance
        )

        return PostSettlingMetrics(
            settling_time=(
                self.settling_time
            ),
            samples=(
                self.samples
            ),
            mean_x=float(
                mean[0]
            ),
            mean_z_error=float(
                mean[1]
            ),
            mean_theta_y=float(
                mean[2]
            ),
            rms_x=float(
                rms[0]
            ),
            rms_z_error=float(
                rms[1]
            ),
            rms_theta_y=float(
                rms[2]
            ),
            std_x=float(
                std[0]
            ),
            std_z_error=float(
                std[1]
            ),
            std_theta_y=float(
                std[2]
            ),
        )