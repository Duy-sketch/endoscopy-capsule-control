"""
First-order coil-current dynamics.

The commanded current is not applied instantaneously.
The physical coil current follows the command through a
first-order response:

    dI/dt = (I_cmd - I_actual) / tau

The exact discrete-time update is:

    I[k+1] = I[k] + alpha * (I_cmd - I[k])

where

    alpha = 1 - exp(-dt / tau)
"""

from __future__ import annotations

import numpy as np


class FirstOrderCurrentDynamics:
    """
    First-order model of electromagnet current response.
    """

    def __init__(
        self,
        time_constant: float,
        dt: float,
        current_limit: float,
    ):
        """
        Parameters
        ----------
        time_constant
            Current-response time constant tau [s].

        dt
            Simulation integration time step [s].

        current_limit
            Absolute current limit [A].
        """
        self.time_constant = float(
            time_constant
        )

        self.dt = float(
            dt
        )

        self.current_limit = float(
            current_limit
        )

        if self.time_constant <= 0.0:
            raise ValueError(
                "time_constant must be positive."
            )

        if self.dt <= 0.0:
            raise ValueError(
                "dt must be positive."
            )

        if self.current_limit <= 0.0:
            raise ValueError(
                "current_limit must be positive."
            )

        self.alpha = (
            1.0
            - np.exp(
                -self.dt
                / self.time_constant
            )
        )


    def step(
        self,
        actual_current,
        commanded_current,
    ):
        """
        Advance the coil current by one simulation step.

        Parameters
        ----------
        actual_current
            Current coil current [A].

        commanded_current
            Requested current [A].

        Returns
        -------
        float or np.ndarray
            Updated actual current [A].
        """
        actual_current = np.asarray(
            actual_current,
            dtype=float,
        )

        commanded_current = np.asarray(
            commanded_current,
            dtype=float,
        )

        new_current = (
            actual_current
            + self.alpha
            * (
                commanded_current
                - actual_current
            )
        )

        new_current = np.clip(
            new_current,
            -self.current_limit,
            +self.current_limit,
        )

        if new_current.ndim == 0:
            return float(
                new_current
            )

        return new_current