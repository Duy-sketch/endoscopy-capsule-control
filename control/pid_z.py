"""
PID force controller for capsule Z-position regulation.

The controller generates a magnetic-force correction:

    F_pid = Kp * e
          + Ki * integral(e)
          + Kd * de/dt

where

    e = z_ref - z_measured

The derivative term is low-pass filtered.

Anti-windup is implemented using a two-stage update:

    1. prepare(...)
       Compute a candidate integral state and candidate force.

    2. finalize(accept_integral)
       Accept the candidate integral only if the downstream
       current allocator is not saturated.

This preserves the anti-windup logic used in the original
hover simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZPIDGains:
    """
    PID gains for force-based Z control.
    """

    kp: float
    ki: float
    kd: float


@dataclass(frozen=True)
class ZPIDCandidate:
    """
    Intermediate result of one PID update.

    This object is returned before the actuator saturation
    decision is known.
    """

    error: float
    error_rate_raw: float
    error_rate_filtered: float
    integral_candidate: float
    force_candidate: float


def design_z_pid_gains(
    mass: float,
    drag_coefficient: float,
    natural_frequency: float = 5.0,
    damping_ratio: float = 1.0,
    integral_pole: float = 1.0,
) -> ZPIDGains:
    """
    Compute the Z PID gains used in the hover simulation.

    Parameters
    ----------
    mass
        Capsule mass [kg].

    drag_coefficient
        Translational viscous-drag coefficient [N*s/m].

    natural_frequency
        Desired natural frequency [rad/s].

    damping_ratio
        Desired damping ratio.

    integral_pole
        Additional pole associated with integral action [1/s].

    Returns
    -------
    ZPIDGains
        kp, ki and kd.
    """
    mass = float(mass)
    drag_coefficient = float(
        drag_coefficient
    )

    natural_frequency = float(
        natural_frequency
    )

    damping_ratio = float(
        damping_ratio
    )

    integral_pole = float(
        integral_pole
    )

    if mass <= 0.0:
        raise ValueError(
            "mass must be positive."
        )

    if drag_coefficient < 0.0:
        raise ValueError(
            "drag_coefficient cannot be negative."
        )

    if natural_frequency <= 0.0:
        raise ValueError(
            "natural_frequency must be positive."
        )

    if damping_ratio <= 0.0:
        raise ValueError(
            "damping_ratio must be positive."
        )

    if integral_pole <= 0.0:
        raise ValueError(
            "integral_pole must be positive."
        )

    kd = (
        mass
        * (
            2.0
            * damping_ratio
            * natural_frequency
            + integral_pole
        )
        - drag_coefficient
    )

    kp = (
        mass
        * (
            natural_frequency**2
            + 2.0
            * damping_ratio
            * natural_frequency
            * integral_pole
        )
    )

    ki = (
        mass
        * integral_pole
        * natural_frequency**2
    )

    return ZPIDGains(
        kp=float(kp),
        ki=float(ki),
        kd=float(kd),
    )


class ZPIDController:
    """
    Force-based PID controller for capsule Z position.
    """

    def __init__(
        self,
        gains: ZPIDGains,
        dt: float,
        derivative_filter_tau: float = 0.030,
        integral_limit: float = 0.020,
        force_limit: float = 0.020,
    ):
        """
        Parameters
        ----------
        gains
            PID gains.

        dt
            Controller update period [s].

        derivative_filter_tau
            First-order derivative low-pass time constant [s].

        integral_limit
            Absolute integral-state limit [m*s].

        force_limit
            Absolute PID force-correction limit [N].
        """
        self.gains = gains

        self.dt = float(dt)

        self.derivative_filter_tau = float(
            derivative_filter_tau
        )

        self.integral_limit = float(
            integral_limit
        )

        self.force_limit = float(
            force_limit
        )

        if self.dt <= 0.0:
            raise ValueError(
                "dt must be positive."
            )

        if self.derivative_filter_tau < 0.0:
            raise ValueError(
                "derivative_filter_tau cannot be negative."
            )

        if self.integral_limit <= 0.0:
            raise ValueError(
                "integral_limit must be positive."
            )

        if self.force_limit <= 0.0:
            raise ValueError(
                "force_limit must be positive."
            )

        self.integral = 0.0
        self.previous_error = 0.0
        self.error_rate_filtered = 0.0

        self._candidate: ZPIDCandidate | None = None


    def reset(
        self,
        initial_error: float = 0.0,
    ) -> None:
        """
        Reset internal PID states.

        The initial error should normally be initialized from the
        first measured position. This avoids a derivative kick at
        simulation startup.
        """
        self.integral = 0.0

        self.previous_error = float(
            initial_error
        )

        self.error_rate_filtered = 0.0

        self._candidate = None


    def prepare(
        self,
        z_ref: float,
        z_measured: float,
    ) -> ZPIDCandidate:
        """
        Compute a candidate PID update.

        The integral state is NOT committed here because actuator
        saturation is not known yet.

        Returns
        -------
        ZPIDCandidate
            Candidate controller quantities.
        """
        error = (
            float(z_ref)
            - float(z_measured)
        )

        error_rate_raw = (
            error
            - self.previous_error
        ) / self.dt

        if self.derivative_filter_tau == 0.0:
            alpha_d = 1.0
        else:
            alpha_d = (
                self.dt
                / (
                    self.derivative_filter_tau
                    + self.dt
                )
            )

        self.error_rate_filtered += (
            alpha_d
            * (
                error_rate_raw
                - self.error_rate_filtered
            )
        )

        self.previous_error = error

        integral_candidate = float(
            np.clip(
                self.integral
                + error
                * self.dt,
                -self.integral_limit,
                +self.integral_limit,
            )
        )

        force_candidate = (
            self.gains.kp
            * error
            + self.gains.ki
            * integral_candidate
            + self.gains.kd
            * self.error_rate_filtered
        )

        force_candidate = float(
            np.clip(
                force_candidate,
                -self.force_limit,
                +self.force_limit,
            )
        )

        candidate = ZPIDCandidate(
            error=float(error),
            error_rate_raw=float(
                error_rate_raw
            ),
            error_rate_filtered=float(
                self.error_rate_filtered
            ),
            integral_candidate=float(
                integral_candidate
            ),
            force_candidate=float(
                force_candidate
            ),
        )

        self._candidate = candidate

        return candidate


    def finalize(
        self,
        accept_integral: bool,
    ) -> float:
        """
        Finalize one PID update.

        Parameters
        ----------
        accept_integral
            True when the downstream actuator/current allocator
            is not saturated.

            False freezes the integral state for anti-windup.

        Returns
        -------
        float
            Final PID force correction [N].
        """
        if self._candidate is None:
            raise RuntimeError(
                "prepare() must be called before finalize()."
            )

        candidate = self._candidate

        if accept_integral:
            self.integral = (
                candidate.integral_candidate
            )

        force = (
            self.gains.kp
            * candidate.error
            + self.gains.ki
            * self.integral
            + self.gains.kd
            * candidate.error_rate_filtered
        )

        force = float(
            np.clip(
                force,
                -self.force_limit,
                +self.force_limit,
            )
        )

        self._candidate = None

        return force