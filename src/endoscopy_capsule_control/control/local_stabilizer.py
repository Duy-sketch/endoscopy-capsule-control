"""
Local stabilization of the DEMA-MCE operating point.

This controller is NOT a task-level trajectory controller.

Its purpose is to stabilize the locally unstable coupled mode
involving:

    - capsule displacement along DEMA local X
    - capsule magnetic-moment tilt about DEMA local Y

The common current Ic provides authority over both Fx and tau_y.

Around the nominal operating point:

    Ic > 0  -> Fx > 0
    Ic > 0  -> tau_y < 0

Therefore the stabilization law is:

    Ic = -Kx * (x - x_ref)
         + Ktheta * theta_y

where theta_y is derived directly from the capsule magnetic
moment expressed in the DEMA local frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalStabilizerGains:
    """
    Gains for local X / orientation stabilization.

    kx
        Position feedback gain [A/m].

    ktheta
        Magnetic-moment tilt feedback gain [A/rad].
    """

    kx: float
    ktheta: float


def magnetic_moment_tilt_y(
    magnetic_moment_lcs: np.ndarray,
) -> float:
    """
    Compute capsule tilt about local Y from its magnetic moment.

    Nominal equilibrium:

        m_LCS = [+m, 0, 0]

    Using the standard positive-Y rotation convention:

        theta_y = atan2(-m_z, m_x)

    Returns
    -------
    float
        Tilt angle [rad].
    """

    moment = np.asarray(
        magnetic_moment_lcs,
        dtype=float,
    ).reshape(3)

    mx = float(
        moment[0]
    )

    mz = float(
        moment[2]
    )

    if np.hypot(
        mx,
        mz,
    ) <= 1e-12:
        raise ValueError(
            "Magnetic moment has insufficient XZ-plane magnitude."
        )

    return float(
        np.arctan2(
            -mz,
            mx,
        )
    )


class LocalStabilizer:
    """
    Common-current stabilizer for the local unstable mode.
    """

    def __init__(
        self,
        gains: LocalStabilizerGains,
        common_current_limit: float,
    ):
        self.gains = gains

        self.common_current_limit = float(
            common_current_limit
        )

        if self.common_current_limit <= 0.0:
            raise ValueError(
                "common_current_limit must be positive."
            )


    def compute(
        self,
        *,
        x_measured: float,
        magnetic_moment_lcs: np.ndarray,
        x_ref: float = 0.0,
    ) -> float:
        """
        Compute requested common current Ic.

        Parameters
        ----------
        x_measured
            Capsule local-X position [m].

        magnetic_moment_lcs
            Capsule magnetic moment in DEMA local coordinates
            [A*m^2].

        x_ref
            Desired local-X equilibrium position [m].

        Returns
        -------
        float
            Common-current command Ic [A].
        """

        x_error = (
            float(x_measured)
            - float(x_ref)
        )

        theta_y = (
            magnetic_moment_tilt_y(
                magnetic_moment_lcs
            )
        )

        ic_unsaturated = (
            -self.gains.kx
            * x_error
            + self.gains.ktheta
            * theta_y
        )

        return float(
            np.clip(
                ic_unsaturated,
                -self.common_current_limit,
                +self.common_current_limit,
            )
        )