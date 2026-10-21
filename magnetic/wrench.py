"""
Magnetic force and torque acting on the capsule permanent magnet.
"""

from __future__ import annotations

import numpy as np

from .dipole import total_field


DEFAULT_GRADIENT_STEP = 1e-5


def force_fixed_capsule_moment(
    p_capsule: np.ndarray,
    capsule_moment_world: np.ndarray,
    em_positions,
    em_axes,
    currents,
    em_gain: float,
    gradient_step: float = DEFAULT_GRADIENT_STEP,
) -> np.ndarray:
    """
    Compute magnetic force on a permanent magnetic dipole.

    The force relation is

        F = grad(m dot B)

    The capsule magnetic moment is held fixed while taking
    the spatial derivative.

    A central finite difference is used:

        df/dx ~= [f(x+h) - f(x-h)] / (2h)

    Parameters
    ----------
    p_capsule
        Capsule position in world coordinates [m].

    capsule_moment_world
        Capsule magnetic moment in world coordinates [A*m^2].

    em_positions
        Electromagnet positions.

    em_axes
        Electromagnet magnetic-axis directions.

    currents
        Signed electromagnet currents [A].

    em_gain
        Effective electromagnet dipole gain.

    gradient_step
        Finite-difference spatial step [m].

    Returns
    -------
    np.ndarray, shape (3,)
        Magnetic force in world coordinates [N].
    """
    if gradient_step <= 0.0:
        raise ValueError(
            "gradient_step must be positive."
        )

    p_capsule = np.asarray(
        p_capsule,
        dtype=float,
    )

    m_world = np.asarray(
        capsule_moment_world,
        dtype=float,
    )

    force = np.zeros(
        3,
        dtype=float,
    )

    for axis in range(3):
        dp = np.zeros(
            3,
            dtype=float,
        )

        dp[axis] = (
            gradient_step
        )

        B_plus = total_field(
            p_query=p_capsule + dp,
            em_positions=em_positions,
            em_axes=em_axes,
            currents=currents,
            em_gain=em_gain,
        )

        B_minus = total_field(
            p_query=p_capsule - dp,
            em_positions=em_positions,
            em_axes=em_axes,
            currents=currents,
            em_gain=em_gain,
        )

        energy_gradient = (
            np.dot(
                m_world,
                B_plus,
            )
            - np.dot(
                m_world,
                B_minus,
            )
        ) / (
            2.0
            * gradient_step
        )

        force[axis] = (
            energy_gradient
        )

    return force


def magnetic_wrench(
    p_capsule: np.ndarray,
    capsule_moment_world: np.ndarray,
    em_positions,
    em_axes,
    currents,
    em_gain: float,
    gradient_step: float = DEFAULT_GRADIENT_STEP,
):
    """
    Compute the complete magnetic wrench acting on the capsule.

    Force:
        F = grad(m dot B)

    Torque:
        tau = m x B

    Returns
    -------
    force : np.ndarray, shape (3,)
        Magnetic force [N].

    torque : np.ndarray, shape (3,)
        Magnetic torque [N*m].

    B_world : np.ndarray, shape (3,)
        Magnetic field at the capsule position [T].
    """
    m_world = np.asarray(
        capsule_moment_world,
        dtype=float,
    )

    B_world = total_field(
        p_query=p_capsule,
        em_positions=em_positions,
        em_axes=em_axes,
        currents=currents,
        em_gain=em_gain,
    )

    force = (
        force_fixed_capsule_moment(
            p_capsule=p_capsule,
            capsule_moment_world=m_world,
            em_positions=em_positions,
            em_axes=em_axes,
            currents=currents,
            em_gain=em_gain,
            gradient_step=gradient_step,
        )
    )

    torque = np.cross(
        m_world,
        B_world,
    )

    return (
        force,
        torque,
        B_world,
    )
