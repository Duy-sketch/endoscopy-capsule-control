"""
Permanent-magnet utilities for the magnetic capsule.

The capsule magnetic moment is body-fixed:

    m_world = R_world_body @ m_body

The permanent magnet does not instantaneously align with the
external magnetic field. Its orientation follows the rigid-body
dynamics of the capsule.
"""

from __future__ import annotations

import numpy as np


def unit(
    vector: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Return the normalized version of a vector.

    Raises
    ------
    ValueError
        If the vector magnitude is too small.
    """
    vector = np.asarray(
        vector,
        dtype=float,
    )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm < eps:
        raise ValueError(
            "Cannot normalize a near-zero vector."
        )

    return (
        vector
        / norm
    )


def moment_world_from_body(
    R_world_body: np.ndarray,
    moment_body: np.ndarray,
) -> np.ndarray:
    """
    Transform the capsule permanent-magnet moment from body frame
    to world frame.

    Parameters
    ----------
    R_world_body
        Rotation matrix from capsule body frame to world frame.

    moment_body
        Permanent-magnet moment expressed in capsule body frame
        [A*m^2].

    Returns
    -------
    np.ndarray, shape (3,)
        Magnetic moment expressed in world frame [A*m^2].

    Notes
    -----
    The moment direction convention is SOUTH -> NORTH.
    """
    R_world_body = np.asarray(
        R_world_body,
        dtype=float,
    ).reshape(
        3,
        3,
    )

    moment_body = np.asarray(
        moment_body,
        dtype=float,
    ).reshape(
        3,
    )

    return (
        R_world_body
        @ moment_body
    )


def magnetic_potential_energy(
    capsule_moment_world: np.ndarray,
    B_world: np.ndarray,
) -> float:
    """
    Magnetic potential energy of a permanent dipole.

        U = -m dot B
    """
    m_world = np.asarray(
        capsule_moment_world,
        dtype=float,
    )

    B_world = np.asarray(
        B_world,
        dtype=float,
    )

    return -float(
        np.dot(
            m_world,
            B_world,
        )
    )


def angle_between_deg(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Compute the angle between two vectors in degrees.

    Returns NaN if either vector is approximately zero.
    """
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    norm_a = float(
        np.linalg.norm(a)
    )

    norm_b = float(
        np.linalg.norm(b)
    )

    if (
        norm_a < 1e-12
        or norm_b < 1e-12
    ):
        return float("nan")

    cosine = (
        np.dot(
            a,
            b,
        )
        / (
            norm_a
            * norm_b
        )
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return float(
        np.rad2deg(
            np.arccos(
                cosine
            )
        )
    )


def magnetic_moment_tilt_y(magnetic_moment_lcs: np.ndarray) -> float:
    """Return capsule magnetic-moment tilt about DEMA local Y [rad].

    The nominal hover orientation has the magnetic moment along local +X.
    With the project's positive-Y convention:

        theta_y = atan2(-m_z, m_x)
    """

    moment = np.asarray(magnetic_moment_lcs, dtype=float).reshape(3)
    mx = float(moment[0])
    mz = float(moment[2])
    if np.hypot(mx, mz) <= 1e-12:
        raise ValueError("Magnetic moment has insufficient XZ-plane magnitude")
    return float(np.arctan2(-mz, mx))
