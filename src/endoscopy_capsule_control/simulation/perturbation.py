"""
Utilities for applying controlled perturbations to the capsule.

These functions are intended for local stability and recovery tests
around the nominal hover operating point.

They do NOT represent a general global positioning controller.
"""

from __future__ import annotations

import numpy as np

from .capsule_state import (
    get_capsule_state,
)

from .dema_geometry import (
    get_dema_geometry,
    lcs_to_world_position,
)

from .initialization import (
    set_free_joint_pose,
)

from .mujoco_system import (
    MujocoSystem,
)


# ============================================================
# ROTATION UTILITIES
# ============================================================


def rotation_y(
    angle_rad: float,
) -> np.ndarray:
    """
    Return a rotation matrix about local +Y.

    Parameters
    ----------
    angle_rad
        Rotation angle [rad].

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """

    c = np.cos(
        angle_rad
    )

    s = np.sin(
        angle_rad
    )

    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def rotation_matrix_to_quaternion_wxyz(
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion.

    MuJoCo quaternion convention:

        [w, x, y, z]

    Parameters
    ----------
    rotation
        3x3 rotation matrix.

    Returns
    -------
    np.ndarray
        Normalized quaternion [w, x, y, z].
    """

    R = np.asarray(
        rotation,
        dtype=float,
    ).reshape(
        3,
        3,
    )

    trace = float(
        np.trace(R)
    )

    if trace > 0.0:
        s = np.sqrt(
            trace + 1.0
        ) * 2.0

        qw = 0.25 * s

        qx = (
            R[2, 1]
            - R[1, 2]
        ) / s

        qy = (
            R[0, 2]
            - R[2, 0]
        ) / s

        qz = (
            R[1, 0]
            - R[0, 1]
        ) / s

    elif (
        R[0, 0] > R[1, 1]
        and R[0, 0] > R[2, 2]
    ):
        s = np.sqrt(
            1.0
            + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        ) * 2.0

        qw = (
            R[2, 1]
            - R[1, 2]
        ) / s

        qx = 0.25 * s

        qy = (
            R[0, 1]
            + R[1, 0]
        ) / s

        qz = (
            R[0, 2]
            + R[2, 0]
        ) / s

    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2.0

        qw = (
            R[0, 2]
            - R[2, 0]
        ) / s

        qx = (
            R[0, 1]
            + R[1, 0]
        ) / s

        qy = 0.25 * s

        qz = (
            R[1, 2]
            + R[2, 1]
        ) / s

    else:
        s = np.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2.0

        qw = (
            R[1, 0]
            - R[0, 1]
        ) / s

        qx = (
            R[0, 2]
            + R[2, 0]
        ) / s

        qy = (
            R[1, 2]
            + R[2, 1]
        ) / s

        qz = 0.25 * s

    quaternion = np.array(
        [
            qw,
            qx,
            qy,
            qz,
        ],
        dtype=float,
    )

    quaternion_norm = np.linalg.norm(
        quaternion
    )

    if quaternion_norm <= 1e-12:
        raise ValueError(
            "Cannot convert invalid rotation matrix "
            "to quaternion."
        )

    quaternion /= (
        quaternion_norm
    )

    return quaternion


# ============================================================
# CAPSULE PERTURBATION
# ============================================================


def apply_initial_perturbation(
    *,
    system: MujocoSystem,
    magnetic_moment_body: np.ndarray,
    x_offset: float = 0.0,
    z_offset: float = 0.0,
    theta_y_offset: float = 0.0,
) -> None:
    """
    Apply a local perturbation around the current hover operating point.

    The perturbation is expressed in the DEMA local coordinate system.

    Parameters
    ----------
    system
        MuJoCo simulation system.

    magnetic_moment_body
        Capsule permanent magnetic moment expressed in the capsule
        body frame [A*m^2].

    x_offset
        Local-X displacement from the nominal hover position [m].

    z_offset
        Local-Z displacement from the nominal hover position [m].

    theta_y_offset
        Rotation of the capsule about DEMA local +Y [rad].

    Notes
    -----
    Example:

        x_offset = 0.001
        z_offset = 0.002
        theta_y_offset = np.deg2rad(1.0)

    means:

        x0 = x_hover + 1 mm
        z0 = z_hover + 2 mm
        theta_y0 = theta_hover + 1 deg

    This function is meant for LOCAL recovery tests and should not
    be interpreted as arbitrary global capsule initialization.
    """

    # --------------------------------------------------------
    # Nominal operating-point geometry
    # --------------------------------------------------------

    geometry = get_dema_geometry(
        system
    )

    nominal_state = get_capsule_state(
        system=system,
        lcs_pose=(
            geometry.lcs_pose
        ),
        magnetic_moment_body=(
            magnetic_moment_body
        ),
    )

    # --------------------------------------------------------
    # Desired magnetic-center position in DEMA LCS
    # --------------------------------------------------------

    target_position_lcs = (
        nominal_state.position_lcs
        + np.array(
            [
                float(x_offset),
                0.0,
                float(z_offset),
            ],
            dtype=float,
        )
    )

    target_magnetic_center_world = (
        lcs_to_world_position(
            target_position_lcs,
            geometry.lcs_pose,
        )
    )

    # --------------------------------------------------------
    # Desired capsule orientation
    #
    # The perturbation is a rotation about DEMA local +Y.
    # --------------------------------------------------------

    R_lcs_body_nominal = (
        nominal_state
        .rotation_lcs_body
    )

    R_lcs_body_target = (
        rotation_y(
            float(
                theta_y_offset
            )
        )
        @ R_lcs_body_nominal
    )

    R_world_body_target = (
        geometry.lcs_pose.rotation
        @ R_lcs_body_target
    )

    # --------------------------------------------------------
    # MCE magnetic-center site may not coincide exactly with
    # the MCE body origin.
    #
    # Preserve that body-fixed offset when changing pose.
    # --------------------------------------------------------

    body_pose_nominal = (
        system.get_body_pose(
            "MCE"
        )
    )

    magnetic_center_offset_body = (
        body_pose_nominal.rotation.T
        @ (
            nominal_state.position_world
            - body_pose_nominal.position
        )
    )

    target_body_position_world = (
        target_magnetic_center_world
        - R_world_body_target
        @ magnetic_center_offset_body
    )

    # --------------------------------------------------------
    # Rotation matrix -> MuJoCo quaternion
    # --------------------------------------------------------

    target_quaternion_wxyz = (
        rotation_matrix_to_quaternion_wxyz(
            R_world_body_target
        )
    )

    # --------------------------------------------------------
    # Set MCE free-joint state
    #
    # set_free_joint_pose() also zeros the capsule velocity.
    # --------------------------------------------------------

    set_free_joint_pose(
        system=system,
        joint_name="MCE_free",
        position_world=(
            target_body_position_world
        ),
        quaternion_wxyz=(
            target_quaternion_wxyz
        ),
    )