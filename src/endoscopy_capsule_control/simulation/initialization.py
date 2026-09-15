"""
Initialization utilities for the AUBO-DEMA-MCE simulation.

The XML contains a generic/default model state.

The actual hover simulation starts from a predefined AUBO
working configuration chosen so that the DEMA local +Z axis
is approximately aligned with world -Z.
"""

from __future__ import annotations

import numpy as np

from .mujoco_system import MujocoSystem


AUBO_JOINT_NAMES = (
    "shoulder_joint",
    "upperArm_joint",
    "foreArm_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
)


AUBO_ACTUATOR_NAMES = (
    "J1_position",
    "J2_position",
    "J3_position",
    "J4_position",
    "J5_position",
    "J6_position",
)


AUBO_WORK_Q_DEG = np.array(
    [
        18.04001630,
        -1.46183237,
        54.74315387,
        -33.79438238,
        90.00022338,
        108.04120780,
    ],
    dtype=float,
)


AUBO_WORK_Q_RAD = np.deg2rad(
    AUBO_WORK_Q_DEG
)


def set_aubo_configuration(
    system: MujocoSystem,
    joint_positions: np.ndarray,
) -> None:
    """
    Set the six AUBO joint positions.

    The corresponding position-actuator commands are also updated
    so that the robot holds the requested configuration when the
    simulation starts stepping.
    """

    q = np.asarray(
        joint_positions,
        dtype=float,
    ).reshape(6)

    model = system.model
    data = system.data

    # --------------------------------------------------------
    # Set qpos
    # --------------------------------------------------------

    for joint_name, q_value in zip(
        AUBO_JOINT_NAMES,
        q,
    ):
        joint_id = system.joint_id(
            joint_name
        )

        qpos_address = int(
            model.jnt_qposadr[
                joint_id
            ]
        )

        data.qpos[
            qpos_address
        ] = q_value

    # --------------------------------------------------------
    # Zero AUBO joint velocities
    # --------------------------------------------------------

    for joint_name in AUBO_JOINT_NAMES:
        joint_id = system.joint_id(
            joint_name
        )

        dof_address = int(
            model.jnt_dofadr[
                joint_id
            ]
        )

        data.qvel[
            dof_address
        ] = 0.0

    # --------------------------------------------------------
    # Set position-actuator targets
    # --------------------------------------------------------

    for actuator_name, q_value in zip(
        AUBO_ACTUATOR_NAMES,
        q,
    ):
        actuator_id = system.actuator_id(
            actuator_name
        )

        data.ctrl[
            actuator_id
        ] = q_value

    system.forward()

def set_free_joint_pose(
    system: MujocoSystem,
    joint_name: str,
    position_world: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> None:
    """
    Set pose of a MuJoCo free joint.

    Parameters
    ----------
    system
        MuJoCo system.

    joint_name
        Name of the free joint.

    position_world
        World-frame position [m].

    quaternion_wxyz
        MuJoCo quaternion [w, x, y, z].
    """

    position_world = np.asarray(
        position_world,
        dtype=float,
    ).reshape(3)

    quaternion_wxyz = np.asarray(
        quaternion_wxyz,
        dtype=float,
    ).reshape(4)

    norm = np.linalg.norm(
        quaternion_wxyz
    )

    if norm <= 1e-12:
        raise ValueError(
            "Quaternion norm must be non-zero."
        )

    quaternion_wxyz = (
        quaternion_wxyz
        / norm
    )

    joint_id = system.joint_id(
        joint_name
    )

    qpos_address = int(
        system.model.jnt_qposadr[
            joint_id
        ]
    )

    dof_address = int(
        system.model.jnt_dofadr[
            joint_id
        ]
    )

    system.data.qpos[
        qpos_address:
        qpos_address + 3
    ] = position_world

    system.data.qpos[
        qpos_address + 3:
        qpos_address + 7
    ] = quaternion_wxyz

    # Free joint has 6 velocity DOFs.
    system.data.qvel[
        dof_address:
        dof_address + 6
    ] = 0.0

    system.forward()

def initialize_aubo_work_pose(
    system: MujocoSystem,
) -> None:
    """
    Move AUBO to the nominal DEMA hover working configuration.
    """

    set_aubo_configuration(
        system=system,
        joint_positions=AUBO_WORK_Q_RAD,
    )
def initialize_mce_hover_pose(
    system: MujocoSystem,
    hover_position_lcs: np.ndarray | None = None,
) -> None:
    """
    Place the MCE at the nominal hover operating point.

    Default operating position:

        p_MCE_LCS = [0, 0, 0.100] m

    The initial orientation preserves the original XML MCE
    orientation:

        quaternion = [sqrt(1/2), 0, sqrt(1/2), 0]

    With the current AUBO working pose this maps the capsule
    body +Z magnetic moment approximately onto DEMA local +X.
    """

    from .dema_geometry import (
        get_dema_geometry,
        lcs_to_world_position,
    )

    if hover_position_lcs is None:
        hover_position_lcs = np.array(
            [
                0.0,
                0.0,
                0.100,
            ],
            dtype=float,
        )

    hover_position_lcs = np.asarray(
        hover_position_lcs,
        dtype=float,
    ).reshape(3)

    geometry = get_dema_geometry(
        system
    )

    position_world = (
        lcs_to_world_position(
            hover_position_lcs,
            geometry.lcs_pose,
        )
    )

    # Original XML MCE orientation.
    quaternion_wxyz = np.array(
        [
            np.sqrt(0.5),
            0.0,
            np.sqrt(0.5),
            0.0,
        ],
        dtype=float,
    )

    set_free_joint_pose(
        system=system,
        joint_name="MCE_free",
        position_world=position_world,
        quaternion_wxyz=quaternion_wxyz,
    )
def initialize_hover_operating_point(
    system: MujocoSystem,
) -> None:
    """
    Initialize the complete nominal hover operating point.
    """

    initialize_aubo_work_pose(
        system
    )

    initialize_mce_hover_pose(
        system
    )   