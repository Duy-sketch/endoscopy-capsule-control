"""
Runtime state extraction for the magnetic capsule.

This module reads the capsule state from MuJoCo and expresses it
both in the MuJoCo world frame and in the DEMA local coordinate
system.

It does not make assumptions about which coordinate is controlled.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from endoscopy_capsule_control.magnetic import (
    moment_world_from_body,
)

from .dema_geometry import (
    world_to_lcs_position,
    world_to_lcs_vector,
)

from .mujoco_system import (
    MujocoSystem,
    Pose,
)


@dataclass(frozen=True)
class CapsuleState:
    """
    Complete runtime state of the magnetic capsule.
    """

    # Magnetic-center position
    position_world: np.ndarray
    position_lcs: np.ndarray

    # Capsule orientation
    rotation_world_body: np.ndarray
    rotation_lcs_body: np.ndarray

    # Velocities
    linear_velocity_world: np.ndarray
    angular_velocity_world: np.ndarray

    linear_velocity_lcs: np.ndarray
    angular_velocity_lcs: np.ndarray

    # Permanent magnetic dipole moment
    magnetic_moment_body: np.ndarray
    magnetic_moment_world: np.ndarray


def _body_velocity_world(
    system: MujocoSystem,
    body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return angular and linear body velocity in world coordinates.

    MuJoCo mj_objectVelocity returns:

        [angular_velocity,
         linear_velocity]
    """

    body_id = system.body_id(
        body_name
    )

    velocity = np.zeros(
        6,
        dtype=float,
    )

    mujoco.mj_objectVelocity(
        system.model,
        system.data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity,
        0,  # world-frame result
    )

    angular_velocity = (
        velocity[0:3].copy()
    )

    linear_velocity = (
        velocity[3:6].copy()
    )

    return (
        angular_velocity,
        linear_velocity,
    )


def get_capsule_state(
    system: MujocoSystem,
    lcs_pose: Pose,
    magnetic_moment_body: np.ndarray,
) -> CapsuleState:
    """
    Read the current MCE state from MuJoCo.

    Current XML names:

        body: MCE
        site: MCE_magnetic_center

    Parameters
    ----------
    system
        MuJoCo simulation wrapper.

    lcs_pose
        Current DEMA local-coordinate-system pose.

    magnetic_moment_body
        Permanent capsule dipole moment expressed in the capsule
        body frame [A*m^2].
    """

    magnetic_moment_body = np.asarray(
        magnetic_moment_body,
        dtype=float,
    ).reshape(3)

    # --------------------------------------------------------
    # Position: magnetic center
    # --------------------------------------------------------

    magnetic_center_pose = (
        system.get_site_pose(
            "MCE_magnetic_center"
        )
    )

    position_world = (
        magnetic_center_pose.position.copy()
    )

    position_lcs = (
        world_to_lcs_position(
            position_world,
            lcs_pose,
        )
    )

    # --------------------------------------------------------
    # Orientation: MCE rigid body
    # --------------------------------------------------------

    body_pose = system.get_body_pose(
        "MCE"
    )

    rotation_world_body = (
        body_pose.rotation.copy()
    )

    rotation_lcs_body = (
        lcs_pose.rotation.T
        @ rotation_world_body
    )

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    (
        angular_velocity_world,
        linear_velocity_world,
    ) = _body_velocity_world(
        system=system,
        body_name="MCE",
    )

    linear_velocity_lcs = (
        world_to_lcs_vector(
            linear_velocity_world,
            lcs_pose,
        )
    )

    angular_velocity_lcs = (
        world_to_lcs_vector(
            angular_velocity_world,
            lcs_pose,
        )
    )

    # --------------------------------------------------------
    # Permanent magnetic moment
    # --------------------------------------------------------

    magnetic_moment_world = (
        moment_world_from_body(
            rotation_world_body,
            magnetic_moment_body,
        )
    )

    return CapsuleState(
        position_world=position_world,
        position_lcs=position_lcs,
        rotation_world_body=rotation_world_body,
        rotation_lcs_body=rotation_lcs_body,
        linear_velocity_world=linear_velocity_world,
        angular_velocity_world=angular_velocity_world,
        linear_velocity_lcs=linear_velocity_lcs,
        angular_velocity_lcs=angular_velocity_lcs,
        magnetic_moment_body=(
            magnetic_moment_body.copy()
        ),
        magnetic_moment_world=(
            magnetic_moment_world.copy()
        ),
    )