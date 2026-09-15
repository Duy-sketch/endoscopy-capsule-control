"""
Geometry extraction for the DEMA actuator.

All geometry is read directly from the MuJoCo model.

No electromagnet position or orientation is hard-coded here.
Therefore, if the AUBO robot moves, the DEMA geometry updates
automatically.

The magnetic model uses world-frame quantities:
- electromagnet center positions,
- electromagnet magnetic axes.

The DEMA local coordinate system (LCS) is also returned so that
capsule position and forces can later be expressed in DEMA-local
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from endoscopy_capsule_control.magnetic import unit

from .mujoco_system import (
    MujocoSystem,
    Pose,
)


@dataclass(frozen=True)
class ElectromagnetGeometry:
    """
    Geometry of one electromagnet.

    position_world
        Electromagnet magnetic-center position [m].

    axis_world
        Unit magnetic-axis vector in MuJoCo world coordinates.
    """

    position_world: np.ndarray
    axis_world: np.ndarray


@dataclass(frozen=True)
class DEMAGeometry:
    """
    Complete geometry of the dual-electromagnet actuator.
    """

    lcs_pose: Pose

    em1: ElectromagnetGeometry
    em2: ElectromagnetGeometry


def get_dema_geometry(
    system: MujocoSystem,
) -> DEMAGeometry:
    """
    Read the current DEMA geometry from MuJoCo.

    Site names are taken directly from the current AUBO-DEMA XML:

        LCS_origin
        EM1_center
        EM2_center

    The local +Z axis of each electromagnet-center site is used as
    its magnetic dipole axis, matching the original simulation.
    """

    # --------------------------------------------------------
    # DEMA local coordinate system
    # --------------------------------------------------------

    lcs_pose = system.get_site_pose(
        "LCS_origin"
    )

    # --------------------------------------------------------
    # Electromagnet 1
    # --------------------------------------------------------

    em1_pose = system.get_site_pose(
        "EM1_center"
    )

    em1_axis_world = unit(
        em1_pose.rotation[:, 2]
    )

    em1 = ElectromagnetGeometry(
        position_world=(
            em1_pose.position.copy()
        ),
        axis_world=(
            em1_axis_world.copy()
        ),
    )

    # --------------------------------------------------------
    # Electromagnet 2
    # --------------------------------------------------------

    em2_pose = system.get_site_pose(
        "EM2_center"
    )

    em2_axis_world = unit(
        em2_pose.rotation[:, 2]
    )

    em2 = ElectromagnetGeometry(
        position_world=(
            em2_pose.position.copy()
        ),
        axis_world=(
            em2_axis_world.copy()
        ),
    )

    return DEMAGeometry(
        lcs_pose=lcs_pose,
        em1=em1,
        em2=em2,
    )


def world_to_lcs_position(
    position_world: np.ndarray,
    lcs_pose: Pose,
) -> np.ndarray:
    """
    Transform a world-frame position into the DEMA local frame.

    R_world_lcs maps:

        v_lcs -> v_world

    Therefore:

        p_lcs =
            R_world_lcs.T
            @ (p_world - p_lcs_origin_world)
    """

    position_world = np.asarray(
        position_world,
        dtype=float,
    ).reshape(3)

    return (
        lcs_pose.rotation.T
        @ (
            position_world
            - lcs_pose.position
        )
    )


def lcs_to_world_position(
    position_local: np.ndarray,
    lcs_pose: Pose,
) -> np.ndarray:
    """
    Transform a DEMA-local position into the world frame.
    """

    position_local = np.asarray(
        position_local,
        dtype=float,
    ).reshape(3)

    return (
        lcs_pose.position
        + lcs_pose.rotation
        @ position_local
    )


def world_to_lcs_vector(
    vector_world: np.ndarray,
    lcs_pose: Pose,
) -> np.ndarray:
    """
    Transform a vector from world coordinates to DEMA-local
    coordinates.

    Unlike a position, vectors are not translated.
    """

    vector_world = np.asarray(
        vector_world,
        dtype=float,
    ).reshape(3)

    return (
        lcs_pose.rotation.T
        @ vector_world
    )


def lcs_to_world_vector(
    vector_local: np.ndarray,
    lcs_pose: Pose,
) -> np.ndarray:
    """
    Transform a vector from DEMA-local coordinates to world
    coordinates.
    """

    vector_local = np.asarray(
        vector_local,
        dtype=float,
    ).reshape(3)

    return (
        lcs_pose.rotation
        @ vector_local
    )