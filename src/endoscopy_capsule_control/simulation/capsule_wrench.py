"""
External wrench application for the magnetic capsule.

MuJoCo handles gravity internally.

This module applies the additional physical effects:
- magnetic force and torque,
- buoyancy,
- translational fluid drag,
- rotational fluid drag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from endoscopy_capsule_control.plant.fluid import (
    buoyancy_force,
    rotational_drag,
    translational_drag,
)


@dataclass(frozen=True)
class CapsuleExternalWrench:
    """
    External forces and torques applied to the MCE.
    """

    magnetic_force_world: np.ndarray
    magnetic_torque_world: np.ndarray

    buoyancy_force_world: np.ndarray
    drag_force_world: np.ndarray
    drag_torque_world: np.ndarray

    total_force_world: np.ndarray
    total_torque_world: np.ndarray


def compute_capsule_external_wrench(
    *,
    magnetic_force_world: np.ndarray,
    magnetic_torque_world: np.ndarray,
    linear_velocity_world: np.ndarray,
    angular_velocity_world: np.ndarray,
    fluid_density: float,
    capsule_volume_value: float,
    gravity_world: np.ndarray,
    translational_drag_coefficient: float,
    rotational_damping: float,
) -> CapsuleExternalWrench:
    """
    Compute all non-gravitational external wrench components.

    Gravity is deliberately not included because MuJoCo already
    applies gravity to the capsule mass.
    """

    magnetic_force_world = np.asarray(
        magnetic_force_world,
        dtype=float,
    ).reshape(3)

    magnetic_torque_world = np.asarray(
        magnetic_torque_world,
        dtype=float,
    ).reshape(3)

    buoyancy = buoyancy_force(
        fluid_density=fluid_density,
        volume=capsule_volume_value,
        gravity_vector=gravity_world,
    )

    drag_force = translational_drag(
        velocity_world=linear_velocity_world,
        drag_coefficient=(
            translational_drag_coefficient
        ),
    )

    drag_torque = rotational_drag(
        angular_velocity_world=angular_velocity_world,
        rotational_damping=rotational_damping,
    )

    total_force = (
        magnetic_force_world
        + buoyancy
        + drag_force
    )

    total_torque = (
        magnetic_torque_world
        + drag_torque
    )

    return CapsuleExternalWrench(
        magnetic_force_world=(
            magnetic_force_world.copy()
        ),
        magnetic_torque_world=(
            magnetic_torque_world.copy()
        ),
        buoyancy_force_world=(
            buoyancy.copy()
        ),
        drag_force_world=(
            drag_force.copy()
        ),
        drag_torque_world=(
            drag_torque.copy()
        ),
        total_force_world=(
            total_force.copy()
        ),
        total_torque_world=(
            total_torque.copy()
        ),
    )


def apply_capsule_external_wrench(
    system,
    wrench: CapsuleExternalWrench,
) -> None:
    """
    Apply the external wrench to the MCE body.

    MuJoCo's xfrc_applied contains:

        [Fx, Fy, Fz, Tx, Ty, Tz]

    expressed in the world frame.
    """

    body_id = system.body_id(
        "MCE"
    )

    system.data.xfrc_applied[
        body_id,
        0:3,
    ] = wrench.total_force_world

    system.data.xfrc_applied[
        body_id,
        3:6,
    ] = wrench.total_torque_world