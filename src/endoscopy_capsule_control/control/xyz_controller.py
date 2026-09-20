"""
Coordinator for the new 3-DOF position-control architecture.

Actuator assignment:
    X/Y -> AUBO motion of the DEMA
    Z   -> DEMA differential current Id

No common-current lateral stabilizer is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aubo_xy_controller import AUBOXYController, AUBOXYControlOutput
from .capsule_xy_controller import (
    CapsuleXYController,
    CapsuleXYControlOutput,
)
from .z_hover_controller import (
    ZHoverController,
    ZHoverControlInput,
    ZHoverControlOutput,
)


@dataclass(frozen=True)
class XYZControlInput:
    capsule_position_world: np.ndarray
    capsule_position_lcs: np.ndarray
    magnetic_moment_world: np.ndarray

    dema_position_world: np.ndarray
    dema_rotation_world: np.ndarray

    em_positions_world: tuple[np.ndarray, np.ndarray]
    em_axes_world: tuple[np.ndarray, np.ndarray]

    target_capsule_position_world: np.ndarray


@dataclass(frozen=True)
class XYZControlOutput:
    current_command: np.ndarray
    dema_twist_world: np.ndarray

    desired_dema_position_world: np.ndarray
    target_z_lcs: float

    xy: CapsuleXYControlOutput
    aubo: AUBOXYControlOutput
    z: ZHoverControlOutput


class XYZController:
    """Coordinate AUBO X/Y control with magnetic Z hover control."""

    def __init__(
        self,
        *,
        xy_controller: CapsuleXYController,
        aubo_controller: AUBOXYController,
        z_controller: ZHoverController,
    ):
        self.xy_controller = xy_controller
        self.aubo_controller = aubo_controller
        self.z_controller = z_controller

        self._initialized = False
        self._dema_z_hold = 0.0
        self._dema_rotation_hold = np.eye(3, dtype=float)

    def reset(
        self,
        *,
        initial_capsule_position_world: np.ndarray,
        initial_capsule_position_lcs: np.ndarray,
        initial_dema_position_world: np.ndarray,
        initial_dema_rotation_world: np.ndarray,
    ) -> None:
        initial_capsule_world = np.asarray(
            initial_capsule_position_world,
            dtype=float,
        ).reshape(3)

        initial_capsule_lcs = np.asarray(
            initial_capsule_position_lcs,
            dtype=float,
        ).reshape(3)

        initial_dema_world = np.asarray(
            initial_dema_position_world,
            dtype=float,
        ).reshape(3)

        initial_dema_rotation = np.asarray(
            initial_dema_rotation_world,
            dtype=float,
        ).reshape(3, 3)

        self.xy_controller.reset(
            initial_capsule_position_world=initial_capsule_world,
            initial_dema_position_world=initial_dema_world,
        )

        self.z_controller.reset(
            initial_z=float(initial_capsule_lcs[2])
        )

        self._dema_z_hold = float(initial_dema_world[2])
        self._dema_rotation_hold = initial_dema_rotation.copy()
        self._initialized = True

    def update(
        self,
        control_input: XYZControlInput,
    ) -> XYZControlOutput:
        if not self._initialized:
            raise RuntimeError(
                "XYZController.reset() must be called before update()."
            )

        capsule_world = np.asarray(
            control_input.capsule_position_world,
            dtype=float,
        ).reshape(3)

        capsule_lcs = np.asarray(
            control_input.capsule_position_lcs,
            dtype=float,
        ).reshape(3)

        dema_world = np.asarray(
            control_input.dema_position_world,
            dtype=float,
        ).reshape(3)

        dema_rotation = np.asarray(
            control_input.dema_rotation_world,
            dtype=float,
        ).reshape(3, 3)

        target_world = np.asarray(
            control_input.target_capsule_position_world,
            dtype=float,
        ).reshape(3)

        # ----------------------------------------------------
        # Capsule X/Y outer loop
        # ----------------------------------------------------
        xy_output = self.xy_controller.compute(
            capsule_position_world=capsule_world,
            target_capsule_position_world=target_world,
        )

        desired_dema_position = np.array(
            [
                xy_output.desired_dema_xy_world[0],
                xy_output.desired_dema_xy_world[1],
                self._dema_z_hold,
            ],
            dtype=float,
        )

        # ----------------------------------------------------
        # AUBO Cartesian servo
        # ----------------------------------------------------
        aubo_output = self.aubo_controller.compute(
            current_position_world=dema_world,
            current_rotation_world=dema_rotation,
            target_position_world=desired_dema_position,
            target_rotation_world=self._dema_rotation_hold,
        )

        # ----------------------------------------------------
        # World target -> DEMA-local Z target
        # ----------------------------------------------------
        target_local = (
            dema_rotation.T
            @ (target_world - dema_world)
        )

        target_z_lcs = float(target_local[2])

        # ----------------------------------------------------
        # Magnetic Z loop
        # ----------------------------------------------------
        z_output = self.z_controller.update(
            ZHoverControlInput(
                z_measured=float(capsule_lcs[2]),
                capsule_position_world=capsule_world,
                magnetic_moment_world=control_input.magnetic_moment_world,
                em_positions_world=control_input.em_positions_world,
                em_axes_world=control_input.em_axes_world,
                control_axis_world=dema_rotation[:, 2],
            ),
            z_ref=target_z_lcs,
        )

        return XYZControlOutput(
            current_command=z_output.current_command.copy(),
            dema_twist_world=aubo_output.twist_world.copy(),
            desired_dema_position_world=desired_dema_position.copy(),
            target_z_lcs=target_z_lcs,
            xy=xy_output,
            aubo=aubo_output,
            z=z_output,
        )
