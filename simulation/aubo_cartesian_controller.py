"""Slow Cartesian position servo for the AUBO i10 DEMA mount.

This module is intentionally robot-only.  It does not know about the
capsule controller or the magnetic current allocator.

The controller keeps the DEMA orientation approximately fixed by
commanding zero angular velocity while translating the LCS origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from endoscopy_capsule_control.config import AuboMotionConfig

from .aubo_ik import AUBODifferentialIK
from .mujoco_system import MujocoSystem


def _limit_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if max_norm <= 0.0 or norm <= max_norm:
        return vector.copy()
    return vector * (float(max_norm) / norm)


@dataclass(frozen=True)
class AuboCartesianStep:
    position_world_m: np.ndarray
    target_position_world_m: np.ndarray
    position_error_world_m: np.ndarray
    commanded_linear_velocity_world_mps: np.ndarray
    joint_velocity_rad_s: np.ndarray


class AuboCartesianController:
    """Position servo for the DEMA LCS origin with conservative speed limits."""

    def __init__(
        self,
        *,
        system: MujocoSystem,
        config: AuboMotionConfig,
        site_name: str = "LCS_origin",
    ) -> None:
        self.system = system
        self.config = config

        if self.config.max_linear_speed_mps <= 0.0:
            raise ValueError("max_linear_speed_mps must be positive")
        if self.config.max_linear_acceleration_mps2 <= 0.0:
            raise ValueError("max_linear_acceleration_mps2 must be positive")
        if self.config.position_kp_per_s <= 0.0:
            raise ValueError("position_kp_per_s must be positive")

        self.ik = AUBODifferentialIK(
            system=self.system,
            site_name=site_name,
            damping=self.config.ik_damping,
            max_joint_speed=self.config.max_joint_speed_rad_s,
        )
        self.previous_linear_velocity_world = np.zeros(3, dtype=float)

    def reset(self) -> None:
        self.ik.reset_command_to_current()
        self.previous_linear_velocity_world[:] = 0.0

    def current_position_world(self) -> np.ndarray:
        position, _ = self.ik.site_pose()
        return position

    def command_position(
        self,
        *,
        target_position_world_m: np.ndarray,
        dt: float,
    ) -> AuboCartesianStep:
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        target = np.asarray(target_position_world_m, dtype=float).reshape(3)
        current = self.current_position_world()
        error = target - current

        desired_velocity = self.config.position_kp_per_s * error
        desired_velocity = _limit_norm(
            desired_velocity,
            self.config.max_linear_speed_mps,
        )

        delta_v = desired_velocity - self.previous_linear_velocity_world
        max_delta_v = self.config.max_linear_acceleration_mps2 * float(dt)
        delta_v = _limit_norm(delta_v, max_delta_v)
        velocity_command = self.previous_linear_velocity_world + delta_v
        velocity_command = _limit_norm(
            velocity_command,
            self.config.max_linear_speed_mps,
        )
        self.previous_linear_velocity_world = velocity_command.copy()

        # Twist convention used by AUBODifferentialIK:
        # [vx, vy, vz, wx, wy, wz].  Zero angular velocity keeps the
        # DEMA orientation fixed to first order while translating.
        twist_world = np.concatenate(
            [velocity_command, np.zeros(3, dtype=float)]
        )
        qdot = self.ik.command_twist(
            twist_world=twist_world,
            dt=float(dt),
        )

        return AuboCartesianStep(
            position_world_m=current.copy(),
            target_position_world_m=target.copy(),
            position_error_world_m=error.copy(),
            commanded_linear_velocity_world_mps=velocity_command.copy(),
            joint_velocity_rad_s=qdot.copy(),
        )
