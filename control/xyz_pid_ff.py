"""3-D force-based PID + feedforward controller.

The outer loop is expressed in the world coordinate system (WCS):

    F_des_world = F_ff_world + F_pid_world

where the PID acts on capsule position error in WCS.  The downstream
allocator decides how much of the requested magnetic-force change is
produced by slow AUBO-i10 DEMA translation and by fast coil-current
scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pid_z import ZPIDController, ZPIDGains, design_z_pid_gains


@dataclass(frozen=True)
class XYZPIDFFCandidate:
    position_error_world_m: np.ndarray
    pid_force_candidate_world_N: np.ndarray
    desired_force_candidate_world_N: np.ndarray


@dataclass(frozen=True)
class XYZPIDFFOutput:
    position_error_world_m: np.ndarray
    pid_force_world_N: np.ndarray
    feedforward_force_world_N: np.ndarray
    desired_force_world_N: np.ndarray


def design_xyz_pid_gains(
    *,
    mass: float,
    drag_coefficient: float,
    natural_frequency_rad_s,
    damping_ratio,
    integral_pole_per_s,
) -> tuple[ZPIDGains, ZPIDGains, ZPIDGains]:
    wn = np.asarray(natural_frequency_rad_s, dtype=float).reshape(3)
    zeta = np.asarray(damping_ratio, dtype=float).reshape(3)
    pole = np.asarray(integral_pole_per_s, dtype=float).reshape(3)

    gains = []
    for axis in range(3):
        gains.append(
            design_z_pid_gains(
                mass=mass,
                drag_coefficient=drag_coefficient,
                natural_frequency=float(wn[axis]),
                damping_ratio=float(zeta[axis]),
                integral_pole=float(pole[axis]),
            )
        )
    return tuple(gains)  # type: ignore[return-value]


class XYZPIDFFController:
    """Three independent force PID loops in WCS plus static feedforward."""

    def __init__(
        self,
        *,
        target_position_world_m: np.ndarray,
        feedforward_force_world_N: np.ndarray,
        gains: tuple[ZPIDGains, ZPIDGains, ZPIDGains],
        dt: float,
        derivative_filter_tau_s: float,
        integral_limit_m_s,
        pid_force_limit_N,
    ) -> None:
        self.target_position_world_m = np.asarray(
            target_position_world_m, dtype=float
        ).reshape(3)
        self.feedforward_force_world_N = np.asarray(
            feedforward_force_world_N, dtype=float
        ).reshape(3)

        integral_limit = np.asarray(integral_limit_m_s, dtype=float).reshape(3)
        force_limit = np.asarray(pid_force_limit_N, dtype=float).reshape(3)

        self.axis_controllers = tuple(
            ZPIDController(
                gains=gains[i],
                dt=float(dt),
                derivative_filter_tau=float(derivative_filter_tau_s),
                integral_limit=float(integral_limit[i]),
                force_limit=float(force_limit[i]),
            )
            for i in range(3)
        )
        self._candidate: XYZPIDFFCandidate | None = None

    def reset(self, *, initial_position_world_m: np.ndarray) -> None:
        position = np.asarray(initial_position_world_m, dtype=float).reshape(3)
        error = self.target_position_world_m - position
        for axis, controller in enumerate(self.axis_controllers):
            controller.reset(initial_error=float(error[axis]))
        self._candidate = None

    def prepare(self, *, measured_position_world_m: np.ndarray) -> XYZPIDFFCandidate:
        position = np.asarray(measured_position_world_m, dtype=float).reshape(3)
        error = self.target_position_world_m - position

        pid_force = np.zeros(3, dtype=float)
        for axis, controller in enumerate(self.axis_controllers):
            candidate = controller.prepare(
                z_ref=float(self.target_position_world_m[axis]),
                z_measured=float(position[axis]),
            )
            pid_force[axis] = candidate.force_candidate

        desired = self.feedforward_force_world_N + pid_force
        out = XYZPIDFFCandidate(
            position_error_world_m=error.copy(),
            pid_force_candidate_world_N=pid_force.copy(),
            desired_force_candidate_world_N=desired.copy(),
        )
        self._candidate = out
        return out

    def finalize(self, *, accept_integral_axes) -> XYZPIDFFOutput:
        if self._candidate is None:
            raise RuntimeError("prepare() must be called before finalize()")

        accept = np.asarray(accept_integral_axes, dtype=bool).reshape(3)
        pid_force = np.array(
            [
                controller.finalize(accept_integral=bool(accept[i]))
                for i, controller in enumerate(self.axis_controllers)
            ],
            dtype=float,
        )
        desired = self.feedforward_force_world_N + pid_force
        candidate = self._candidate
        self._candidate = None

        return XYZPIDFFOutput(
            position_error_world_m=candidate.position_error_world_m.copy(),
            pid_force_world_N=pid_force,
            feedforward_force_world_N=self.feedforward_force_world_N.copy(),
            desired_force_world_N=desired,
        )
