"""
Z-axis hover controller for the DEMA.

New XYZ architecture:
    X/Y -> AUBO motion
    Z   -> DEMA differential current Id

Common current is disabled:
    Ic = 0
    I1 = -Id
    I2 = +Id
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .allocator import CurrentAllocationResult, MagneticCurrentAllocator
from .pid_z import ZPIDController, design_z_pid_gains


@dataclass(frozen=True)
class ZHoverControlInput:
    z_measured: float
    capsule_position_world: np.ndarray
    magnetic_moment_world: np.ndarray
    em_positions_world: tuple[np.ndarray, np.ndarray]
    em_axes_world: tuple[np.ndarray, np.ndarray]
    control_axis_world: np.ndarray


@dataclass(frozen=True)
class ZHoverControlOutput:
    current_command: np.ndarray
    differential_current: float
    desired_force_z: float
    pid_force: float
    z_error: float
    allocation: CurrentAllocationResult
    force_limit_saturated: bool
    integral_accepted: bool


class ZHoverController:
    """Feedforward + PID hover controller using differential current only."""

    def __init__(
        self,
        *,
        mass: float,
        drag_coefficient: float,
        control_dt: float,
        z_ref: float,
        natural_frequency: float,
        damping_ratio: float,
        integral_pole: float,
        derivative_filter_tau: float,
        integral_limit: float,
        pid_force_limit: float,
        feedforward_force_z: float,
        desired_force_z_min: float,
        desired_force_z_max: float,
        current_abs_max: float,
        differential_current_min: float,
        differential_current_max: float,
        gradient_step: float,
        em_gain: float,
    ):
        self.z_ref = float(z_ref)
        self.feedforward_force_z = float(feedforward_force_z)
        self.desired_force_z_min = float(desired_force_z_min)
        self.desired_force_z_max = float(desired_force_z_max)
        self.em_gain = float(em_gain)

        gains = design_z_pid_gains(
            mass=mass,
            drag_coefficient=drag_coefficient,
            natural_frequency=natural_frequency,
            damping_ratio=damping_ratio,
            integral_pole=integral_pole,
        )

        self.pid = ZPIDController(
            gains=gains,
            dt=control_dt,
            derivative_filter_tau=derivative_filter_tau,
            integral_limit=integral_limit,
            force_limit=pid_force_limit,
        )

        self.allocator = MagneticCurrentAllocator(
            current_abs_max=current_abs_max,
            differential_current_min=differential_current_min,
            differential_current_max=differential_current_max,
            common_current_abs_max=0.0,
            gradient_step=gradient_step,
        )

    def reset(
        self,
        *,
        initial_z: float,
        z_ref: float | None = None,
    ) -> None:
        reference = self.z_ref if z_ref is None else float(z_ref)
        self.pid.reset(initial_error=reference - float(initial_z))

    def update(
        self,
        control_input: ZHoverControlInput,
        *,
        z_ref: float | None = None,
    ) -> ZHoverControlOutput:
        reference = self.z_ref if z_ref is None else float(z_ref)

        candidate = self.pid.prepare(
            z_ref=reference,
            z_measured=float(control_input.z_measured),
        )

        force_before_axis_limit = (
            self.feedforward_force_z
            + candidate.force_candidate
        )

        desired_force_candidate = float(
            np.clip(
                force_before_axis_limit,
                self.desired_force_z_min,
                self.desired_force_z_max,
            )
        )

        candidate_allocation = self.allocator.allocate(
            p_capsule=control_input.capsule_position_world,
            capsule_moment_world=control_input.magnetic_moment_world,
            em_positions=control_input.em_positions_world,
            em_axes=control_input.em_axes_world,
            control_axis_world=control_input.control_axis_world,
            desired_force=desired_force_candidate,
            em_gain=self.em_gain,
            common_current_request=0.0,
        )

        force_limit_saturated = bool(
            abs(desired_force_candidate - force_before_axis_limit) > 1e-12
        )

        integral_accepted = not (
            force_limit_saturated
            or candidate_allocation.saturated
        )

        pid_force = float(
            self.pid.finalize(
                accept_integral=integral_accepted
            )
        )

        desired_force_z = float(
            np.clip(
                self.feedforward_force_z + pid_force,
                self.desired_force_z_min,
                self.desired_force_z_max,
            )
        )

        allocation = self.allocator.allocate(
            p_capsule=control_input.capsule_position_world,
            capsule_moment_world=control_input.magnetic_moment_world,
            em_positions=control_input.em_positions_world,
            em_axes=control_input.em_axes_world,
            control_axis_world=control_input.control_axis_world,
            desired_force=desired_force_z,
            em_gain=self.em_gain,
            common_current_request=0.0,
        )

        current_command = np.array(
            [
                allocation.current_1,
                allocation.current_2,
            ],
            dtype=float,
        )

        return ZHoverControlOutput(
            current_command=current_command,
            differential_current=float(allocation.differential_current),
            desired_force_z=desired_force_z,
            pid_force=pid_force,
            z_error=float(reference - control_input.z_measured),
            allocation=allocation,
            force_limit_saturated=force_limit_saturated,
            integral_accepted=integral_accepted,
        )
