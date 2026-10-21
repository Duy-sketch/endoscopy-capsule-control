"""Constrained local allocator for slow AUBO translation + coil current.

The allocator linearizes magnetic force around the current reported
operating point and solves for an incremental command

    [dx_D, dy_D, dz_D, dgamma]

where DEMA translation is expressed in the current DEMA LCS and

    I = gamma * I_base.

Robot position corrections are bounded by a local trust region; the AUBO
Cartesian servo independently enforces the configured slow speed and
acceleration limits.  Current scaling is allowed to react faster.  A weak
secondary objective
recenters gamma toward its nominal value, so persistent load can be
trimmed by slow robot repositioning instead of permanently biasing current.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from endoscopy_capsule_control.config import AuboMotionConfig, XYZControlConfig
from endoscopy_capsule_control.simulation.dema_geometry import DEMAGeometry

from .actuation_jacobian import (
    ActuationJacobianResult,
    compute_additive_gamma_force_linearization,
)


@dataclass(frozen=True)
class RobotCurrentAllocationResult:
    dema_translation_increment_lcs_m: np.ndarray
    gamma_previous: float
    gamma_command: float
    gamma_increment: float
    current_command_A: np.ndarray
    baseline_force_lcs_N: np.ndarray
    desired_force_lcs_N: np.ndarray
    predicted_force_lcs_N: np.ndarray
    force_residual_lcs_N: np.ndarray
    force_residual_saturated_axes: np.ndarray
    normalized_solution: np.ndarray
    variable_at_bound: np.ndarray
    jacobian: ActuationJacobianResult


def _bounded_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Small active-set box-constrained least-squares solver.

    The Stage-2 problem has only four variables, so a compact active-set
    implementation avoids adding SciPy as a project dependency.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)

    n = A.shape[1]
    if lower.size != n or upper.size != n:
        raise ValueError("bounds must match the number of columns in A")
    if np.any(lower > upper):
        raise ValueError("invalid box bounds")

    x = np.zeros(n, dtype=float)
    free = np.ones(n, dtype=bool)

    # A variable can be fixed at a one-sided bound at most once, hence n+1
    # passes are sufficient for this tiny convex quadratic problem.
    for _ in range(n + 1):
        fixed = ~free
        rhs = b.copy()
        if np.any(fixed):
            rhs = rhs - A[:, fixed] @ x[fixed]

        if np.any(free):
            x_free, *_ = np.linalg.lstsq(A[:, free], rhs, rcond=None)
            x[free] = x_free

        below = free & (x < lower - 1e-12)
        above = free & (x > upper + 1e-12)
        if not np.any(below | above):
            return np.clip(x, lower, upper)

        # Fix the most strongly violated free variable and resolve.
        violation = np.zeros(n, dtype=float)
        violation[below] = lower[below] - x[below]
        violation[above] = x[above] - upper[above]
        index = int(np.argmax(violation))
        x[index] = lower[index] if below[index] else upper[index]
        free[index] = False

    return np.clip(x, lower, upper)


class RobotCurrentAllocator:
    """Allocate desired local magnetic force to robot translation + current."""

    def __init__(
        self,
        *,
        xyz_config: XYZControlConfig,
        aubo_motion_config: AuboMotionConfig,
        control_dt: float,
        em_gain: float,
        gradient_step: float,
    ) -> None:
        self.config = xyz_config
        self.aubo_motion_config = aubo_motion_config
        self.control_dt = float(control_dt)
        self.em_gain = float(em_gain)
        self.gradient_step = float(gradient_step)

        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if not (self.config.gamma_min < self.config.gamma_max):
            raise ValueError("gamma_min must be less than gamma_max")
        if self.config.max_gamma_step <= 0.0:
            raise ValueError("max_gamma_step must be positive")

        self.base_currents_A = np.asarray(
            self.config.nominal_currents_A, dtype=float
        ).reshape(2)
        self.force_scale_N = np.asarray(
            self.config.allocator_force_scale_N, dtype=float
        ).reshape(3)
        if np.any(self.force_scale_N <= 0.0):
            raise ValueError("allocator_force_scale_N must be positive")

        # IMPORTANT: Delta P in the magnetic inverse problem is a *relative
        # DEMA position correction*, not a Cartesian displacement that must be
        # completed within one control period.  The AUBO position servo below
        # already enforces the physical speed/acceleration limits.
        #
        # If we used v_max * dt here (0.05 mm at 5 mm/s and 100 Hz), the
        # outer allocator would never place the position target farther than
        # 0.05 mm from the current DEMA pose.  With v = Kp * position_error
        # and Kp=2 1/s, that unintentionally limits the robot to only
        # 0.1 mm/s -- fifty times slower than the configured 5 mm/s.
        #
        # Instead, choose a local position-correction trust region large
        # enough to let the inner Cartesian servo reach v_max, while keeping
        # the finite-difference magnetic linearization local.
        servo_offset_for_max_speed_m = (
            float(self.aubo_motion_config.max_linear_speed_mps)
            / float(self.aubo_motion_config.position_kp_per_s)
        )
        jacobian_locality_limit_m = (
            5.0 * float(self.config.jacobian_translation_probe_m)
        )
        self.robot_displacement_limit_m = min(
            servo_offset_for_max_speed_m,
            jacobian_locality_limit_m,
        )
        if self.robot_displacement_limit_m <= 0.0:
            raise ValueError("robot displacement trust-region limit must be positive")

    def allocate(
        self,
        *,
        desired_force_lcs_N: np.ndarray,
        capsule_position_world: np.ndarray,
        capsule_moment_world: np.ndarray,
        geometry: DEMAGeometry,
        gamma_current: float,
    ) -> RobotCurrentAllocationResult:
        """Allocate fast current and slow DEMA translation on separate time scales.

        Stage-2 has two physically different actuators:

        * ``gamma`` changes coil current essentially at the control rate;
        * DEMA translation is only a *position target* for the slow AUBO servo.

        Therefore a requested DEMA displacement must not be counted as if its
        magnetic-force effect already existed in the current control sample.
        The fast current channel first protects the local-Z levitation force at
        the *current physical DEMA pose* using the instantaneous fixed-moment
        model.  The remaining slow robot-position target uses the quasi-static
        field-aligned force map because the capsule rotates while AUBO moves.

        This is intentionally a Stage-2 translational allocator.  The later
        5-DOF controller will replace the local-Z specialisation with a full
        force/field (or force/torque) allocation.
        """
        desired_force = np.asarray(desired_force_lcs_N, dtype=float).reshape(3)
        gamma_current = float(gamma_current)

        # Two magnetic linearizations are needed because the two actuators
        # operate on very different time scales.
        #
        # FAST current update: the capsule attitude cannot change
        # instantaneously, so use the measured/fixed magnetic moment.
        fast_linearization = compute_additive_gamma_force_linearization(
            capsule_position_world=capsule_position_world,
            capsule_moment_world=capsule_moment_world,
            geometry=geometry,
            base_currents_A=self.base_currents_A,
            gamma=gamma_current,
            em_gain=self.em_gain,
            gradient_step=self.gradient_step,
            translation_probe_m=self.config.jacobian_translation_probe_m,
            gamma_probe=self.config.jacobian_gamma_probe,
            align_moment_to_field=False,
        )

        # SLOW robot translation: during AUBO motion the capsule has time to
        # rotate under magnetic torque.  Use the quasi-static field-aligned
        # force map for the robot translation columns.
        slow_linearization = compute_additive_gamma_force_linearization(
            capsule_position_world=capsule_position_world,
            capsule_moment_world=capsule_moment_world,
            geometry=geometry,
            base_currents_A=self.base_currents_A,
            gamma=gamma_current,
            em_gain=self.em_gain,
            gradient_step=self.gradient_step,
            translation_probe_m=self.config.jacobian_translation_probe_m,
            gamma_probe=self.config.jacobian_gamma_probe,
            align_moment_to_field=True,
        )
        if slow_linearization.rank < 3:
            raise RuntimeError(
                "Slow quasi-static robot force map lost rank; "
                f"rank={slow_linearization.rank}."
            )

        g_gamma = np.asarray(
            fast_linearization.jacobian_force_lcs[:, 3], dtype=float
        )
        baseline = np.asarray(
            fast_linearization.baseline_force_lcs_N, dtype=float
        )

        # ------------------------------------------------------------------
        # 1. FAST CHANNEL: current scaling protects levitation immediately.
        # ------------------------------------------------------------------
        # With the Stage-2 DEMA pose convention, local Z is the levitation
        # direction.  Solve dgamma from the force error at the CURRENT DEMA
        # pose; do not give the slow robot motion credit before it happens.
        z_axis = 2
        k_gamma_z = float(g_gamma[z_axis])
        if abs(k_gamma_z) < 1e-12:
            raise RuntimeError(
                "Current-scaling authority along local Z is approximately zero."
            )

        delta_gamma_requested = float(
            (desired_force[z_axis] - baseline[z_axis]) / k_gamma_z
        )

        gamma_scale = float(self.config.max_gamma_step)
        lower_dgamma = max(
            -gamma_scale,
            float(self.config.gamma_min) - gamma_current,
        )
        upper_dgamma = min(
            +gamma_scale,
            float(self.config.gamma_max) - gamma_current,
        )
        delta_gamma = float(
            np.clip(delta_gamma_requested, lower_dgamma, upper_dgamma)
        )
        gamma_command = float(
            np.clip(
                gamma_current + delta_gamma,
                self.config.gamma_min,
                self.config.gamma_max,
            )
        )
        current_command = gamma_command * self.base_currents_A

        force_after_fast_current = baseline + g_gamma * delta_gamma

        # ------------------------------------------------------------------
        # 2. SLOW CHANNEL: turn the remaining force request into an AUBO
        #    position target.  Robot motion is slow enough for the capsule
        #    magnetic heading to follow the field, so recompute the
        #    quasi-static robot map at the NEW current command.
        # ------------------------------------------------------------------
        slow_linearization = compute_additive_gamma_force_linearization(
            capsule_position_world=capsule_position_world,
            capsule_moment_world=capsule_moment_world,
            geometry=geometry,
            base_currents_A=self.base_currents_A,
            gamma=gamma_command,
            em_gain=self.em_gain,
            gradient_step=self.gradient_step,
            translation_probe_m=self.config.jacobian_translation_probe_m,
            gamma_probe=self.config.jacobian_gamma_probe,
            align_moment_to_field=True,
        )
        if slow_linearization.rank < 3:
            raise RuntimeError(
                "Slow quasi-static robot force map lost rank after current update; "
                f"rank={slow_linearization.rank}."
            )

        G_robot = np.asarray(
            slow_linearization.jacobian_force_lcs[:, :3], dtype=float
        )
        slow_baseline = np.asarray(
            slow_linearization.baseline_force_lcs_N, dtype=float
        )
        remaining_force = desired_force - slow_baseline

        position_scale = float(self.robot_displacement_limit_m)
        variable_scale_robot = np.full(3, position_scale, dtype=float)

        A_force = (
            np.diag(1.0 / self.force_scale_N)
            @ G_robot
            @ np.diag(variable_scale_robot)
        )
        b_force = np.diag(1.0 / self.force_scale_N) @ remaining_force

        rows = [A_force]
        rhs = [b_force]

        robot_reg = float(self.config.allocator_robot_regularization)
        if robot_reg > 0.0:
            rows.append(np.sqrt(robot_reg) * np.eye(3))
            rhs.append(np.zeros(3, dtype=float))

        A = np.vstack(rows)
        b = np.concatenate(rhs)
        lower_robot = -np.ones(3, dtype=float)
        upper_robot = +np.ones(3, dtype=float)
        normalized_robot = _bounded_least_squares(
            A, b, lower_robot, upper_robot
        )
        delta_position_lcs = variable_scale_robot * normalized_robot

        # Future force estimate after the requested slow robot target is
        # eventually reached.  This is useful for diagnostics/anti-windup,
        # while the fast gamma command above is based only on the current pose.
        predicted = (
            slow_baseline
            + G_robot @ delta_position_lcs
        )
        residual = desired_force - predicted
        saturated_axes = (
            np.abs(residual)
            > float(self.config.allocator_force_residual_tolerance_N)
        )

        normalized_gamma = (
            delta_gamma / gamma_scale if gamma_scale > 0.0 else 0.0
        )
        normalized = np.concatenate(
            [normalized_robot, np.array([normalized_gamma], dtype=float)]
        )

        gamma_at_bound = (
            abs(delta_gamma - lower_dgamma) <= 1e-10
            or abs(delta_gamma - upper_dgamma) <= 1e-10
        )
        at_bound = np.concatenate(
            [
                np.isclose(normalized_robot, lower_robot, atol=1e-8)
                | np.isclose(normalized_robot, upper_robot, atol=1e-8),
                np.array([gamma_at_bound], dtype=bool),
            ]
        )

        return RobotCurrentAllocationResult(
            dema_translation_increment_lcs_m=delta_position_lcs.copy(),
            gamma_previous=gamma_current,
            gamma_command=gamma_command,
            gamma_increment=delta_gamma,
            current_command_A=current_command.copy(),
            baseline_force_lcs_N=baseline.copy(),
            desired_force_lcs_N=desired_force.copy(),
            predicted_force_lcs_N=predicted.copy(),
            force_residual_lcs_N=residual.copy(),
            force_residual_saturated_axes=saturated_axes.copy(),
            normalized_solution=normalized.copy(),
            variable_at_bound=at_bound.copy(),
            jacobian=slow_linearization,
        )
