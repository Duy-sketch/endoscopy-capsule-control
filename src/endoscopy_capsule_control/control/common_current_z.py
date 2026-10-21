"""Z PID+FF extracted from test_passive_follower.py; no capsule XY/angle PID.
The measured Z is in the moving DEMA LCS, exactly as in the baseline.
"""

import numpy as np
from endoscopy_capsule_control.magnetic import magnetic_wrench
from endoscopy_capsule_control.control.pid_z import ZPIDController, design_z_pid_gains


class CommonCurrentZ:
    def __init__(self, plant):
        self.plant = plant
        g = plant.reported_geometry()
        rot = g.lcs_pose.rotation
        self.effective_mass = (
            plant.config.capsule.mass
            - plant.config.fluid.density * plant.capsule_volume
        )
        pos = g.lcs_pose.position + rot[:, 2] * 0.1
        unit, _, field = magnetic_wrench(
            pos,
            rot[:, 2] * plant.magnetic_moment_magnitude,
            [g.em1.position_world, g.em2.position_world],
            [g.em1.axis_world, g.em2.axis_world],
            [1.0, 1.0],
            plant.config.electromagnet.magnetic_gain,
        )
        current = (
            -self.effective_mass
            * (rot[:, 2] @ plant.system.model.opt.gravity)
            / (unit @ rot[:, 2])
        )
        limit = plant.config.power_supply.current_abs_max
        if not 0 < current <= limit or field @ rot[:, 2] <= 0:
            raise ValueError("No feasible positive common-current equilibrium.")
        self.currents = np.full(2, current)
        plant.power_supply.reset(initial_current_A=self.currents)
        self.initial_current = float(current)
        cfg = plant.config.z_control
        gains = design_z_pid_gains(
            mass=plant.config.capsule.mass,
            drag_coefficient=plant.drag_coefficient,
            natural_frequency=cfg.natural_frequency,
            damping_ratio=cfg.damping_ratio,
            integral_pole=cfg.integral_pole,
        )
        pid = ZPIDController(
            gains=gains,
            dt=plant.config.simulation.control_dt,
            derivative_filter_tau=cfg.derivative_filter_tau,
            integral_limit=cfg.integral_limit,
            force_limit=cfg.pid_force_limit,
        )
        measurement = plant.sample_localization()
        pid.reset(initial_error=cfg.z_ref - measurement.position_lcs_m[2])
        # Initialization sampled t=0 for PID reset. Clear only the delay queue:
        # the first control sample then starts a regularly spaced delay history.
        # Zero-delay baseline retains its original RNG sequence and output.
        if plant.config.localization.delay_samples:
            plant.localization.reset()
        self.pid = pid
        self.measurement = measurement

    def update(self):
        plant = self.plant
        cfg = plant.config.z_control
        pid = self.pid
        effective_mass = self.effective_mass
        currents = self.currents
        # Use localization, not true_state, for the controller input.
        measurement = plant.sample_localization()
        geometry = plant.reported_geometry()
        rot = geometry.lcs_pose.rotation
        pos = geometry.lcs_pose.position + rot @ measurement.position_lcs_m
        positions = [geometry.em1.position_world, geometry.em2.position_world]
        axes = [geometry.em1.axis_world, geometry.em2.axis_world]
        _, _, field = magnetic_wrench(
            pos,
            rot[:, 2] * plant.magnetic_moment_magnitude,
            positions,
            axes,
            [1.0, 1.0],
            plant.config.electromagnet.magnetic_gain,
        )
        estimated_moment = (
            plant.magnetic_moment_magnitude * field / np.linalg.norm(field)
        )
        unit_force, _, _ = magnetic_wrench(
            pos,
            estimated_moment,
            positions,
            axes,
            [1.0, 1.0],
            plant.config.electromagnet.magnetic_gain,
        )
        authority = float(unit_force @ rot[:, 2])
        if authority >= -1e-10:
            raise RuntimeError("Common current cannot provide upward Z force here")
        ff = -effective_mass * (rot[:, 2] @ plant.system.model.opt.gravity)
        candidate = pid.prepare(
            z_ref=cfg.z_ref, z_measured=float(measurement.position_lcs_m[2])
        )
        raw_force = ff + candidate.force_candidate
        limited = float(
            np.clip(raw_force, cfg.desired_force_min, cfg.desired_force_max)
        )
        raw_current = limited / authority
        limit = min(
            plant.config.power_supply.current_abs_max,
            plant.config.electromagnet.current_abs_max,
        )
        clipped = float(np.clip(raw_current, 0.0, limit))
        accepted = (
            abs(raw_force - limited) < 1e-12 and abs(raw_current - clipped) < 1e-12
        )
        correction = pid.finalize(accept_integral=accepted)
        force = float(
            np.clip(ff + correction, cfg.desired_force_min, cfg.desired_force_max)
        )
        currents[:] = np.clip(force / authority, 0.0, limit)
        self.measurement = measurement
        return currents.copy()
