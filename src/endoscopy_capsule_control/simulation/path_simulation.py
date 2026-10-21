"""Shared orchestration for hold, line, circle and known-map A* following.

This file schedules modules; the PID, trajectories, guard and reporting each
live in their own module. Physics/controller sample order matches the baseline.
"""

import time
import numpy as np
import mujoco

from endoscopy_capsule_control.plant.z_hover_plant import ZHoverPlant
from endoscopy_capsule_control.control.common_current_z import CommonCurrentZ
from endoscopy_capsule_control.planning.trajectory import leader_offset, leader_velocity
from endoscopy_capsule_control.evaluation.results import FollowerResults
from .aubo_cartesian_controller import AuboCartesianController
from .safety import RobotEnvironmentGuard
from .viewer import configure_follower_camera


from endoscopy_capsule_control.simulation.aubo_ik import AUBODifferentialIK

def initialize_path_robot(plant, origin, rotation, start_xy_mm):
    """Set robot start pose once, before physics; keep the map frame fixed."""
    system = plant.system
    model, data = system.model, system.data
    ik = AUBODifferentialIK(system=system)
    target = origin + rotation @ np.r_[np.asarray(start_xy_mm) / 1000, 0.0]
    saved_qpos, saved_ctrl = data.qpos.copy(), data.ctrl.copy()
    try:
        for _ in range(600):
            system.forward()
            position, current_rotation = ik.site_pose()
            ep = target - position
            er = 0.5 * sum(
                np.cross(current_rotation[:, k], rotation[:, k])
                for k in range(3)
            )
            if (np.linalg.norm(ep) < 1e-6
                    and np.linalg.norm(current_rotation - rotation) < 1e-5):
                break
            J = ik.jacobian()
            dq = J.T @ np.linalg.solve(
                J @ J.T + 1e-6 * np.eye(6), np.r_[ep, er]
            )
            dq *= min(1.0, 0.05 / max(np.max(np.abs(dq)), 1e-12))
            q = data.qpos[ik.qpos_indices] + dq
            for k, jid in enumerate(ik.joint_ids):
                if model.jnt_limited[jid]:
                    q[k] = np.clip(q[k], *model.jnt_range[jid])
            data.qpos[ik.qpos_indices] = q
        else:
            raise ValueError('Cannot place DEMA above start: IK did not converge.')
        q = data.qpos[ik.qpos_indices]
        for k, aid in enumerate(ik.actuator_ids):
            if model.actuator_ctrllimited[aid]:
                lo, hi = model.actuator_ctrlrange[aid]
                if not lo <= q[k] <= hi:
                    raise ValueError('Start pose exceeds actuator limits.')
        data.ctrl[ik.actuator_ids] = q
        data.qvel[ik.dof_indices] = 0.0
        system.forward()
    except Exception:
        data.qpos[:] = saved_qpos
        data.ctrl[:] = saved_ctrl
        system.forward()
        raise


class FollowerSimulation:
    def __init__(self, options, config, noise):
        self.options = options
        self.plant = ZHoverPlant(
            config=config,
            noise=noise,
            seed=options.seed,
            vertical=True,
            initial_current_A=(0.0, 0.0),
            x_offset_m=options.x0_mm / 1000,
            y_offset_m=options.y0_mm / 1000,
            z_offset_m=options.z0_mm / 1000,
            theta_y_offset_rad=np.deg2rad(options.theta0_deg),
        )
        geometry = self.plant.reported_geometry()
        self.origin = geometry.lcs_pose.position.copy()
        self.rotation = geometry.lcs_pose.rotation.copy()
        self.plan = None
        self.path_monitor = None
        if options.mode == "path":
            from endoscopy_capsule_control.planning.environment import prepare_path
            from .safety import CapsulePathMonitor

            initialize_path_robot(
                self.plant, self.origin, self.rotation,
                [options.x0_mm, options.y0_mm],
            )
            self.plan, settings = prepare_path(
                self.plant, options, self.origin, self.rotation
            )
            self.path_monitor = CapsulePathMonitor(
                self.plant, self.plan, settings, self.origin, self.rotation
            )
        self.guard = RobotEnvironmentGuard(self.plant, options.table_clearance_mm)
        if options.mode == "path":
            clearance, pair = self.guard.check()
            if clearance <= options.table_clearance_mm / 1000:
                raise ValueError(f"Unsafe start pose: {pair}")
        self.controller = CommonCurrentZ(self.plant)
        self.robot = AuboCartesianController(
            system=self.plant.system, config=config.aubo_motion
        )
        self.robot.reset()
        options.csv.parent.mkdir(parents=True, exist_ok=True)
        self.results = FollowerResults(
            options,
            self.plant,
            self.controller,
            self.guard,
            self.origin,
            self.rotation,
            self.target_at,
        )
        self.results.plan = self.plan
        self.results.path_monitor = self.path_monitor
        self.reason = "duration completed"
        self.dt = self.plant.physics_dt
        self.every = int(round(config.simulation.control_dt / self.dt))
        if not np.isclose(self.every * self.dt, config.simulation.control_dt):
            raise ValueError(
                "Control period must contain an integer number of physics steps."
            )
        print(
            "Assumed source tau / localization delay [ms]:",
            options.current_tau_ms,
            options.localization_delay_ms,
        )
        print("Noise:", noise.enabled_names() or ("none",), "Seed:", options.seed)
        print("MuJoCo version:", mujoco.__version__)
        print("Clearance query capped at [mm]:", self.guard.clearance_query_m * 1000)
        print(
            "Robot/table clearance guard:",
            (
                "ON"
                if self.guard.table_present and self.guard.moving
                else "unavailable: install updated table XML"
            ),
        )
        print("Nominal common currents [A]:", self.controller.currents)
        if options.mode == "follow":
            peak = (
                3 * np.pi * options.radius_mm
                if options.trajectory == "circle"
                else 1.5 * abs(options.distance_mm)
            ) / options.move_time
            print("Trajectory:", options.trajectory, "planned peak speed [mm/s]:", peak)
            print("Robot velocity feedforward:", not options.no_leader_feedforward)
            if peak > config.aubo_motion.max_linear_speed_mps * 1000:
                print(
                    "Requested speed exceeds robot limit; expect leader tracking lag."
                )
            if options.time < options.move_time:
                print("Run ends before the requested trajectory completes.")

    def target_at(self, t):
        """Leader target in world metres; no capsule-Z error enters this target."""
        a = self.options
        if self.plan is not None:
            position, _, _ = self.plan.reference(t)
            return self.origin + self.rotation @ np.array(
                [position[0], position[1], 0.0]
            )
        if a.mode != "follow":
            return self.origin.copy()
        return self.origin + self.rotation @ leader_offset(
            t,
            a.move_time,
            a.trajectory,
            a.axis,
            a.distance_mm / 1000,
            a.radius_mm / 1000,
        )

    def step(self, k):
        """Guard -> robot reference -> measured-Z control -> physics -> log."""
        a, plant = self.options, self.plant
        if k % self.every == 0:
            clearance, pair = self.guard.check()
            if clearance <= a.table_clearance_mm / 1000:
                self.reason = f"STOP: clearance {clearance*1000:.2f} mm, {pair}"
                return False
            if a.mode in ("follow", "path"):
                target = self.target_at(plant.time_s)
                if not a.no_leader_feedforward:
                    velocity = self.rotation @ leader_velocity(
                        plant.time_s,
                        a.move_time,
                        a.trajectory,
                        a.axis,
                        a.distance_mm / 1000,
                        a.radius_mm / 1000,
                    )
                    if self.plan is not None:
                        velocity = self.rotation @ self.plan.reference(plant.time_s)[1]
                    target = (
                        target + velocity / plant.config.aubo_motion.position_kp_per_s
                    )
                self.robot.command_position(
                    target_position_world_m=target,
                    dt=plant.config.simulation.control_dt,
                )
            if a.open_loop:
                self.controller.measurement = plant.sample_localization()
            else:
                self.controller.update()
        snapshot = plant.step(self.controller.currents)
        self.results.noise_sq += snapshot.power_supply.noise_A**2
        self.results.noise_count += 1
        if (k + 1) % self.every == 0:
            self.results.record()
        if self.path_monitor is not None:
            stop = self.path_monitor.check()
            if stop:
                self.reason = "STOP: " + stop
                return False
        p = plant.true_state().position_lcs
        if (
            not np.isfinite(p).all()
            or np.linalg.norm(p[:2]) > 0.08
            or not 0.04 < p[2] < 0.18
        ):
            self.reason = "STOP: left diagnostic ROI (radial 80 mm, Z 40..180 mm)"
            return False
        return True

    def save(self):
        self.results.status = self.reason
        self.results.save_logs()

    def run(self):
        if self.options.mode == "scan":
            from .follower_scan import run_scan

            run_scan(self)
            return
        if self.path_monitor is not None:
            stop = self.path_monitor.check()
            if stop:
                self.reason = "STOP: " + stop
                self.save()
                return
        self.results.record()
        self.results.write_metadata("running")
        steps = int(np.ceil(self.options.time / self.dt))
        print(
            "Z control:",
            "OFF (fixed currents)" if self.options.open_loop else "PID+FF, I1=I2",
        )
        print("No XY or angle PID. Capsule rotation and translation remain free.")
        if self.options.headless:
            for k in range(steps):
                if not self.step(k):
                    break
        else:
            self.run_viewer(steps)
        self.save()
        self.results.summary()

    def run_viewer(self, steps):
        """Native viewer stays open after saving, just like the original runner."""
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(
            self.plant.system.model, self.plant.system.data
        ) as viewer:
            configure_follower_camera(
                viewer, self.plant.system, self.options.camera, self.guard.table_present
            )
            for k in range(steps):
                if not viewer.is_running():
                    self.reason = "viewer closed"
                    break
                started = time.perf_counter()
                keep = self.step(k)
                viewer.sync()
                if not keep:
                    break
                time.sleep(max(0.0, self.dt - (time.perf_counter() - started)))
            self.save()
            print(self.reason, "- CSV saved. Close viewer to exit")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)


def main(argv=None):
    from .follower_options import parse_options

    options, config, noise = parse_options(argv)
    try:
        simulation = FollowerSimulation(options, config, noise)
    except ValueError as exc:
        raise SystemExit("Configuration/planning error: " + str(exc)) from exc
    simulation.run()
