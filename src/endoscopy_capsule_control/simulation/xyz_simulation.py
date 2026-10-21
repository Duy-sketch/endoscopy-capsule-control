"""Stage-2 closed-loop XYZ hover using PID+FF and AUBO/current allocation."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from endoscopy_capsule_control.config import AppConfig, DEFAULT_CONFIG
from endoscopy_capsule_control.control.robot_current_allocator import (
    RobotCurrentAllocationResult,
    RobotCurrentAllocator,
)
from endoscopy_capsule_control.control.xyz_pid_ff import (
    XYZPIDFFController,
    XYZPIDFFOutput,
    design_xyz_pid_gains,
)
from endoscopy_capsule_control.plant.localization import LocalizationMeasurement
from endoscopy_capsule_control.plant.noise import NoiseSwitches
from endoscopy_capsule_control.plant.z_hover_plant import ZHoverPlant

from .aubo_cartesian_controller import AuboCartesianController, AuboCartesianStep
from .dema_geometry import lcs_to_world_position
from .viewer import (
    configure_capsule_tracking_camera,
    configure_dema_overview_camera,
)


@dataclass(frozen=True)
class XYZSimulationResult:
    noise: NoiseSwitches
    seed: int
    simulation_time_s: float
    settling_time_s: float

    target_position_world_m: np.ndarray
    time_s: np.ndarray
    true_position_world_m: np.ndarray
    measured_position_world_m: np.ndarray
    position_error_world_m: np.ndarray
    dema_position_world_m: np.ndarray
    dema_target_world_m: np.ndarray
    gamma: np.ndarray
    current_command_A: np.ndarray
    current_actual_A: np.ndarray

    robot_translation_bias_world_m: np.ndarray
    rms_true_error_world_m: np.ndarray
    rms_true_error_norm_m: float
    max_true_error_norm_m: float
    rms_localization_residual_world_m: np.ndarray
    rms_current_noise_A: np.ndarray

    final_true_position_world_m: np.ndarray
    final_measured_position_world_m: np.ndarray
    final_dema_position_world_m: np.ndarray
    final_gamma: float
    final_current_command_A: np.ndarray
    final_current_actual_A: np.ndarray

    def print_summary(self) -> None:
        enabled = self.noise.enabled_names()
        noise_text = ", ".join(enabled) if enabled else "none"

        print("\n======================================")
        print("XYZ HOVER — PID+FF + AUBO/CURRENT")
        print("======================================")
        print("Noise enabled:", noise_text)
        print("Seed:", self.seed)
        print("Simulation time [s]:", self.simulation_time_s)
        print("Settling time [s]:", self.settling_time_s)

        print("\nTarget position WCS [mm]:")
        print(1000.0 * self.target_position_world_m)
        print("AUBO i10 fixed pose bias WCS [mm]:")
        print(1000.0 * self.robot_translation_bias_world_m)

        print("\nPost-settling true XYZ RMS error [mm]:")
        print(1000.0 * self.rms_true_error_world_m)
        print("Post-settling true 3-D error-norm RMS [mm]:")
        print(1000.0 * self.rms_true_error_norm_m)
        print("Post-settling max true error norm [mm]:")
        print(1000.0 * self.max_true_error_norm_m)
        print("Post-settling localization residual RMS WCS [mm]:")
        print(1000.0 * self.rms_localization_residual_world_m)

        print("\nPower-supply current-noise RMS [mA], channels 1/2:")
        print(1000.0 * self.rms_current_noise_A)

        print("\nFinal true position WCS [mm]:")
        print(1000.0 * self.final_true_position_world_m)
        print("Final measured position WCS [mm]:")
        print(1000.0 * self.final_measured_position_world_m)
        print("Final DEMA position WCS [mm]:")
        print(1000.0 * self.final_dema_position_world_m)
        print("Final gamma:", self.final_gamma)
        print("Final current command [A]:", self.final_current_command_A)
        print("Final current actual [A]:", self.final_current_actual_A)


class XYZSimulation:
    """Local 3-D hover controller with coupled AUBO/current actuation."""

    def __init__(
        self,
        *,
        config: AppConfig = DEFAULT_CONFIG,
        noise: NoiseSwitches | None = None,
        seed: int = 1,
        simulation_time_s: float = 12.0,
        settling_time_s: float = 6.0,
        x_offset_m: float = 0.0,
        y_offset_m: float = 0.0,
        z_offset_m: float = 0.0,
        theta_y_offset_rad: float = 0.0,
    ) -> None:
        if simulation_time_s <= 0.0:
            raise ValueError("simulation_time_s must be positive")
        if not (0.0 <= settling_time_s < simulation_time_s):
            raise ValueError(
                "settling_time_s must satisfy 0 <= settling < simulation_time"
            )

        self.config = config
        self.noise = NoiseSwitches.none() if noise is None else noise
        self.seed = int(seed)
        self.simulation_time_s = float(simulation_time_s)
        self.settling_time_s = float(settling_time_s)

        initial_current = np.asarray(
            self.config.xyz_control.nominal_currents_A, dtype=float
        )
        self.plant = ZHoverPlant(
            config=self.config,
            noise=self.noise,
            seed=self.seed,
            x_offset_m=float(x_offset_m),
            y_offset_m=float(y_offset_m),
            z_offset_m=float(z_offset_m),
            theta_y_offset_rad=float(theta_y_offset_rad),
            initial_current_A=initial_current,
        )

        self.physics_dt = float(self.plant.physics_dt)
        self.control_dt = float(self.config.simulation.control_dt)
        self.control_every = int(round(self.control_dt / self.physics_dt))
        if self.control_every <= 0 or not np.isclose(
            self.control_every * self.physics_dt,
            self.control_dt,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError("control_dt must be an integer multiple of physics_dt")
        self.number_of_physics_steps = int(
            round(self.simulation_time_s / self.physics_dt)
        )

        # Fixed WCS capsule target derived from the nominal operating point.
        initial_geometry = self.plant.reported_geometry()
        self.initial_dema_position_world = initial_geometry.lcs_pose.position.copy()
        self.initial_dema_rotation_world_lcs = initial_geometry.lcs_pose.rotation.copy()
        self.target_position_world_m = lcs_to_world_position(
            np.asarray(self.config.xyz_control.target_position_lcs_m, dtype=float),
            initial_geometry.lcs_pose,
        )

        # Static magnetic feedforward = -(gravity + buoyancy).
        apparent_mass = (
            float(self.config.capsule.mass)
            - float(self.config.fluid.density) * float(self.plant.capsule_volume)
        )
        gravity_world = np.asarray(self.plant.system.model.opt.gravity, dtype=float)
        self.feedforward_force_world_N = -apparent_mass * gravity_world

        gains = design_xyz_pid_gains(
            mass=self.config.capsule.mass,
            drag_coefficient=self.plant.drag_coefficient,
            natural_frequency_rad_s=(
                self.config.xyz_control.natural_frequency_rad_s
            ),
            damping_ratio=self.config.xyz_control.damping_ratio,
            integral_pole_per_s=self.config.xyz_control.integral_pole_per_s,
        )
        self.pid = XYZPIDFFController(
            target_position_world_m=self.target_position_world_m,
            feedforward_force_world_N=self.feedforward_force_world_N,
            gains=gains,
            dt=self.control_dt,
            derivative_filter_tau_s=(
                self.config.xyz_control.derivative_filter_tau_s
            ),
            integral_limit_m_s=self.config.xyz_control.integral_limit_m_s,
            pid_force_limit_N=self.config.xyz_control.pid_force_limit_N,
        )

        self.allocator = RobotCurrentAllocator(
            xyz_config=self.config.xyz_control,
            aubo_motion_config=self.config.aubo_motion,
            control_dt=self.control_dt,
            em_gain=self.config.electromagnet.magnetic_gain,
            gradient_step=self.config.electromagnet.magnetic_gradient_step,
        )
        self.robot = AuboCartesianController(
            system=self.plant.system,
            config=self.config.aubo_motion,
        )
        self.robot.reset()

        self.gamma = float(self.config.xyz_control.gamma_nominal)
        self.current_command_A = (
            self.gamma
            * np.asarray(self.config.xyz_control.nominal_currents_A, dtype=float)
        )
        self.dema_target_world_m = self.initial_dema_position_world.copy()

        self.last_measurement = self.plant.sample_localization()
        measured_world, _ = self._measurement_to_world(self.last_measurement)
        self.pid.reset(initial_position_world_m=measured_world)

        self.last_pid_output: XYZPIDFFOutput | None = None
        self.last_allocation: RobotCurrentAllocationResult | None = None
        self.last_robot_step: AuboCartesianStep | None = None
        self._update_control(self.last_measurement)

        self._time_log: list[float] = []
        self._true_position_log: list[np.ndarray] = []
        self._measured_position_log: list[np.ndarray] = []
        self._error_log: list[np.ndarray] = []
        self._dema_position_log: list[np.ndarray] = []
        self._dema_target_log: list[np.ndarray] = []
        self._gamma_log: list[float] = []
        self._current_command_log: list[np.ndarray] = []
        self._current_actual_log: list[np.ndarray] = []

        self._sum_sq_current_noise = np.zeros(2, dtype=float)
        self._current_noise_samples = 0
        self._record_control_sample()

    def _measurement_to_world(
        self,
        measurement: LocalizationMeasurement,
    ) -> tuple[np.ndarray, np.ndarray]:
        # RF localization is produced in the physical DEMA frame; the
        # controller reconstructs WCS position with reported AUBO kinematics.
        geometry = self.plant.reported_geometry()
        measured_position_world = lcs_to_world_position(
            measurement.position_lcs_m,
            geometry.lcs_pose,
        )

        theta_y = float(measurement.theta_y_rad)
        moment_magnitude = float(self.plant.magnetic_moment_magnitude)
        measured_moment_lcs = moment_magnitude * np.array(
            [np.cos(theta_y), 0.0, -np.sin(theta_y)], dtype=float
        )
        measured_moment_world = geometry.lcs_pose.rotation @ measured_moment_lcs
        return measured_position_world, measured_moment_world

    def _clip_dema_target(self, target_world: np.ndarray) -> tuple[np.ndarray, bool]:
        target_world = np.asarray(target_world, dtype=float).reshape(3)
        offset_lcs0 = (
            self.initial_dema_rotation_world_lcs.T
            @ (target_world - self.initial_dema_position_world)
        )
        limit = np.asarray(
            self.config.xyz_control.max_dema_target_offset_m, dtype=float
        ).reshape(3)
        clipped = np.clip(offset_lcs0, -limit, +limit)
        saturated = not np.allclose(clipped, offset_lcs0, atol=1e-12, rtol=0.0)
        world = (
            self.initial_dema_position_world
            + self.initial_dema_rotation_world_lcs @ clipped
        )
        return world, saturated

    def _update_control(self, measurement: LocalizationMeasurement) -> None:
        geometry = self.plant.reported_geometry()
        measured_position_world, measured_moment_world = self._measurement_to_world(
            measurement
        )

        candidate = self.pid.prepare(
            measured_position_world_m=measured_position_world
        )
        desired_force_lcs = (
            geometry.lcs_pose.rotation.T
            @ candidate.desired_force_candidate_world_N
        )

        allocation = self.allocator.allocate(
            desired_force_lcs_N=desired_force_lcs,
            capsule_position_world=measured_position_world,
            capsule_moment_world=measured_moment_world,
            geometry=geometry,
            gamma_current=self.gamma,
        )

        # Move the DEMA target only incrementally.  This target itself is a
        # slow reference trajectory; the Cartesian servo adds acceleration
        # limiting and differential IK beneath it.
        target_increment_world = (
            geometry.lcs_pose.rotation
            @ allocation.dema_translation_increment_lcs_m
        )

        # IMPORTANT: the allocator returns the DEMA displacement required
        # relative to the *current physical/reported DEMA pose*.  It is not
        # a velocity command and must therefore not be integrated into the
        # previous DEMA target.  Integrating it makes the target run away
        # whenever the slow AUBO servo lags behind the requested displacement.
        proposed_target = geometry.lcs_pose.position + target_increment_world
        self.dema_target_world_m, workspace_saturated = self._clip_dema_target(
            proposed_target
        )
        robot_step = self.robot.command_position(
            target_position_world_m=self.dema_target_world_m,
            dt=self.control_dt,
        )

        self.gamma = allocation.gamma_command
        self.current_command_A = allocation.current_command_A.copy()

        # Freeze integral action if the local force request cannot be met or
        # if the DEMA reference hits the local validation workspace bound.
        accept_integral = not (
            bool(np.any(allocation.force_residual_saturated_axes))
            or workspace_saturated
        )
        pid_output = self.pid.finalize(
            accept_integral_axes=np.full(3, accept_integral, dtype=bool)
        )

        self.last_allocation = allocation
        self.last_robot_step = robot_step
        self.last_pid_output = pid_output

    def _record_control_sample(self) -> None:
        state = self.plant.true_state()
        measured_world, _ = self._measurement_to_world(self.last_measurement)
        geometry = self.plant.reported_geometry()
        supply = self.plant.power_supply.last_sample

        self._time_log.append(self.plant.time_s)
        self._true_position_log.append(state.position_world.copy())
        self._measured_position_log.append(measured_world.copy())
        self._error_log.append((self.target_position_world_m - state.position_world).copy())
        self._dema_position_log.append(geometry.lcs_pose.position.copy())
        self._dema_target_log.append(self.dema_target_world_m.copy())
        self._gamma_log.append(float(self.gamma))
        self._current_command_log.append(self.current_command_A.copy())
        self._current_actual_log.append(supply.actual_A.copy())

    def _physics_step(self) -> None:
        snapshot = self.plant.step(self.current_command_A)
        self._sum_sq_current_noise += snapshot.power_supply.noise_A**2
        self._current_noise_samples += 1

        step_index = int(round(self.plant.time_s / self.physics_dt))
        if step_index % self.control_every == 0:
            self.last_measurement = self.plant.sample_localization()
            self._update_control(self.last_measurement)
            self._record_control_sample()

    def run(
        self,
        *,
        viewer: bool = False,
        camera_mode: str = "overview",
    ) -> XYZSimulationResult:
        if camera_mode not in {"capsule", "overview"}:
            raise ValueError("camera_mode must be 'capsule' or 'overview'")

        print("Physics dt [s] =", self.physics_dt)
        print("Control dt [s] =", self.control_dt)
        print("AUBO max speed [mm/s] =", 1000.0 * self.config.aubo_motion.max_linear_speed_mps)
        print("Noise =", self.noise.enabled_names() or ("none",))
        print("Seed =", self.seed)

        if viewer:
            self._run_with_viewer(camera_mode=camera_mode)
        else:
            for _ in range(self.number_of_physics_steps):
                self._physics_step()
        return self._build_result()

    def _run_with_viewer(self, *, camera_mode: str) -> None:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            self.plant.system.model,
            self.plant.system.data,
        ) as viewer:
            if camera_mode == "capsule":
                configure_capsule_tracking_camera(
                    viewer,
                    self.plant.system,
                    distance=0.18,
                    azimuth=90.0,
                    elevation=-20.0,
                )
            else:
                configure_dema_overview_camera(
                    viewer,
                    self.plant.system,
                    distance=0.45,
                    azimuth=90.0,
                    elevation=-25.0,
                )
            viewer.sync()

            for _ in range(self.number_of_physics_steps):
                if not viewer.is_running():
                    break
                wall_start = time.perf_counter()
                self._physics_step()
                viewer.sync()
                sleep_s = self.physics_dt - (time.perf_counter() - wall_start)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

    def _build_result(self) -> XYZSimulationResult:
        time_s = np.asarray(self._time_log, dtype=float)
        true_position = np.asarray(self._true_position_log, dtype=float)
        measured_position = np.asarray(self._measured_position_log, dtype=float)
        position_error = self.target_position_world_m[None, :] - true_position
        dema_position = np.asarray(self._dema_position_log, dtype=float)
        dema_target = np.asarray(self._dema_target_log, dtype=float)
        gamma = np.asarray(self._gamma_log, dtype=float)
        current_command = np.asarray(self._current_command_log, dtype=float)
        current_actual = np.asarray(self._current_actual_log, dtype=float)

        mask = time_s >= self.settling_time_s
        if not np.any(mask):
            mask = np.ones_like(time_s, dtype=bool)

        rms_axis = np.sqrt(np.mean(position_error[mask] ** 2, axis=0))
        error_norm = np.linalg.norm(position_error, axis=1)
        rms_norm = float(np.sqrt(np.mean(error_norm[mask] ** 2)))
        max_norm = float(np.max(error_norm[mask]))

        localization_residual_world = measured_position - true_position
        rms_localization = np.sqrt(
            np.mean(localization_residual_world[mask] ** 2, axis=0)
        )

        if self._current_noise_samples > 0:
            rms_current_noise = np.sqrt(
                self._sum_sq_current_noise / self._current_noise_samples
            )
        else:
            rms_current_noise = np.zeros(2, dtype=float)

        return XYZSimulationResult(
            noise=self.noise,
            seed=self.seed,
            simulation_time_s=self.simulation_time_s,
            settling_time_s=self.settling_time_s,
            target_position_world_m=self.target_position_world_m.copy(),
            time_s=time_s,
            true_position_world_m=true_position,
            measured_position_world_m=measured_position,
            position_error_world_m=position_error,
            dema_position_world_m=dema_position,
            dema_target_world_m=dema_target,
            gamma=gamma,
            current_command_A=current_command,
            current_actual_A=current_actual,
            robot_translation_bias_world_m=(
                self.plant.robot_error.translation_world_m.copy()
            ),
            rms_true_error_world_m=rms_axis,
            rms_true_error_norm_m=rms_norm,
            max_true_error_norm_m=max_norm,
            rms_localization_residual_world_m=rms_localization,
            rms_current_noise_A=rms_current_noise,
            final_true_position_world_m=true_position[-1].copy(),
            final_measured_position_world_m=measured_position[-1].copy(),
            final_dema_position_world_m=dema_position[-1].copy(),
            final_gamma=float(gamma[-1]),
            final_current_command_A=current_command[-1].copy(),
            final_current_actual_A=current_actual[-1].copy(),
        )
