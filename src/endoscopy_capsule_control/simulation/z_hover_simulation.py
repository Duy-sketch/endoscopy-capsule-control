"""Closed-loop Z-hover experiment built on the refactored physical plant."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from endoscopy_capsule_control.config import AppConfig, DEFAULT_CONFIG
from endoscopy_capsule_control.control import ZHoverControlInput, ZHoverController
from endoscopy_capsule_control.plant.localization import LocalizationMeasurement
from endoscopy_capsule_control.plant.noise import NoiseSwitches
from endoscopy_capsule_control.plant.z_hover_plant import ZHoverPlant

from .dema_geometry import lcs_to_world_position
from .viewer import (
    configure_capsule_tracking_camera,
    configure_dema_overview_camera,
)


@dataclass(frozen=True)
class ZHoverSimulationResult:
    """Compact logged result for one Z-hover run."""

    noise: NoiseSwitches
    seed: int
    simulation_time_s: float
    settling_time_s: float

    time_s: np.ndarray
    true_z_m: np.ndarray
    measured_z_m: np.ndarray
    z_ref_m: np.ndarray
    current_command_A: np.ndarray
    current_actual_A: np.ndarray
    current_noise_A: np.ndarray

    robot_translation_bias_world_m: np.ndarray

    rms_true_z_error_m: float
    max_abs_true_z_error_m: float
    rms_localization_z_residual_m: float
    rms_current_noise_A: np.ndarray

    final_true_z_m: float
    final_measured_z_m: float
    final_current_command_A: np.ndarray
    final_current_actual_A: np.ndarray

    def print_summary(self) -> None:
        enabled = self.noise.enabled_names()
        noise_text = ", ".join(enabled) if enabled else "none"

        print("\n======================================")
        print("Z-HOVER PLANT / CONTROLLER TEST")
        print("======================================")
        print("Noise enabled:", noise_text)
        print("Seed:", self.seed)
        print("Simulation time [s]:", self.simulation_time_s)
        print("Settling time [s]:", self.settling_time_s)

        print("\nAUBO i10 fixed pose bias XYZ [mm]:")
        print(1000.0 * self.robot_translation_bias_world_m)

        print("\nPost-settling true Z RMS error [mm]:")
        print(1000.0 * self.rms_true_z_error_m)
        print("Post-settling max |true Z error| [mm]:")
        print(1000.0 * self.max_abs_true_z_error_m)
        print("Post-settling localization Z residual RMS [mm]:")
        print(1000.0 * self.rms_localization_z_residual_m)

        print("\nPower-supply current-noise RMS [mA], channels 1/2:")
        print(1000.0 * self.rms_current_noise_A)

        print("\nFinal true Z [mm]:", 1000.0 * self.final_true_z_m)
        print("Final measured Z [mm]:", 1000.0 * self.final_measured_z_m)
        print("Final current command [A]:", self.final_current_command_A)
        print("Final current actual [A]:", self.final_current_actual_A)


class ZHoverSimulation:
    """Run a Z-only hover controller against the physical plant."""

    def __init__(
        self,
        *,
        config: AppConfig = DEFAULT_CONFIG,
        noise: NoiseSwitches | None = None,
        seed: int = 1,
        simulation_time_s: float = 10.0,
        settling_time_s: float = 5.0,
        x_offset_m: float = 0.0,
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

        self.plant = ZHoverPlant(
            config=self.config,
            noise=self.noise,
            seed=self.seed,
            x_offset_m=float(x_offset_m),
            z_offset_m=float(z_offset_m),
            theta_y_offset_rad=float(theta_y_offset_rad),
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
            raise RuntimeError(
                "control_dt must be an integer multiple of physics_dt"
            )

        self.number_of_physics_steps = int(
            round(self.simulation_time_s / self.physics_dt)
        )

        self.controller = ZHoverController(
            mass=self.config.capsule.mass,
            drag_coefficient=self.plant.drag_coefficient,
            control_dt=self.control_dt,
            z_ref=self.config.z_control.z_ref,
            natural_frequency=self.config.z_control.natural_frequency,
            damping_ratio=self.config.z_control.damping_ratio,
            integral_pole=self.config.z_control.integral_pole,
            derivative_filter_tau=self.config.z_control.derivative_filter_tau,
            integral_limit=self.config.z_control.integral_limit,
            pid_force_limit=self.config.z_control.pid_force_limit,
            feedforward_force_z=self.config.z_control.feedforward_force,
            desired_force_z_min=self.config.z_control.desired_force_min,
            desired_force_z_max=self.config.z_control.desired_force_max,
            current_abs_max=self.config.electromagnet.current_abs_max,
            differential_current_min=(
                self.config.electromagnet.differential_current_min
            ),
            differential_current_max=(
                self.config.electromagnet.differential_current_max
            ),
            gradient_step=self.config.electromagnet.magnetic_gradient_step,
            em_gain=self.config.electromagnet.magnetic_gain,
        )

        self.current_command_A = np.array([-15.0, 15.0], dtype=float)
        self.last_measurement = self.plant.sample_localization()
        self.controller.reset(
            initial_z=float(self.last_measurement.position_lcs_m[2])
        )
        self._update_controller(self.last_measurement)

        self._time_log: list[float] = []
        self._true_z_log: list[float] = []
        self._measured_z_log: list[float] = []
        self._z_ref_log: list[float] = []
        self._current_command_log: list[np.ndarray] = []
        self._current_actual_log: list[np.ndarray] = []
        self._current_noise_control_log: list[np.ndarray] = []

        self._sum_sq_current_noise = np.zeros(2, dtype=float)
        self._current_noise_samples = 0

        self._record_control_sample()

    def _make_control_input(
        self,
        measurement: LocalizationMeasurement,
    ) -> ZHoverControlInput:
        # Controller uses reported/nominal AUBO kinematics, not the hidden
        # true robot-pose perturbation used by the physical magnetic plant.
        geometry = self.plant.reported_geometry()

        measured_position_lcs = measurement.position_lcs_m.copy()
        measured_position_world = lcs_to_world_position(
            measured_position_lcs,
            geometry.lcs_pose,
        )

        theta_y = float(measurement.theta_y_rad)
        m = float(self.plant.magnetic_moment_magnitude)
        measured_moment_lcs = m * np.array(
            [np.cos(theta_y), 0.0, -np.sin(theta_y)],
            dtype=float,
        )
        measured_moment_world = (
            geometry.lcs_pose.rotation @ measured_moment_lcs
        )

        return ZHoverControlInput(
            z_measured=float(measured_position_lcs[2]),
            capsule_position_world=measured_position_world,
            magnetic_moment_world=measured_moment_world,
            em_positions_world=(
                geometry.em1.position_world,
                geometry.em2.position_world,
            ),
            em_axes_world=(
                geometry.em1.axis_world,
                geometry.em2.axis_world,
            ),
            control_axis_world=geometry.lcs_pose.rotation[:, 2],
        )

    def _update_controller(self, measurement: LocalizationMeasurement) -> None:
        output = self.controller.update(
            self._make_control_input(measurement)
        )
        self.current_command_A = output.current_command.copy()

    def _record_control_sample(self) -> None:
        state = self.plant.true_state()
        sample = self.plant.power_supply.last_sample

        self._time_log.append(self.plant.time_s)
        self._true_z_log.append(float(state.position_lcs[2]))
        self._measured_z_log.append(
            float(self.last_measurement.position_lcs_m[2])
        )
        self._z_ref_log.append(float(self.config.z_control.z_ref))
        self._current_command_log.append(self.current_command_A.copy())
        self._current_actual_log.append(sample.actual_A.copy())
        self._current_noise_control_log.append(sample.noise_A.copy())

    def _physics_step(self) -> None:
        snapshot = self.plant.step(self.current_command_A)
        self._sum_sq_current_noise += snapshot.power_supply.noise_A**2
        self._current_noise_samples += 1

        step_index = int(round(self.plant.time_s / self.physics_dt))
        if step_index % self.control_every == 0:
            self.last_measurement = self.plant.sample_localization()
            self._update_controller(self.last_measurement)
            self._record_control_sample()

    def run(
        self,
        *,
        viewer: bool = False,
        camera_mode: str = "capsule",
    ) -> ZHoverSimulationResult:
        if camera_mode not in {"capsule", "overview"}:
            raise ValueError("camera_mode must be 'capsule' or 'overview'")

        print("Physics dt [s] =", self.physics_dt)
        print("Control dt [s] =", self.control_dt)
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

    def _build_result(self) -> ZHoverSimulationResult:
        time_s = np.asarray(self._time_log, dtype=float)
        true_z = np.asarray(self._true_z_log, dtype=float)
        measured_z = np.asarray(self._measured_z_log, dtype=float)
        z_ref = np.asarray(self._z_ref_log, dtype=float)
        current_command = np.asarray(self._current_command_log, dtype=float)
        current_actual = np.asarray(self._current_actual_log, dtype=float)
        current_noise = np.asarray(self._current_noise_control_log, dtype=float)

        mask = time_s >= self.settling_time_s
        if not np.any(mask):
            mask = np.ones_like(time_s, dtype=bool)

        true_error = z_ref - true_z
        localization_residual_z = measured_z - true_z

        rms_true_z_error = float(np.sqrt(np.mean(true_error[mask] ** 2)))
        max_abs_true_z_error = float(np.max(np.abs(true_error[mask])))
        rms_localization_z_residual = float(
            np.sqrt(np.mean(localization_residual_z[mask] ** 2))
        )

        if self._current_noise_samples > 0:
            rms_current_noise = np.sqrt(
                self._sum_sq_current_noise / self._current_noise_samples
            )
        else:
            rms_current_noise = np.zeros(2, dtype=float)

        return ZHoverSimulationResult(
            noise=self.noise,
            seed=self.seed,
            simulation_time_s=self.simulation_time_s,
            settling_time_s=self.settling_time_s,
            time_s=time_s,
            true_z_m=true_z,
            measured_z_m=measured_z,
            z_ref_m=z_ref,
            current_command_A=current_command,
            current_actual_A=current_actual,
            current_noise_A=current_noise,
            robot_translation_bias_world_m=(
                self.plant.robot_error.translation_world_m.copy()
            ),
            rms_true_z_error_m=rms_true_z_error,
            max_abs_true_z_error_m=max_abs_true_z_error,
            rms_localization_z_residual_m=rms_localization_z_residual,
            rms_current_noise_A=rms_current_noise,
            final_true_z_m=float(true_z[-1]),
            final_measured_z_m=float(measured_z[-1]),
            final_current_command_A=current_command[-1].copy(),
            final_current_actual_A=current_actual[-1].copy(),
        )
