"""
Hover simulation orchestration.

True MuJoCo state
    |
    +--> physical plant
    |
    +--> measurement model
    |       |
    |       v
    |   noisy + delayed measurement
    |       |
    |       v
    |   HoverController
    |
    +--> true-state diagnostics

Physics always uses the true MuJoCo state.

Measurement noise and delay affect only the controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import mujoco.viewer
import numpy as np

from endoscopy_capsule_control.config import (
    DEFAULT_CONFIG,
)

from endoscopy_capsule_control.control import (
    HoverController,
    HoverControlInput,
    magnetic_moment_tilt_y,
)

from endoscopy_capsule_control.dynamics import (
    FirstOrderCurrentDynamics,
    capsule_volume,
    dynamic_viscosity,
    stokes_drag_coefficient,
)

from endoscopy_capsule_control.magnetic import (
    magnetic_wrench,
)

from endoscopy_capsule_control.sensing import (
    Measurement,
    MeasurementModel,
)

from .capsule_state import (
    get_capsule_state,
)

from .capsule_wrench import (
    apply_capsule_external_wrench,
    compute_capsule_external_wrench,
)

from .dema_geometry import (
    get_dema_geometry,
    lcs_to_world_position,
)

from .diagnostics import (
    PostSettlingDiagnostics,
    PostSettlingMetrics,
)

from .initialization import (
    initialize_hover_operating_point,
)

from .mujoco_system import (
    MujocoSystem,
)

from .perturbation import (
    apply_initial_perturbation,
)

from .viewer import (
    configure_capsule_tracking_camera,
    configure_dema_overview_camera,
)


# ============================================================
# RESULT
# ============================================================


@dataclass(frozen=True)
class HoverSimulationResult:
    simulation_time: float

    initial_position_lcs: np.ndarray
    final_position_lcs: np.ndarray
    final_position_error_lcs: np.ndarray

    initial_theta_y: float
    final_theta_y: float

    max_abs_z_error: float
    max_abs_x: float
    max_abs_y: float
    max_abs_theta_y: float

    max_abs_ic: float
    max_abs_id: float

    current_command: np.ndarray
    current_actual: np.ndarray

    desired_force_z: float

    final_ic: float
    final_id: float

    allocator_saturated: bool

    requested_force_z: float
    achieved_force_z: float

    final_linear_velocity_lcs: np.ndarray
    final_angular_velocity_lcs: np.ndarray

    controller_updates: int
    physics_steps: int

    sensing_enabled: bool

    position_noise_std: float
    theta_y_noise_std: float

    measurement_delay_samples: int
    measurement_seed: int
    measurement_samples: int

    rms_position_residual_lcs: np.ndarray
    max_abs_position_residual_lcs: np.ndarray

    rms_theta_y_residual: float
    max_abs_theta_y_residual: float

    final_measured_position_lcs: np.ndarray
    final_measured_theta_y: float

    post_settling: PostSettlingMetrics

    def print_summary(
        self,
    ) -> None:
        print(
            "\n======================================"
        )
        print(
            "HOVER + SENSOR ROBUSTNESS TEST"
        )
        print(
            "======================================"
        )

        print(
            "Simulation time [s] =",
            self.simulation_time,
        )

        print(
            "\nInitial position LCS [m] =",
            self.initial_position_lcs,
        )

        print(
            "Final true position LCS [m] =",
            self.final_position_lcs,
        )

        print(
            "Final true position error [mm] =",
            1000.0
            * self.final_position_error_lcs,
        )

        print(
            "\nInitial true theta_y [deg] =",
            np.rad2deg(
                self.initial_theta_y
            ),
        )

        print(
            "Final true theta_y [deg] =",
            np.rad2deg(
                self.final_theta_y
            ),
        )

        print(
            "\nMax |true z error| [mm] =",
            1000.0
            * self.max_abs_z_error,
        )

        print(
            "Max |true x| [mm] =",
            1000.0
            * self.max_abs_x,
        )

        print(
            "Max |true y| [mm] =",
            1000.0
            * self.max_abs_y,
        )

        print(
            "Max |true theta_y| [deg] =",
            np.rad2deg(
                self.max_abs_theta_y
            ),
        )

        print(
            "\nMax |Ic| [A] =",
            self.max_abs_ic,
        )

        print(
            "Max |Id| [A] =",
            self.max_abs_id,
        )

        print(
            "\nCurrent command [A] =",
            self.current_command,
        )

        print(
            "Current actual [A] =",
            self.current_actual,
        )

        print(
            "Desired Fz [mN] =",
            1000.0
            * self.desired_force_z,
        )

        print(
            "Final Ic [A] =",
            self.final_ic,
        )

        print(
            "Final Id [A] =",
            self.final_id,
        )

        print(
            "Allocator saturated =",
            self.allocator_saturated,
        )

        print(
            "Requested Fz [mN] =",
            1000.0
            * self.requested_force_z,
        )

        print(
            "Achieved Fz [mN] =",
            1000.0
            * self.achieved_force_z,
        )

        print(
            "\nFinal linear velocity LCS [m/s] =",
            self.final_linear_velocity_lcs,
        )

        print(
            "Final angular velocity LCS [rad/s] =",
            self.final_angular_velocity_lcs,
        )

        # ====================================================
        # POST-SETTLING TRUE PERFORMANCE
        # ====================================================

        ps = self.post_settling

        print(
            "\n--------------------------------------"
        )
        print(
            "POST-SETTLING TRUE-STATE METRICS"
        )
        print(
            "--------------------------------------"
        )

        print(
            "Settling time [s] =",
            ps.settling_time,
        )

        print(
            "Samples =",
            ps.samples,
        )

        print(
            "\nMean x [mm] =",
            1000.0
            * ps.mean_x,
        )

        print(
            "RMS x [mm] =",
            1000.0
            * ps.rms_x,
        )

        print(
            "Std x [mm] =",
            1000.0
            * ps.std_x,
        )

        print(
            "\nMean z error [mm] =",
            1000.0
            * ps.mean_z_error,
        )

        print(
            "RMS z error [mm] =",
            1000.0
            * ps.rms_z_error,
        )

        print(
            "Std z error [mm] =",
            1000.0
            * ps.std_z_error,
        )

        print(
            "\nMean theta_y [deg] =",
            np.rad2deg(
                ps.mean_theta_y
            ),
        )

        print(
            "RMS theta_y [deg] =",
            np.rad2deg(
                ps.rms_theta_y
            ),
        )

        print(
            "Std theta_y [deg] =",
            np.rad2deg(
                ps.std_theta_y
            ),
        )

        # ====================================================
        # SENSOR
        # ====================================================

        print(
            "\n--------------------------------------"
        )
        print(
            "SENSING"
        )
        print(
            "--------------------------------------"
        )

        print(
            "Sensing enabled =",
            self.sensing_enabled,
        )

        print(
            "Position noise sigma [mm/axis] =",
            1000.0
            * self.position_noise_std,
        )

        print(
            "theta_y noise sigma [deg] =",
            np.rad2deg(
                self.theta_y_noise_std
            ),
        )

        print(
            "Measurement delay [samples] =",
            self.measurement_delay_samples,
        )

        print(
            "Measurement seed =",
            self.measurement_seed,
        )

        print(
            "Measurement samples =",
            self.measurement_samples,
        )

        print(
            "\nRMS measurement residual XYZ [mm] =",
            1000.0
            * self.rms_position_residual_lcs,
        )

        print(
            "Max |measurement residual XYZ| [mm] =",
            1000.0
            * self.max_abs_position_residual_lcs,
        )

        print(
            "RMS theta_y residual [deg] =",
            np.rad2deg(
                self.rms_theta_y_residual
            ),
        )

        print(
            "Max |theta_y residual| [deg] =",
            np.rad2deg(
                self.max_abs_theta_y_residual
            ),
        )

        print(
            "\nFinal measured position LCS [m] =",
            self.final_measured_position_lcs,
        )

        print(
            "Final measured theta_y [deg] =",
            np.rad2deg(
                self.final_measured_theta_y
            ),
        )

        print(
            "\nController updates =",
            self.controller_updates,
        )

        print(
            "Physics steps =",
            self.physics_steps,
        )


# ============================================================
# SIMULATION
# ============================================================


class HoverSimulation:
    def __init__(
        self,
        *,
        simulation_time: float = 10.0,
        settling_time: float = 5.0,
        x_offset: float = 0.0,
        z_offset: float = 0.0,
        theta_y_offset: float = 0.0,
        local_kx: float = 2.5,
        local_ktheta: float = 1.0,
        sensing_enabled: bool = True,
        position_noise_std: float = 0.00036,
        theta_y_noise_std: float = np.deg2rad(
            0.20
        ),
        measurement_delay_samples: int = 1,
        measurement_seed: int = 1,
    ):
        if simulation_time <= 0.0:
            raise ValueError(
                "simulation_time must be positive."
            )

        if settling_time < 0.0:
            raise ValueError(
                "settling_time must be non-negative."
            )

        if settling_time >= simulation_time:
            raise ValueError(
                "settling_time must be smaller "
                "than simulation_time."
            )

        self.cfg = DEFAULT_CONFIG

        self.simulation_time = float(
            simulation_time
        )

        self.settling_time = float(
            settling_time
        )

        self.x_offset = float(
            x_offset
        )

        self.z_offset = float(
            z_offset
        )

        self.theta_y_offset = float(
            theta_y_offset
        )

        self.local_kx = float(
            local_kx
        )

        self.local_ktheta = float(
            local_ktheta
        )

        self.sensing_enabled = bool(
            sensing_enabled
        )

        if self.sensing_enabled:
            self.position_noise_std = float(
                position_noise_std
            )

            self.theta_y_noise_std = float(
                theta_y_noise_std
            )

            self.measurement_delay_samples = int(
                measurement_delay_samples
            )

        else:
            self.position_noise_std = 0.0
            self.theta_y_noise_std = 0.0
            self.measurement_delay_samples = 0

        self.measurement_seed = int(
            measurement_seed
        )

        # ====================================================
        # MUJOCO
        # ====================================================

        self.system = MujocoSystem()

        initialize_hover_operating_point(
            self.system
        )

        self.physics_dt = float(
            self.system.timestep
        )

        self.control_dt = float(
            self.cfg.simulation.control_dt
        )

        self.control_every = int(
            round(
                self.control_dt
                / self.physics_dt
            )
        )

        if self.control_every <= 0:
            raise RuntimeError(
                "Invalid controller / physics timestep ratio."
            )

        if not np.isclose(
            self.control_every
            * self.physics_dt,
            self.control_dt,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                "control_dt must be an integer multiple "
                "of physics_dt."
            )

        self.number_of_physics_steps = int(
            round(
                self.simulation_time
                / self.physics_dt
            )
        )

        # ====================================================
        # CAPSULE + FLUID
        # ====================================================

        self.magnetic_moment_body = np.array(
            self.cfg.capsule.moment_body,
            dtype=float,
        )

        self.magnetic_moment_magnitude = float(
            np.linalg.norm(
                self.magnetic_moment_body
            )
        )

        if self.magnetic_moment_magnitude <= 0.0:
            raise RuntimeError(
                "Capsule magnetic moment must be non-zero."
            )

        self.capsule_volume_value = (
            capsule_volume(
                length=(
                    self.cfg.capsule.length
                ),
                diameter=(
                    self.cfg.capsule.diameter
                ),
            )
        )

        dynamic_viscosity_value = (
            dynamic_viscosity(
                fluid_density=(
                    self.cfg.fluid.density
                ),
                kinematic_viscosity=(
                    self.cfg.fluid
                    .kinematic_viscosity
                ),
            )
        )

        self.drag_coefficient = (
            stokes_drag_coefficient(
                dynamic_viscosity_value=(
                    dynamic_viscosity_value
                ),
                diameter=(
                    self.cfg.capsule.diameter
                ),
            )
        )

        # ====================================================
        # INITIAL PERTURBATION
        # ====================================================

        apply_initial_perturbation(
            system=self.system,
            magnetic_moment_body=(
                self.magnetic_moment_body
            ),
            x_offset=(
                self.x_offset
            ),
            z_offset=(
                self.z_offset
            ),
            theta_y_offset=(
                self.theta_y_offset
            ),
        )

        # ====================================================
        # INITIAL TRUE STATE
        # ====================================================

        initial_geometry = get_dema_geometry(
            self.system
        )

        initial_state = get_capsule_state(
            system=self.system,
            lcs_pose=(
                initial_geometry.lcs_pose
            ),
            magnetic_moment_body=(
                self.magnetic_moment_body
            ),
        )

        self.initial_position_lcs = (
            initial_state
            .position_lcs
            .copy()
        )

        initial_moment_lcs = (
            initial_geometry
            .lcs_pose
            .rotation
            .T
            @ initial_state
            .magnetic_moment_world
        )

        self.initial_theta_y = (
            magnetic_moment_tilt_y(
                initial_moment_lcs
            )
        )

        # ====================================================
        # SENSOR
        # ====================================================

        self.measurement_model = (
            MeasurementModel(
                position_noise_std=(
                    self.position_noise_std
                ),
                theta_y_noise_std=(
                    self.theta_y_noise_std
                ),
                delay_samples=(
                    self.measurement_delay_samples
                ),
                seed=(
                    self.measurement_seed
                ),
            )
        )

        self.measurement_samples = 0

        self.sum_sq_position_residual = (
            np.zeros(
                3,
                dtype=float,
            )
        )

        self.max_abs_position_residual = (
            np.zeros(
                3,
                dtype=float,
            )
        )

        self.sum_sq_theta_residual = 0.0
        self.max_abs_theta_residual = 0.0

        self.last_measurement = None

        initial_measurement = (
            self._measure(
                geometry=(
                    initial_geometry
                ),
                state=(
                    initial_state
                ),
            )
        )

        # ====================================================
        # CONTROLLER
        # ====================================================

        self.controller = HoverController(
            mass=(
                self.cfg.capsule.mass
            ),
            drag_coefficient=(
                self.drag_coefficient
            ),
            control_dt=(
                self.control_dt
            ),
            z_ref=(
                self.cfg.z_control.z_ref
            ),
            feedforward_force=(
                self.cfg.z_control
                .feedforward_force
            ),
            desired_force_min=(
                self.cfg.z_control
                .desired_force_min
            ),
            desired_force_max=(
                self.cfg.z_control
                .desired_force_max
            ),
            natural_frequency=(
                self.cfg.z_control
                .natural_frequency
            ),
            damping_ratio=(
                self.cfg.z_control
                .damping_ratio
            ),
            integral_pole=(
                self.cfg.z_control
                .integral_pole
            ),
            derivative_filter_tau=(
                self.cfg.z_control
                .derivative_filter_tau
            ),
            integral_limit=(
                self.cfg.z_control
                .integral_limit
            ),
            pid_force_limit=(
                self.cfg.z_control
                .pid_force_limit
            ),
            local_kx=(
                self.local_kx
            ),
            local_ktheta=(
                self.local_ktheta
            ),
            x_ref=0.0,
            current_abs_max=(
                self.cfg.electromagnet
                .current_abs_max
            ),
            differential_current_min=(
                self.cfg.electromagnet
                .differential_current_min
            ),
            differential_current_max=(
                self.cfg.electromagnet
                .differential_current_max
            ),
            common_current_abs_max=(
                self.cfg.electromagnet
                .common_current_abs_max
            ),
            magnetic_gradient_step=(
                self.cfg.electromagnet
                .magnetic_gradient_step
            ),
            magnetic_gain=(
                self.cfg.electromagnet
                .magnetic_gain
            ),
        )

        self.controller.reset(
            initial_z=float(
                initial_measurement
                .position_lcs[2]
            )
        )

        # ====================================================
        # CURRENT DYNAMICS
        # ====================================================

        self.current_dynamics = (
            FirstOrderCurrentDynamics(
                time_constant=(
                    self.cfg.electromagnet
                    .current_time_constant
                ),
                dt=(
                    self.physics_dt
                ),
                current_limit=(
                    self.cfg.electromagnet
                    .current_abs_max
                ),
            )
        )

        self.current_actual = np.array(
            [
                -15.0,
                +15.0,
            ],
            dtype=float,
        )

        self.current_command = (
            self.current_actual.copy()
        )

        # ====================================================
        # GLOBAL TRUE-STATE METRICS
        # ====================================================

        self.physics_step_count = 0
        self.controller_step_count = 0

        self.max_abs_z_error = abs(
            self.cfg.z_control.z_ref
            - float(
                initial_state.position_lcs[2]
            )
        )

        self.max_abs_x = abs(
            float(
                initial_state.position_lcs[0]
            )
        )

        self.max_abs_y = abs(
            float(
                initial_state.position_lcs[1]
            )
        )

        self.max_abs_theta_y = abs(
            self.initial_theta_y
        )

        self.max_abs_ic = 0.0
        self.max_abs_id = 0.0

        self.last_control_output = None

        # ====================================================
        # POST-SETTLING TRUE-STATE DIAGNOSTICS
        # ====================================================

        self.post_settling_diagnostics = (
            PostSettlingDiagnostics(
                settling_time=(
                    self.settling_time
                )
            )
        )

        # ====================================================
        # INITIAL CONTROLLER UPDATE
        # ====================================================

        self._update_controller(
            measurement=(
                initial_measurement
            )
        )

    # ========================================================
    # SENSOR
    # ========================================================

    def _record_measurement(
        self,
        measurement: Measurement,
    ) -> None:
        residual = (
            measurement
            .position_residual_lcs
        )

        self.sum_sq_position_residual += (
            residual**2
        )

        self.max_abs_position_residual = (
            np.maximum(
                self.max_abs_position_residual,
                np.abs(
                    residual
                ),
            )
        )

        theta_residual = float(
            measurement
            .theta_y_residual
        )

        self.sum_sq_theta_residual += (
            theta_residual**2
        )

        self.max_abs_theta_residual = max(
            self.max_abs_theta_residual,
            abs(
                theta_residual
            ),
        )

        self.measurement_samples += 1

    def _measure(
        self,
        *,
        geometry,
        state,
    ) -> Measurement:
        true_moment_lcs = (
            geometry
            .lcs_pose
            .rotation
            .T
            @ state
            .magnetic_moment_world
        )

        true_theta_y = (
            magnetic_moment_tilt_y(
                true_moment_lcs
            )
        )

        measurement = (
            self.measurement_model.sample(
                true_position_lcs=(
                    state.position_lcs
                ),
                true_theta_y=(
                    true_theta_y
                ),
            )
        )

        self.last_measurement = (
            measurement
        )

        self._record_measurement(
            measurement
        )

        return measurement

    # ========================================================
    # CONTROL INPUT
    # ========================================================

    def _make_control_input(
        self,
        *,
        geometry,
        measurement: Measurement,
    ) -> HoverControlInput:
        measured_position_lcs = (
            measurement
            .position_lcs
            .copy()
        )

        measured_position_world = (
            lcs_to_world_position(
                measured_position_lcs,
                geometry.lcs_pose,
            )
        )

        theta_y = float(
            measurement.theta_y
        )

        measured_moment_lcs = (
            self.magnetic_moment_magnitude
            * np.array(
                [
                    np.cos(
                        theta_y
                    ),
                    0.0,
                    -np.sin(
                        theta_y
                    ),
                ],
                dtype=float,
            )
        )

        measured_moment_world = (
            geometry
            .lcs_pose
            .rotation
            @ measured_moment_lcs
        )

        return HoverControlInput(
            position_lcs=(
                measured_position_lcs
            ),
            position_world=(
                measured_position_world
            ),
            magnetic_moment_lcs=(
                measured_moment_lcs
            ),
            magnetic_moment_world=(
                measured_moment_world
            ),
            em_positions_world=(
                geometry.em1.position_world,
                geometry.em2.position_world,
            ),
            em_axes_world=(
                geometry.em1.axis_world,
                geometry.em2.axis_world,
            ),
            control_axis_world=(
                geometry
                .lcs_pose
                .rotation[:, 2]
            ),
        )

    # ========================================================
    # CONTROLLER
    # ========================================================

    def _update_controller(
        self,
        *,
        measurement: Measurement | None = None,
    ) -> None:
        geometry = get_dema_geometry(
            self.system
        )

        state = get_capsule_state(
            system=self.system,
            lcs_pose=(
                geometry.lcs_pose
            ),
            magnetic_moment_body=(
                self.magnetic_moment_body
            ),
        )

        if measurement is None:
            measurement = (
                self._measure(
                    geometry=(
                        geometry
                    ),
                    state=(
                        state
                    ),
                )
            )

        control_input = (
            self._make_control_input(
                geometry=(
                    geometry
                ),
                measurement=(
                    measurement
                ),
            )
        )

        output = (
            self.controller.update(
                control_input
            )
        )

        self.current_command = (
            output
            .current_command
            .copy()
        )

        self.last_control_output = (
            output
        )

        self.max_abs_ic = max(
            self.max_abs_ic,
            abs(
                output
                .allocation
                .common_current
            ),
        )

        self.max_abs_id = max(
            self.max_abs_id,
            abs(
                output
                .allocation
                .differential_current
            ),
        )

        self.controller_step_count += 1

    # ========================================================
    # TRUE METRICS
    # ========================================================

    def _update_metrics(
        self,
    ) -> None:
        geometry = get_dema_geometry(
            self.system
        )

        state = get_capsule_state(
            system=self.system,
            lcs_pose=(
                geometry.lcs_pose
            ),
            magnetic_moment_body=(
                self.magnetic_moment_body
            ),
        )

        x = float(
            state.position_lcs[0]
        )

        y = float(
            state.position_lcs[1]
        )

        z = float(
            state.position_lcs[2]
        )

        z_error = (
            self.cfg.z_control.z_ref
            - z
        )

        moment_lcs = (
            geometry
            .lcs_pose
            .rotation
            .T
            @ state
            .magnetic_moment_world
        )

        theta_y = (
            magnetic_moment_tilt_y(
                moment_lcs
            )
        )

        self.max_abs_z_error = max(
            self.max_abs_z_error,
            abs(
                z_error
            ),
        )

        self.max_abs_x = max(
            self.max_abs_x,
            abs(
                x
            ),
        )

        self.max_abs_y = max(
            self.max_abs_y,
            abs(
                y
            ),
        )

        self.max_abs_theta_y = max(
            self.max_abs_theta_y,
            abs(
                theta_y
            ),
        )

        self.post_settling_diagnostics.update(
            time=(
                self.system.time
            ),
            x=(
                x
            ),
            z_error=(
                z_error
            ),
            theta_y=(
                theta_y
            ),
        )

    # ========================================================
    # PHYSICS
    # ========================================================

    def _step_physics(
        self,
    ) -> None:
        self.current_actual = (
            self.current_dynamics.step(
                actual_current=(
                    self.current_actual
                ),
                commanded_current=(
                    self.current_command
                ),
            )
        )

        geometry = get_dema_geometry(
            self.system
        )

        state = get_capsule_state(
            system=self.system,
            lcs_pose=(
                geometry.lcs_pose
            ),
            magnetic_moment_body=(
                self.magnetic_moment_body
            ),
        )

        (
            magnetic_force_world,
            magnetic_torque_world,
            _,
        ) = magnetic_wrench(
            p_capsule=(
                state.position_world
            ),
            capsule_moment_world=(
                state.magnetic_moment_world
            ),
            em_positions=[
                geometry.em1.position_world,
                geometry.em2.position_world,
            ],
            em_axes=[
                geometry.em1.axis_world,
                geometry.em2.axis_world,
            ],
            currents=(
                self.current_actual
            ),
            em_gain=(
                self.cfg.electromagnet
                .magnetic_gain
            ),
            gradient_step=(
                self.cfg.electromagnet
                .magnetic_gradient_step
            ),
        )

        external_wrench = (
            compute_capsule_external_wrench(
                magnetic_force_world=(
                    magnetic_force_world
                ),
                magnetic_torque_world=(
                    magnetic_torque_world
                ),
                linear_velocity_world=(
                    state.linear_velocity_world
                ),
                angular_velocity_world=(
                    state.angular_velocity_world
                ),
                fluid_density=(
                    self.cfg.fluid.density
                ),
                capsule_volume_value=(
                    self.capsule_volume_value
                ),
                gravity_world=(
                    self.system.model.opt.gravity
                ),
                translational_drag_coefficient=(
                    self.drag_coefficient
                ),
                rotational_damping=(
                    self.cfg.fluid
                    .rotational_damping
                ),
            )
        )

        apply_capsule_external_wrench(
            system=self.system,
            wrench=(
                external_wrench
            ),
        )

        self.system.step()

        self.physics_step_count += 1

        if (
            self.physics_step_count
            % self.control_every
            == 0
        ):
            self._update_controller()

        self._update_metrics()

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        *,
        viewer: bool = False,
        camera_mode: str = "capsule",
    ) -> HoverSimulationResult:
        if camera_mode not in {
            "capsule",
            "overview",
        }:
            raise ValueError(
                "camera_mode must be "
                "'capsule' or 'overview'."
            )

        print(
            "Physics dt [s] =",
            self.physics_dt,
        )

        print(
            "Control dt [s] =",
            self.control_dt,
        )

        print(
            "Control every physics steps =",
            self.control_every,
        )

        print(
            "Requested simulation time [s] =",
            self.simulation_time,
        )

        print(
            "Settling time [s] =",
            self.settling_time,
        )

        print(
            "Sensing enabled =",
            self.sensing_enabled,
        )

        print(
            "Position noise sigma [mm/axis] =",
            1000.0
            * self.position_noise_std,
        )

        print(
            "theta_y noise sigma [deg] =",
            np.rad2deg(
                self.theta_y_noise_std
            ),
        )

        print(
            "Measurement delay [samples] =",
            self.measurement_delay_samples,
        )

        self._print_initial_state()

        if viewer:
            self._run_with_viewer(
                camera_mode=(
                    camera_mode
                )
            )

        else:
            self._run_headless()

        return self._build_result()

    def _run_headless(
        self,
    ) -> None:
        for _ in range(
            self.number_of_physics_steps
        ):
            self._step_physics()

    def _run_with_viewer(
        self,
        *,
        camera_mode: str,
    ) -> None:
        with mujoco.viewer.launch_passive(
            self.system.model,
            self.system.data,
        ) as viewer:

            if camera_mode == "capsule":
                configure_capsule_tracking_camera(
                    viewer,
                    self.system,
                    distance=0.18,
                    azimuth=90.0,
                    elevation=-20.0,
                )

            else:
                configure_dema_overview_camera(
                    viewer,
                    self.system,
                    distance=0.45,
                    azimuth=90.0,
                    elevation=-25.0,
                )

            viewer.sync()

            for _ in range(
                self.number_of_physics_steps
            ):
                if not viewer.is_running():
                    break

                wall_start = (
                    time.perf_counter()
                )

                self._step_physics()

                viewer.sync()

                elapsed = (
                    time.perf_counter()
                    - wall_start
                )

                sleep_time = (
                    self.physics_dt
                    - elapsed
                )

                if sleep_time > 0.0:
                    time.sleep(
                        sleep_time
                    )

    # ========================================================
    # PRINT INITIAL STATE
    # ========================================================

    def _print_initial_state(
        self,
    ) -> None:
        print(
            "\n======================================"
        )

        print(
            "INITIAL TRUE STATE"
        )

        print(
            "======================================"
        )

        print(
            "Position LCS [m] =",
            self.initial_position_lcs,
        )

        print(
            "X displacement [mm] =",
            1000.0
            * self.initial_position_lcs[0],
        )

        print(
            "Z displacement from reference [mm] =",
            1000.0
            * (
                self.initial_position_lcs[2]
                - self.cfg.z_control.z_ref
            ),
        )

        print(
            "theta_y [deg] =",
            np.rad2deg(
                self.initial_theta_y
            ),
        )

    # ========================================================
    # RESULT
    # ========================================================

    def _build_result(
        self,
    ) -> HoverSimulationResult:
        geometry = get_dema_geometry(
            self.system
        )

        final_state = get_capsule_state(
            system=self.system,
            lcs_pose=(
                geometry.lcs_pose
            ),
            magnetic_moment_body=(
                self.magnetic_moment_body
            ),
        )

        final_moment_lcs = (
            geometry
            .lcs_pose
            .rotation
            .T
            @ final_state
            .magnetic_moment_world
        )

        final_theta_y = (
            magnetic_moment_tilt_y(
                final_moment_lcs
            )
        )

        reference = np.array(
            [
                0.0,
                0.0,
                self.cfg.z_control.z_ref,
            ],
            dtype=float,
        )

        final_position_error = (
            reference
            - final_state.position_lcs
        )

        if self.last_control_output is None:
            raise RuntimeError(
                "Controller has not produced an output."
            )

        if self.last_measurement is None:
            raise RuntimeError(
                "Measurement model has not produced a sample."
            )

        allocation = (
            self.last_control_output
            .allocation
        )

        n_measurements = max(
            self.measurement_samples,
            1,
        )

        rms_position_residual = np.sqrt(
            self.sum_sq_position_residual
            / float(
                n_measurements
            )
        )

        rms_theta_residual = float(
            np.sqrt(
                self.sum_sq_theta_residual
                / float(
                    n_measurements
                )
            )
        )

        post_settling = (
            self.post_settling_diagnostics
            .result()
        )

        return HoverSimulationResult(
            simulation_time=float(
                self.system.time
            ),
            initial_position_lcs=(
                self.initial_position_lcs.copy()
            ),
            final_position_lcs=(
                final_state.position_lcs.copy()
            ),
            final_position_error_lcs=(
                final_position_error.copy()
            ),
            initial_theta_y=float(
                self.initial_theta_y
            ),
            final_theta_y=float(
                final_theta_y
            ),
            max_abs_z_error=float(
                self.max_abs_z_error
            ),
            max_abs_x=float(
                self.max_abs_x
            ),
            max_abs_y=float(
                self.max_abs_y
            ),
            max_abs_theta_y=float(
                self.max_abs_theta_y
            ),
            max_abs_ic=float(
                self.max_abs_ic
            ),
            max_abs_id=float(
                self.max_abs_id
            ),
            current_command=(
                self.current_command.copy()
            ),
            current_actual=(
                self.current_actual.copy()
            ),
            desired_force_z=float(
                self.last_control_output
                .desired_force_z
            ),
            final_ic=float(
                allocation.common_current
            ),
            final_id=float(
                allocation.differential_current
            ),
            allocator_saturated=bool(
                allocation.saturated
            ),
            requested_force_z=float(
                allocation.requested_force
            ),
            achieved_force_z=float(
                allocation.achieved_force
            ),
            final_linear_velocity_lcs=(
                final_state
                .linear_velocity_lcs
                .copy()
            ),
            final_angular_velocity_lcs=(
                final_state
                .angular_velocity_lcs
                .copy()
            ),
            controller_updates=int(
                self.controller_step_count
            ),
            physics_steps=int(
                self.physics_step_count
            ),
            sensing_enabled=(
                self.sensing_enabled
            ),
            position_noise_std=float(
                self.position_noise_std
            ),
            theta_y_noise_std=float(
                self.theta_y_noise_std
            ),
            measurement_delay_samples=int(
                self.measurement_delay_samples
            ),
            measurement_seed=int(
                self.measurement_seed
            ),
            measurement_samples=int(
                self.measurement_samples
            ),
            rms_position_residual_lcs=(
                rms_position_residual
            ),
            max_abs_position_residual_lcs=(
                self.max_abs_position_residual
                .copy()
            ),
            rms_theta_y_residual=float(
                rms_theta_residual
            ),
            max_abs_theta_y_residual=float(
                self.max_abs_theta_residual
            ),
            final_measured_position_lcs=(
                self.last_measurement
                .position_lcs
                .copy()
            ),
            final_measured_theta_y=float(
                self.last_measurement
                .theta_y
            ),
            post_settling=(
                post_settling
            ),
        )