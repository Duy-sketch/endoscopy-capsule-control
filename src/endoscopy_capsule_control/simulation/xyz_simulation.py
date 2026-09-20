"""
Closed-loop XYZ capsule-position simulation.

X/Y: AUBO translates the DEMA.
Z:   DEMA differential current.

The validated magnetic/fluid/current/sensing plant is reused.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import mujoco.viewer
import numpy as np

from endoscopy_capsule_control.config import DEFAULT_CONFIG
from endoscopy_capsule_control.control.aubo_xy_controller import (
    AUBOXYController,
    AUBOXYGains,
)
from endoscopy_capsule_control.control.capsule_xy_controller import (
    CapsuleXYController,
    CapsuleXYGains,
)
from endoscopy_capsule_control.control.xyz_controller import (
    XYZControlInput,
    XYZController,
)
from endoscopy_capsule_control.control.z_hover_controller import (
    ZHoverController,
)
from endoscopy_capsule_control.dynamics.current_dynamics import (
    FirstOrderCurrentDynamics,
)
from endoscopy_capsule_control.dynamics.fluid import (
    capsule_volume,
    dynamic_viscosity,
    stokes_drag_coefficient,
)
from endoscopy_capsule_control.magnetic.wrench import magnetic_wrench
from endoscopy_capsule_control.sensing.measurement import MeasurementModel
from endoscopy_capsule_control.simulation.aubo_ik import AUBODifferentialIK
from endoscopy_capsule_control.simulation.capsule_state import (
    get_capsule_state,
)
from endoscopy_capsule_control.simulation.capsule_wrench import (
    apply_capsule_external_wrench,
    compute_capsule_external_wrench,
)
from endoscopy_capsule_control.simulation.dema_geometry import (
    get_dema_geometry,
)
from endoscopy_capsule_control.simulation.initialization import (
    initialize_aubo_work_pose,
)
from endoscopy_capsule_control.simulation.mujoco_system import MujocoSystem
from endoscopy_capsule_control.simulation.perturbation import (
    rotation_y,
    rotation_matrix_to_quaternion_wxyz,
)
from endoscopy_capsule_control.simulation.viewer import (
    configure_capsule_tracking_camera,
    configure_dema_overview_camera,
)


@dataclass(frozen=True)
class XYZSimulationResult:
    initial_capsule_position_world: np.ndarray
    target_capsule_position_world: np.ndarray
    final_capsule_position_world: np.ndarray
    final_error_world: np.ndarray
    rms_error_world: np.ndarray
    mean_error_world: np.ndarray
    max_abs_error_world: np.ndarray
    final_dema_position_world: np.ndarray
    max_abs_differential_current: float
    max_abs_coil_current: float
    current_saturated_any: bool
    post_settling_samples: int
    physics_steps: int
    controller_updates: int


def _theta_y_from_moment_lcs(moment_lcs: np.ndarray) -> float:
    moment = np.asarray(moment_lcs, dtype=float).reshape(3)
    return float(np.arctan2(-moment[2], moment[0]))


def _measured_moment_world(
    *,
    theta_y: float,
    moment_magnitude: float,
    dema_rotation_world: np.ndarray,
) -> np.ndarray:
    moment_lcs = moment_magnitude * np.array(
        [
            np.cos(theta_y),
            0.0,
            -np.sin(theta_y),
        ],
        dtype=float,
    )

    return (
        np.asarray(
            dema_rotation_world,
            dtype=float,
        ).reshape(3, 3)
        @ moment_lcs
    )



def _initialize_capsule_at_hover_equilibrium(
    *,
    system: MujocoSystem,
    dema_position_world: np.ndarray,
    dema_rotation_world: np.ndarray,
    z_ref: float,
) -> None:
    """Place MCE exactly at the validated hover operating point."""

    model = system.model
    data = system.data

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "MCE_free",
    )

    if joint_id < 0:
        raise RuntimeError("Cannot find free joint: MCE_free")

    qpos_adr = int(model.jnt_qposadr[joint_id])
    dof_adr = int(model.jnt_dofadr[joint_id])

    p_lcs = np.asarray(
        dema_position_world,
        dtype=float,
    ).reshape(3)

    R_lcs = np.asarray(
        dema_rotation_world,
        dtype=float,
    ).reshape(3, 3)

    # local capsule equilibrium: [0, 0, z_ref]
    p_capsule_world = (
        p_lcs
        + R_lcs
        @ np.array(
            [0.0, 0.0, float(z_ref)],
            dtype=float,
        )
    )

    # validated horizontal equilibrium orientation:
    # body +Z -> DEMA local +X
    R_capsule_world = (
        R_lcs
        @ rotation_y(
            np.deg2rad(90.0)
        )
    )

    q_capsule_wxyz = (
        rotation_matrix_to_quaternion_wxyz(
            R_capsule_world
        )
    )

    data.qpos[qpos_adr:qpos_adr + 3] = (
        p_capsule_world
    )

    data.qpos[qpos_adr + 3:qpos_adr + 7] = (
        q_capsule_wxyz
    )

    data.qvel[dof_adr:dof_adr + 6] = 0.0

    system.forward()


def run_xyz_simulation(
    *,
    simulation_time: float = 12.0,
    settling_time: float = 6.0,
    target_offset_world: np.ndarray | None = None,
    use_ideal_sensing: bool = False,
    position_noise_std: float | None = None,
    theta_y_noise_std_deg: float | None = None,
    delay_samples: int | None = None,
    random_seed: int | None = None,
    capsule_xy_lead_gain: float = 0.30,
    capsule_xy_max_lead: float = 0.003,
    viewer_enabled: bool = False,
    camera: str = "overview",
) -> XYZSimulationResult:
    cfg = DEFAULT_CONFIG

    if simulation_time <= 0.0:
        raise ValueError("simulation_time must be positive.")

    target_offset = (
        np.zeros(3, dtype=float)
        if target_offset_world is None
        else np.asarray(target_offset_world, dtype=float).reshape(3)
    )

    magnetic_moment_body = np.asarray(
        cfg.capsule.moment_body,
        dtype=float,
    ).reshape(3)

    moment_magnitude = float(np.linalg.norm(magnetic_moment_body))

    volume = capsule_volume(
        length=cfg.capsule.length,
        diameter=cfg.capsule.diameter,
    )

    mu = dynamic_viscosity(
        fluid_density=cfg.fluid.density,
        kinematic_viscosity=cfg.fluid.kinematic_viscosity,
    )

    drag_coefficient = stokes_drag_coefficient(
        dynamic_viscosity_value=mu,
        diameter=cfg.capsule.diameter,
    )

    # ========================================================
    # System / initial operating point
    # ========================================================

    system = MujocoSystem()

    # Put the AUBO/DEMA at the validated working pose first.
    initialize_aubo_work_pose(system)
    system.forward()

    # XML MCE pose is only a visual/default pose, not the hover point.
    # Initialize the capsule ABSOLUTELY at the validated equilibrium.
    dema_for_initialization = get_dema_geometry(
        system
    )

    _initialize_capsule_at_hover_equilibrium(
        system=system,
        dema_position_world=(
            dema_for_initialization
            .lcs_pose
            .position
        ),
        dema_rotation_world=(
            dema_for_initialization
            .lcs_pose
            .rotation
        ),
        z_ref=cfg.z_control.z_ref,
    )

    physics_dt = float(system.timestep)
    control_dt = float(cfg.simulation.control_dt)
    control_every = int(round(control_dt / physics_dt))

    if control_every <= 0:
        raise RuntimeError("Invalid control/physics timestep ratio.")

    dema_initial = get_dema_geometry(system)

    capsule_initial = get_capsule_state(
        system=system,
        lcs_pose=dema_initial.lcs_pose,
        magnetic_moment_body=magnetic_moment_body,
    )

    initial_capsule_position_world = (
        capsule_initial.position_world.copy()
    )

    target_capsule_position_world = (
        initial_capsule_position_world
        + target_offset
    )

    # ========================================================
    # Controllers
    # ========================================================

    z_controller = ZHoverController(
        mass=cfg.capsule.mass,
        drag_coefficient=drag_coefficient,
        control_dt=control_dt,
        z_ref=cfg.z_control.z_ref,
        natural_frequency=cfg.z_control.natural_frequency,
        damping_ratio=cfg.z_control.damping_ratio,
        integral_pole=cfg.z_control.integral_pole,
        derivative_filter_tau=cfg.z_control.derivative_filter_tau,
        integral_limit=cfg.z_control.integral_limit,
        pid_force_limit=cfg.z_control.pid_force_limit,
        feedforward_force_z=cfg.z_control.feedforward_force,
        desired_force_z_min=cfg.z_control.desired_force_min,
        desired_force_z_max=cfg.z_control.desired_force_max,
        current_abs_max=cfg.electromagnet.current_abs_max,
        differential_current_min=(
            cfg.electromagnet.differential_current_min
        ),
        differential_current_max=(
            cfg.electromagnet.differential_current_max
        ),
        gradient_step=cfg.electromagnet.magnetic_gradient_step,
        em_gain=cfg.electromagnet.magnetic_gain,
    )

    xy_controller = CapsuleXYController(
        gains=CapsuleXYGains(
            lead_gain=capsule_xy_lead_gain,
            max_lead=capsule_xy_max_lead,
        )
    )

    aubo_controller = AUBOXYController(
        gains=AUBOXYGains(
            kp_xy=2.0,
            kp_z_hold=2.0,
            kp_rotation=2.0,
            max_linear_speed=0.03,
            max_angular_speed=0.50,
        )
    )

    xyz_controller = XYZController(
        xy_controller=xy_controller,
        aubo_controller=aubo_controller,
        z_controller=z_controller,
    )

    xyz_controller.reset(
        initial_capsule_position_world=capsule_initial.position_world,
        initial_capsule_position_lcs=capsule_initial.position_lcs,
        initial_dema_position_world=dema_initial.lcs_pose.position,
        initial_dema_rotation_world=dema_initial.lcs_pose.rotation,
    )

    aubo_ik = AUBODifferentialIK(
        system=system,
        site_name="LCS_origin",
        damping=1e-3,
        max_joint_speed=0.8,
    )
    aubo_ik.reset_command_to_current()

    # ========================================================
    # Sensing
    # ========================================================

    if use_ideal_sensing:
        sensor_position_noise = 0.0
        sensor_theta_noise = 0.0
        sensor_delay = 0
    else:
        sensor_position_noise = float(
            cfg.sensing.position_noise_std
            if position_noise_std is None
            else position_noise_std
        )

        theta_noise_deg = float(
            cfg.sensing.theta_y_noise_std_deg
            if theta_y_noise_std_deg is None
            else theta_y_noise_std_deg
        )

        sensor_theta_noise = float(np.deg2rad(theta_noise_deg))

        sensor_delay = int(
            cfg.sensing.delay_samples
            if delay_samples is None
            else delay_samples
        )

    sensor_seed = int(
        cfg.sensing.random_seed
        if random_seed is None
        else random_seed
    )

    measurement_model = MeasurementModel(
        position_noise_std=sensor_position_noise,
        theta_y_noise_std=sensor_theta_noise,
        delay_samples=sensor_delay,
        seed=sensor_seed,
    )
    measurement_model.reset()

    # ========================================================
    # Current actuator
    # ========================================================

    current_dynamics = FirstOrderCurrentDynamics(
        time_constant=cfg.electromagnet.current_time_constant,
        dt=physics_dt,
        current_limit=cfg.electromagnet.current_abs_max,
    )

    current_command = np.array([-15.0, +15.0], dtype=float)
    current_actual = current_command.copy()

    # ========================================================
    # Metrics
    # ========================================================

    sum_error = np.zeros(3, dtype=float)
    sum_error_sq = np.zeros(3, dtype=float)
    post_settling_samples = 0

    max_abs_error = np.zeros(3, dtype=float)
    max_abs_id = 0.0
    max_abs_coil_current = 0.0
    current_saturated_any = False

    physics_steps = 0
    controller_updates = 0

    gravity_world = system.model.opt.gravity.copy()

    number_of_steps = int(round(simulation_time / physics_dt))

    def step_once():
        nonlocal current_command
        nonlocal current_actual
        nonlocal physics_steps
        nonlocal controller_updates
        nonlocal sum_error
        nonlocal sum_error_sq
        nonlocal post_settling_samples
        nonlocal max_abs_error
        nonlocal max_abs_id
        nonlocal max_abs_coil_current
        nonlocal current_saturated_any

        # ----------------------------------------------------
        # 100-Hz controller
        # ----------------------------------------------------
        if physics_steps % control_every == 0:
            dema = get_dema_geometry(system)

            capsule = get_capsule_state(
                system=system,
                lcs_pose=dema.lcs_pose,
                magnetic_moment_body=magnetic_moment_body,
            )

            moment_lcs_true = (
                dema.lcs_pose.rotation.T
                @ capsule.magnetic_moment_world
            )

            theta_y_true = _theta_y_from_moment_lcs(
                moment_lcs_true
            )

            measurement = measurement_model.sample(
                true_position_lcs=capsule.position_lcs,
                true_theta_y=theta_y_true,
            )

            measured_capsule_world = (
                dema.lcs_pose.position
                + dema.lcs_pose.rotation
                @ measurement.position_lcs
            )

            measured_moment_world = _measured_moment_world(
                theta_y=measurement.theta_y,
                moment_magnitude=moment_magnitude,
                dema_rotation_world=dema.lcs_pose.rotation,
            )

            output = xyz_controller.update(
                XYZControlInput(
                    capsule_position_world=measured_capsule_world,
                    capsule_position_lcs=measurement.position_lcs,
                    magnetic_moment_world=measured_moment_world,
                    dema_position_world=dema.lcs_pose.position,
                    dema_rotation_world=dema.lcs_pose.rotation,
                    em_positions_world=(
                        dema.em1.position_world,
                        dema.em2.position_world,
                    ),
                    em_axes_world=(
                        dema.em1.axis_world,
                        dema.em2.axis_world,
                    ),
                    target_capsule_position_world=(
                        target_capsule_position_world
                    ),
                )
            )

            current_command = output.current_command.copy()

            aubo_ik.command_twist(
                twist_world=output.dema_twist_world,
                dt=control_dt,
            )

            current_saturated_any = (
                current_saturated_any
                or output.z.allocation.saturated
                or output.z.force_limit_saturated
            )

            controller_updates += 1

        # ----------------------------------------------------
        # Coil current lag
        # ----------------------------------------------------
        current_actual = current_dynamics.step(
            actual_current=current_actual,
            commanded_current=current_command,
        )

        # ----------------------------------------------------
        # True nonlinear plant
        # ----------------------------------------------------
        dema = get_dema_geometry(system)

        capsule = get_capsule_state(
            system=system,
            lcs_pose=dema.lcs_pose,
            magnetic_moment_body=magnetic_moment_body,
        )

        (
            magnetic_force_world,
            magnetic_torque_world,
            _,
        ) = magnetic_wrench(
            p_capsule=capsule.position_world,
            capsule_moment_world=capsule.magnetic_moment_world,
            em_positions=[
                dema.em1.position_world,
                dema.em2.position_world,
            ],
            em_axes=[
                dema.em1.axis_world,
                dema.em2.axis_world,
            ],
            currents=current_actual,
            em_gain=cfg.electromagnet.magnetic_gain,
            gradient_step=cfg.electromagnet.magnetic_gradient_step,
        )

        external_wrench = compute_capsule_external_wrench(
            magnetic_force_world=magnetic_force_world,
            magnetic_torque_world=magnetic_torque_world,
            linear_velocity_world=capsule.linear_velocity_world,
            angular_velocity_world=capsule.angular_velocity_world,
            fluid_density=cfg.fluid.density,
            capsule_volume_value=volume,
            gravity_world=gravity_world,
            translational_drag_coefficient=drag_coefficient,
            rotational_damping=cfg.fluid.rotational_damping,
        )

        apply_capsule_external_wrench(
            system=system,
            wrench=external_wrench,
        )

        error_world = (
            target_capsule_position_world
            - capsule.position_world
        )

        max_abs_error = np.maximum(
            max_abs_error,
            np.abs(error_world),
        )

        id_actual = float(
            (current_actual[1] - current_actual[0]) / 2.0
        )

        max_abs_id = max(
            max_abs_id,
            abs(id_actual),
        )

        max_abs_coil_current = max(
            max_abs_coil_current,
            float(np.max(np.abs(current_actual))),
        )

        if system.time >= settling_time:
            sum_error += error_world
            sum_error_sq += error_world**2
            post_settling_samples += 1

        system.step()
        physics_steps += 1

    # ========================================================
    # Run
    # ========================================================

    if viewer_enabled:
        with mujoco.viewer.launch_passive(
            system.model,
            system.data,
        ) as viewer:

            if camera == "capsule":
                configure_capsule_tracking_camera(
                    viewer,
                    system,
                )
            else:
                configure_dema_overview_camera(
                    viewer,
                    system,
                )

            viewer.sync()

            for _ in range(number_of_steps):
                if not viewer.is_running():
                    break

                t0 = time.perf_counter()
                step_once()
                viewer.sync()

                sleep_time = (
                    physics_dt
                    - (time.perf_counter() - t0)
                )

                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    else:
        for _ in range(number_of_steps):
            step_once()

    # ========================================================
    # Final metrics
    # ========================================================

    dema_final = get_dema_geometry(system)

    capsule_final = get_capsule_state(
        system=system,
        lcs_pose=dema_final.lcs_pose,
        magnetic_moment_body=magnetic_moment_body,
    )

    final_error_world = (
        target_capsule_position_world
        - capsule_final.position_world
    )

    if post_settling_samples > 0:
        mean_error_world = (
            sum_error / post_settling_samples
        )

        rms_error_world = np.sqrt(
            sum_error_sq / post_settling_samples
        )
    else:
        mean_error_world = np.full(3, np.nan)
        rms_error_world = np.full(3, np.nan)

    return XYZSimulationResult(
        initial_capsule_position_world=initial_capsule_position_world,
        target_capsule_position_world=target_capsule_position_world,
        final_capsule_position_world=capsule_final.position_world.copy(),
        final_error_world=final_error_world,
        rms_error_world=rms_error_world,
        mean_error_world=mean_error_world,
        max_abs_error_world=max_abs_error,
        final_dema_position_world=dema_final.lcs_pose.position.copy(),
        max_abs_differential_current=max_abs_id,
        max_abs_coil_current=max_abs_coil_current,
        current_saturated_any=current_saturated_any,
        post_settling_samples=post_settling_samples,
        physics_steps=physics_steps,
        controller_updates=controller_updates,
    )
