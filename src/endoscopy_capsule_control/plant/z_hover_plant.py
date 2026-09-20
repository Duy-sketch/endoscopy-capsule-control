"""Physical Z-hover plant for the DEMA-actuated capsule.

This module owns the physical simulation path only:

    current command
        -> power-supply model
        -> actual coil currents
        -> magnetic wrench
        -> fluid wrench
        -> MuJoCo rigid-body physics
        -> true capsule state
        -> RF-localization measurement

The controller is intentionally absent from this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from endoscopy_capsule_control.config import AppConfig, DEFAULT_CONFIG
from .fluid import (
    capsule_volume,
    dynamic_viscosity,
    stokes_drag_coefficient,
)
from endoscopy_capsule_control.magnetic import (
    magnetic_moment_tilt_y,
    magnetic_wrench,
)
from endoscopy_capsule_control.simulation.capsule_state import (
    CapsuleState,
    get_capsule_state,
)
from endoscopy_capsule_control.simulation.capsule_wrench import (
    CapsuleExternalWrench,
    apply_capsule_external_wrench,
    compute_capsule_external_wrench,
)
from endoscopy_capsule_control.simulation.dema_geometry import (
    DEMAGeometry,
    get_dema_geometry,
)
from endoscopy_capsule_control.simulation.initialization import (
    initialize_hover_operating_point,
)
from endoscopy_capsule_control.simulation.mujoco_system import MujocoSystem
from endoscopy_capsule_control.simulation.perturbation import (
    apply_initial_perturbation,
)

from .localization import LocalizationMeasurement, RFLocalizationModel
from .noise import NoiseSwitches
from .power_supply import DualChannelPowerSupply, PowerSupplySample
from .robot_uncertainty import AuboI10PoseError, AuboI10PoseUncertainty


@dataclass(frozen=True)
class ZHoverPlantSnapshot:
    """One physical-plant snapshot after a MuJoCo step."""

    time_s: float
    state: CapsuleState
    reported_geometry: DEMAGeometry
    actual_geometry: DEMAGeometry
    power_supply: PowerSupplySample
    external_wrench: CapsuleExternalWrench


class ZHoverPlant:
    """DEMA-MCE plant used for Z-hover controller experiments."""

    def __init__(
        self,
        *,
        config: AppConfig = DEFAULT_CONFIG,
        noise: NoiseSwitches | None = None,
        seed: int = 1,
        x_offset_m: float = 0.0,
        z_offset_m: float = 0.0,
        theta_y_offset_rad: float = 0.0,
        initial_current_A=(-15.0, 15.0),
    ) -> None:
        self.config = config
        self.noise = NoiseSwitches.none() if noise is None else noise
        self.seed = int(seed)

        # Independent RNG streams keep ablation runs comparable: turning
        # one noise source off does not shift another source's sequence.
        seed_sequence = np.random.SeedSequence(self.seed)
        supply_seed, localization_seed, robot_seed = seed_sequence.spawn(3)

        self.system = MujocoSystem()
        initialize_hover_operating_point(self.system)

        self.physics_dt = float(self.system.timestep)
        if not np.isclose(
            self.physics_dt,
            self.config.simulation.physics_dt,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                "MuJoCo timestep does not match SimulationConfig.physics_dt"
            )

        self.magnetic_moment_body = np.asarray(
            self.config.capsule.moment_body,
            dtype=float,
        ).reshape(3)
        self.magnetic_moment_magnitude = float(
            np.linalg.norm(self.magnetic_moment_body)
        )
        if self.magnetic_moment_magnitude <= 0.0:
            raise RuntimeError("Capsule magnetic moment must be non-zero")

        apply_initial_perturbation(
            system=self.system,
            magnetic_moment_body=self.magnetic_moment_body,
            x_offset=float(x_offset_m),
            z_offset=float(z_offset_m),
            theta_y_offset=float(theta_y_offset_rad),
        )

        self.robot_uncertainty = AuboI10PoseUncertainty(
            config=self.config.aubo_i10,
            rng=np.random.default_rng(robot_seed),
            enabled=self.noise.robot,
        )
        self.robot_error: AuboI10PoseError = self.robot_uncertainty.reset()

        self.localization = RFLocalizationModel(
            config=self.config.localization,
            rng=np.random.default_rng(localization_seed),
            noise_enabled=self.noise.localization,
        )
        self.localization.reset()

        self.power_supply = DualChannelPowerSupply(
            config=self.config.power_supply,
            dt=self.physics_dt,
            rng=np.random.default_rng(supply_seed),
            noise_enabled=self.noise.current,
        )
        self.power_supply.reset(initial_current_A=initial_current_A)

        self.capsule_volume = capsule_volume(
            length=self.config.capsule.length,
            diameter=self.config.capsule.diameter,
        )
        self.dynamic_viscosity = dynamic_viscosity(
            fluid_density=self.config.fluid.density,
            kinematic_viscosity=self.config.fluid.kinematic_viscosity,
        )
        self.drag_coefficient = stokes_drag_coefficient(
            dynamic_viscosity_value=self.dynamic_viscosity,
            diameter=self.config.capsule.diameter,
        )

        # Build a valid initial wrench record for diagnostics.  No force is
        # applied here; it is only a placeholder until the first step().
        zero = np.zeros(3, dtype=float)
        self.last_external_wrench = compute_capsule_external_wrench(
            magnetic_force_world=zero,
            magnetic_torque_world=zero,
            linear_velocity_world=zero,
            angular_velocity_world=zero,
            fluid_density=self.config.fluid.density,
            capsule_volume_value=self.capsule_volume,
            gravity_world=self.system.model.opt.gravity,
            translational_drag_coefficient=self.drag_coefficient,
            rotational_damping=self.config.fluid.rotational_damping,
        )

    @property
    def time_s(self) -> float:
        return float(self.system.time)

    def reported_geometry(self) -> DEMAGeometry:
        """Geometry known from nominal AUBO/MuJoCo kinematics."""

        return get_dema_geometry(self.system)

    def actual_geometry(self) -> DEMAGeometry:
        """Physical DEMA geometry after AUBO pose uncertainty."""

        return self.robot_uncertainty.apply_to_geometry(
            self.reported_geometry()
        )

    def true_state(self) -> CapsuleState:
        """Ground-truth capsule state; never feed this directly to control."""

        geometry = self.actual_geometry()
        return get_capsule_state(
            system=self.system,
            lcs_pose=geometry.lcs_pose,
            magnetic_moment_body=self.magnetic_moment_body,
        )

    def sample_localization(self) -> LocalizationMeasurement:
        """Return the only capsule state measurement intended for control."""

        geometry = self.actual_geometry()
        state = get_capsule_state(
            system=self.system,
            lcs_pose=geometry.lcs_pose,
            magnetic_moment_body=self.magnetic_moment_body,
        )
        moment_lcs = (
            geometry.lcs_pose.rotation.T
            @ state.magnetic_moment_world
        )
        theta_y = magnetic_moment_tilt_y(moment_lcs)

        return self.localization.sample(
            true_position_lcs_m=state.position_lcs,
            true_theta_y_rad=theta_y,
        )

    def step(self, current_command_A) -> ZHoverPlantSnapshot:
        """Advance the physical plant by one MuJoCo physics step."""

        supply_sample = self.power_supply.step(current_command_A)

        actual_geometry = self.actual_geometry()
        state_before = get_capsule_state(
            system=self.system,
            lcs_pose=actual_geometry.lcs_pose,
            magnetic_moment_body=self.magnetic_moment_body,
        )

        magnetic_force_world, magnetic_torque_world, _ = magnetic_wrench(
            p_capsule=state_before.position_world,
            capsule_moment_world=state_before.magnetic_moment_world,
            em_positions=[
                actual_geometry.em1.position_world,
                actual_geometry.em2.position_world,
            ],
            em_axes=[
                actual_geometry.em1.axis_world,
                actual_geometry.em2.axis_world,
            ],
            currents=supply_sample.actual_A,
            em_gain=self.config.electromagnet.magnetic_gain,
            gradient_step=self.config.electromagnet.magnetic_gradient_step,
        )

        external_wrench = compute_capsule_external_wrench(
            magnetic_force_world=magnetic_force_world,
            magnetic_torque_world=magnetic_torque_world,
            linear_velocity_world=state_before.linear_velocity_world,
            angular_velocity_world=state_before.angular_velocity_world,
            fluid_density=self.config.fluid.density,
            capsule_volume_value=self.capsule_volume,
            gravity_world=self.system.model.opt.gravity,
            translational_drag_coefficient=self.drag_coefficient,
            rotational_damping=self.config.fluid.rotational_damping,
        )

        apply_capsule_external_wrench(
            system=self.system,
            wrench=external_wrench,
        )
        self.system.step()
        self.last_external_wrench = external_wrench

        reported_geometry = self.reported_geometry()
        actual_geometry_after = self.actual_geometry()
        state_after = get_capsule_state(
            system=self.system,
            lcs_pose=actual_geometry_after.lcs_pose,
            magnetic_moment_body=self.magnetic_moment_body,
        )

        return ZHoverPlantSnapshot(
            time_s=self.time_s,
            state=state_after,
            reported_geometry=reported_geometry,
            actual_geometry=actual_geometry_after,
            power_supply=supply_sample,
            external_wrench=external_wrench,
        )
