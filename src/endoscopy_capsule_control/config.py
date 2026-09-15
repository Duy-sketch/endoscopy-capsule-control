"""
Central configuration for the DEMA-MCE simulation.

This module contains numerical parameters only.
Physics, control laws, sensing models, and MuJoCo logic
are implemented in their respective modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimulationConfig:
    """
    Simulation and controller update rates.
    """

    physics_dt: float = 0.001
    control_dt: float = 0.010
    simulation_time: float = 10.0


@dataclass(frozen=True)
class CapsuleConfig:
    """
    Capsule physical parameters.
    """

    mass: float = 0.007

    length: float = 0.030
    diameter: float = 0.0114

    # Permanent magnetic moment expressed
    # in the capsule body frame [A*m^2].
    moment_body: tuple[float, float, float] = (
        0.0,
        0.0,
        0.05,
    )


@dataclass(frozen=True)
class FluidConfig:
    """
    Silicone-oil parameters.
    """

    density: float = 965.0

    kinematic_viscosity: float = (
        100e-6
    )

    rotational_damping: float = (
        3e-5
    )


@dataclass(frozen=True)
class ElectromagnetConfig:
    """
    DEMA electromagnetic and current parameters.
    """

    # Temporary simulation calibration gain.
    #
    # This is NOT a hardware-calibrated constant.
    magnetic_gain: float = (
        106.092028004
    )

    current_abs_max: float = 20.0

    differential_current_min: float = 0.0
    differential_current_max: float = 20.0

    common_current_abs_max: float = 3.0

    # First-order coil-current response.
    current_time_constant: float = 0.020

    magnetic_gradient_step: float = (
        1e-5
    )


@dataclass(frozen=True)
class ZControlConfig:
    """
    Z-position hover controller parameters.
    """

    z_ref: float = 0.100

    # Desired closed-loop design parameters.
    natural_frequency: float = 5.0
    damping_ratio: float = 1.0
    integral_pole: float = 1.0

    derivative_filter_tau: float = (
        0.030
    )

    integral_limit: float = 0.020

    # PID force correction limit [N].
    pid_force_limit: float = 0.020

    # Nominal magnetic force needed to compensate
    # gravity minus buoyancy [N].
    #
    # Negative because the current DEMA local +Z
    # axis points downward.
    feedforward_force: float = (
        -43.353877e-3
    )

    desired_force_min: float = -0.100
    desired_force_max: float = 0.0


@dataclass(frozen=True)
class SensingConfig:
    """
    Capsule localization measurement parameters.
    """

    position_noise_std: float = (
        0.00036
    )

    theta_y_noise_std_deg: float = (
        0.20
    )

    delay_samples: int = 1

    random_seed: int = 1


@dataclass(frozen=True)
class AppConfig:
    """
    Complete simulation configuration.
    """

    simulation: SimulationConfig = field(
        default_factory=SimulationConfig
    )

    capsule: CapsuleConfig = field(
        default_factory=CapsuleConfig
    )

    fluid: FluidConfig = field(
        default_factory=FluidConfig
    )

    electromagnet: ElectromagnetConfig = field(
        default_factory=ElectromagnetConfig
    )

    z_control: ZControlConfig = field(
        default_factory=ZControlConfig
    )

    sensing: SensingConfig = field(
        default_factory=SensingConfig
    )


DEFAULT_CONFIG = AppConfig()