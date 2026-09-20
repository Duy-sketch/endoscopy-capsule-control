"""Central numerical configuration for the DEMA-MCE project.

The configuration deliberately separates:

- physical / magnetic parameters,
- controller parameters,
- uncertainty models used by the plant.

Noise switches are runtime choices and are not hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation and controller update rates."""

    physics_dt: float = 0.001
    control_dt: float = 0.010
    simulation_time: float = 10.0


@dataclass(frozen=True)
class CapsuleConfig:
    """Capsule physical parameters."""

    # Current project choice. The paper states approximately 7.6 g,
    # while the present MuJoCo model intentionally uses 7.0 g.
    mass: float = 0.007

    length: float = 0.030
    diameter: float = 0.0114

    # Test/simulation magnetic moment. The paper does not provide a
    # numerical value for the permanent-magnet moment.
    moment_body: tuple[float, float, float] = (0.0, 0.0, 0.05)


@dataclass(frozen=True)
class FluidConfig:
    """Silicone-oil parameters for the closed-loop tank experiment."""

    density: float = 965.0
    kinematic_viscosity: float = 100e-6
    rotational_damping: float = 3e-5


@dataclass(frozen=True)
class ElectromagnetConfig:
    """DEMA magnetic-model parameters."""

    # Temporary simulation calibration gain; not identified by the paper.
    magnetic_gain: float = 106.092028004

    current_abs_max: float = 20.0
    differential_current_min: float = 0.0
    differential_current_max: float = 20.0

    magnetic_gradient_step: float = 1e-5


@dataclass(frozen=True)
class PowerSupplyConfig:
    """Equivalent dual-channel electromagnet power supply.

    The DEMA paper states that the two electromagnets are driven by two
    independent power units, but does not identify the supplies.

    The baseline simulation therefore uses a Kepco BOP 20-20-equivalent
    current source because it supports +/-20 A.  Its current-mode ripple
    and noise specification is 0.03% of full-scale RMS.  At 20 A this is
    0.006 A RMS.

    This module represents source uncertainty only.  A coil L/R model can
    be added later when coil resistance/inductance are identified.
    """

    model_name: str = "Kepco BOP 20-20 equivalent"
    current_abs_max: float = 20.0
    current_noise_rms_A: float = 0.006

    # No invented electrical time constant by default.  Set > 0 only when
    # a justified supply/coil response model is available.
    response_time_constant_s: float = 0.0


@dataclass(frozen=True)
class LocalizationConfig:
    """RF-localization uncertainty used by the controller.

    The paper reports dynamic 3-D position localization errors of
    1.70 +/- 0.74 mm and 1.52 +/- 0.72 mm (2-norm RMSE / dispersion).
    We use the mean RMSE, 1.61 mm, and convert it to an equivalent
    per-axis Gaussian sigma under an isotropic, independent-axis model:

        sigma_axis = 1.61 mm / sqrt(3) ~= 0.9295 mm.

    This per-axis Gaussian interpretation is a modeling assumption; the
    paper does not provide the per-axis probability distribution.

    The software runs at 100 Hz.  The paper does not identify a fixed
    end-to-end localization delay, so the baseline delay is zero samples.
    """

    position_noise_std_m: float = 0.00161 / (3.0 ** 0.5)

    # No paper-identified angular localization sigma is available.
    theta_y_noise_std_rad: float = 0.0

    delay_samples: int = 0


@dataclass(frozen=True)
class AuboI10Config:
    """AUBO i10 pose-uncertainty model used for the DEMA mount.

    The MuJoCo model matches the older AUBO i10 joint-range generation.
    A corresponding AUBO i10 specification gives +/-0.05 mm pose
    repeatability.  For Monte-Carlo simulation we represent this as a
    fixed Cartesian translation bias sampled once per episode.

    The default Gaussian sigma is chosen such that 3 sigma equals the
    quoted repeatability magnitude.  The vector norm is clipped to that
    magnitude.  This is an explicit simulation assumption, not a claim
    that the manufacturer specifies a Gaussian distribution.
    """

    position_repeatability_m: float = 0.00005
    position_bias_sigma_m: float = 0.00005 / 3.0


@dataclass(frozen=True)
class ZControlConfig:
    """Z-position hover controller parameters."""

    z_ref: float = 0.100

    natural_frequency: float = 5.0
    damping_ratio: float = 1.0
    integral_pole: float = 1.0

    derivative_filter_tau: float = 0.030
    integral_limit: float = 0.020
    pid_force_limit: float = 0.020

    # Nominal force that compensates gravity minus buoyancy around the
    # current operating point.  Negative because the current DEMA local
    # +Z axis points downward in this model.
    feedforward_force: float = -43.353877e-3

    desired_force_min: float = -0.100
    desired_force_max: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    capsule: CapsuleConfig = field(default_factory=CapsuleConfig)
    fluid: FluidConfig = field(default_factory=FluidConfig)
    electromagnet: ElectromagnetConfig = field(default_factory=ElectromagnetConfig)
    power_supply: PowerSupplyConfig = field(default_factory=PowerSupplyConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    aubo_i10: AuboI10Config = field(default_factory=AuboI10Config)
    z_control: ZControlConfig = field(default_factory=ZControlConfig)


DEFAULT_CONFIG = AppConfig()
