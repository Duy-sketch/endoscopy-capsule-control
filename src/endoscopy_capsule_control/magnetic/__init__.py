"""
Magnetic models used by the DEMA-MCE simulation.
"""

from .capsule_magnet import (
    angle_between_deg,
    magnetic_potential_energy,
    magnetic_moment_tilt_y,
    moment_world_from_body,
    unit,
)

from .dipole import (
    MU0,
    dipole_field,
    total_field,
)

from .wrench import (
    force_fixed_capsule_moment,
    magnetic_wrench,
)


__all__ = [
    "MU0",
    "unit",
    "dipole_field",
    "total_field",
    "moment_world_from_body",
    "magnetic_moment_tilt_y",
    "magnetic_potential_energy",
    "force_fixed_capsule_moment",
    "magnetic_wrench",
    "angle_between_deg",
]
