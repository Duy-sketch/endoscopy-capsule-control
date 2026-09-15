"""
Controllers used by the magnetic capsule simulation.
"""

from .pid_z import (
    ZPIDCandidate,
    ZPIDController,
    ZPIDGains,
    design_z_pid_gains,
)
from .allocator import (
    CurrentAllocationResult,
    MagneticCurrentAllocator,
    common_differential_from_currents,
    currents_from_common_differential,
    force_basis_along_axis,
)

__all__ = [
    "ZPIDGains",
    "ZPIDCandidate",
    "ZPIDController",
    "design_z_pid_gains",
    "CurrentAllocationResult",
    "MagneticCurrentAllocator",
    "currents_from_common_differential",
    "common_differential_from_currents",
    "force_basis_along_axis",
]
