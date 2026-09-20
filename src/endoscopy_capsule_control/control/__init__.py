"""Controllers used by the Z-hover experiment."""

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
from .z_hover_controller import (
    ZHoverControlInput,
    ZHoverControlOutput,
    ZHoverController,
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
    "ZHoverControlInput",
    "ZHoverControlOutput",
    "ZHoverController",
]
