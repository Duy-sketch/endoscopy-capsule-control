"""Planar vertical-field allocation without active capsule angle control.

Solve B_transverse=0 and F_z=F_requested from unit-current wrenches.
At the symmetry axis this reduces to positive common-mode current.
Two coils cannot generally remove both transverse field components
outside the XZ plane. This is not an XYZ position controller.
"""
from __future__ import annotations
import numpy as np
from endoscopy_capsule_control.magnetic import magnetic_wrench
from .allocator import CurrentAllocationResult


class VerticalFieldAllocator:
    def __init__(self, current_abs_max: float, gradient_step: float = 1e-5):
        if current_abs_max <= 0 or gradient_step <= 0:
            raise ValueError("Current limit and gradient step must be positive")
        self.current_abs_max = float(current_abs_max)
        self.gradient_step = float(gradient_step)

    def allocate(self, *, p_capsule, capsule_moment_world, em_positions,
                 em_axes, control_axis_world, desired_force, em_gain,
                 common_current_request=0.0):
        z = np.asarray(control_axis_world, dtype=float)
        z = z / np.linalg.norm(z)
        transverse = np.asarray(em_positions[0]) - np.asarray(em_positions[1])
        transverse = transverse - z * (transverse @ z)
        if np.linalg.norm(transverse) < 1e-12:
            raise RuntimeError("Cannot determine the DEMA transverse axis")
        x = transverse / np.linalg.norm(transverse)
        columns = [magnetic_wrench(p_capsule, capsule_moment_world, em_positions,
                                  em_axes, current, em_gain, self.gradient_step)
                   for current in ([1., 0.], [0., 1.])]
        f = np.array([c[0] @ z for c in columns])
        bx = np.array([c[2] @ x for c in columns])
        bz = np.array([c[2] @ z for c in columns])
        direction = np.array([-bx[1], bx[0]])
        scale = np.max(np.abs(direction))
        if scale < 1e-16:
            raise RuntimeError("Vertical field allocation is singular")
        direction /= scale
        if bz @ direction < 0:
            direction *= -1
        authority = float(f @ direction)
        if abs(authority) < 1e-12:
            raise RuntimeError("Vertical-field force authority is approximately zero")
        requested_amplitude = float(desired_force) / authority
        # Scale both channels together to preserve the field direction.
        amplitude = float(np.clip(requested_amplitude, 0, self.current_abs_max))
        currents = direction * amplitude
        ic = float(np.mean(currents))
        differential = float((currents[1] - currents[0]) / 2)
        return CurrentAllocationResult(
            common_current=ic, differential_current=differential,
            current_1=float(currents[0]), current_2=float(currents[1]),
            common_force_basis=float(f.sum()),
            differential_force_basis=float(f[1] - f[0]),
            requested_force=float(desired_force), achieved_force=float(f @ currents),
            saturated=bool(abs(amplitude - requested_amplitude) > 1e-12),
        )
