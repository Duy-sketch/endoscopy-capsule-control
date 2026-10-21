"""3-D force linearization for DEMA translation + current scaling.

The local control variables are

    [dx_D, dy_D, dz_D, gamma]

where ``dx_D..dz_D`` translate the complete DEMA in the *current* DEMA
local coordinate system (LCS) and ``gamma`` scales the nominal dual-coil
current vector.

Important modelling choice
--------------------------
The physical MuJoCo plant uses the capsule magnetic moment from the current
rigid-body orientation and computes

    F = grad(m dot B)

with that moment held fixed during the spatial derivative. The Stage-2
allocator should therefore linearize the same instantaneous force map around
the current measured pose. Assuming that the capsule instantly realigns with
the field can change the local force Jacobian, including its sign.

The field-aligned quasi-static model

    F = |m| * grad(|B|)

is retained as an optional diagnostic model for later slow-motion / 5-DOF
analysis, but it is not the default for the current translational feedback
allocator.

"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from endoscopy_capsule_control.magnetic import magnetic_wrench
from endoscopy_capsule_control.magnetic.dipole import total_field
from endoscopy_capsule_control.simulation.dema_geometry import DEMAGeometry


@dataclass(frozen=True)
class ActuationJacobianResult:
    jacobian_force_lcs: np.ndarray
    baseline_force_lcs_N: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_number: float


def _translated_geometry_lcs(
    geometry: DEMAGeometry,
    translation_lcs_m: np.ndarray,
) -> DEMAGeometry:
    """Return a copy of ``geometry`` translated in its current LCS.

    Translation is expressed in the *current* DEMA LCS, not in world axes.
    DEMA orientation is unchanged by this helper.
    """

    translation_lcs = np.asarray(translation_lcs_m, dtype=float).reshape(3)
    shift_world = geometry.lcs_pose.rotation @ translation_lcs

    lcs_pose = replace(
        geometry.lcs_pose,
        position=np.asarray(geometry.lcs_pose.position, dtype=float) + shift_world,
    )
    em1 = replace(
        geometry.em1,
        position_world=(
            np.asarray(geometry.em1.position_world, dtype=float) + shift_world
        ),
    )
    em2 = replace(
        geometry.em2,
        position_world=(
            np.asarray(geometry.em2.position_world, dtype=float) + shift_world
        ),
    )
    return replace(geometry, lcs_pose=lcs_pose, em1=em1, em2=em2)


def _field_world(
    *,
    capsule_position_world: np.ndarray,
    geometry: DEMAGeometry,
    currents_A: np.ndarray,
    em_gain: float,
) -> np.ndarray:
    return np.asarray(
        total_field(
            p_query=np.asarray(capsule_position_world, dtype=float).reshape(3),
            em_positions=[
                geometry.em1.position_world,
                geometry.em2.position_world,
            ],
            em_axes=[
                geometry.em1.axis_world,
                geometry.em2.axis_world,
            ],
            currents=np.asarray(currents_A, dtype=float).reshape(2),
            em_gain=float(em_gain),
        ),
        dtype=float,
    ).reshape(3)


def magnetic_force_world_fixed_moment(
    *,
    capsule_position_world: np.ndarray,
    capsule_moment_world: np.ndarray,
    geometry: DEMAGeometry,
    currents_A: np.ndarray,
    em_gain: float,
    gradient_step: float,
) -> np.ndarray:
    """Instantaneous magnetic force with capsule moment held fixed.

    This matches the instantaneous force model used by the physical MuJoCo
    plant and is the default model for Stage-2 feedback allocation.
    """

    force_world, _, _ = magnetic_wrench(
        p_capsule=np.asarray(capsule_position_world, dtype=float).reshape(3),
        capsule_moment_world=np.asarray(capsule_moment_world, dtype=float).reshape(3),
        em_positions=[
            geometry.em1.position_world,
            geometry.em2.position_world,
        ],
        em_axes=[
            geometry.em1.axis_world,
            geometry.em2.axis_world,
        ],
        currents=np.asarray(currents_A, dtype=float).reshape(2),
        em_gain=float(em_gain),
        gradient_step=float(gradient_step),
    )
    return np.asarray(force_world, dtype=float).reshape(3)


def magnetic_force_world_field_aligned(
    *,
    capsule_position_world: np.ndarray,
    capsule_moment_magnitude_A_m2: float,
    geometry: DEMAGeometry,
    currents_A: np.ndarray,
    em_gain: float,
    gradient_step: float,
) -> np.ndarray:
    """Quasi-static force for a freely rotating dipole aligned with ``B``.

    For an aligned permanent magnetic dipole,

        U = -|m| |B|
        F = -grad(U) = |m| grad(|B|)

    The gradient is evaluated in world coordinates by central differences.
    No preferred world X/Y/Z direction is assumed.
    """

    p = np.asarray(capsule_position_world, dtype=float).reshape(3)
    moment_mag = float(capsule_moment_magnitude_A_m2)
    h = float(gradient_step)

    if moment_mag <= 0.0:
        raise ValueError("capsule magnetic-moment magnitude must be positive")
    if h <= 0.0:
        raise ValueError("gradient_step must be positive")

    force = np.zeros(3, dtype=float)
    for axis in range(3):
        dp = np.zeros(3, dtype=float)
        dp[axis] = h

        B_plus = _field_world(
            capsule_position_world=p + dp,
            geometry=geometry,
            currents_A=currents_A,
            em_gain=em_gain,
        )
        B_minus = _field_world(
            capsule_position_world=p - dp,
            geometry=geometry,
            currents_A=currents_A,
            em_gain=em_gain,
        )

        force[axis] = moment_mag * (
            np.linalg.norm(B_plus) - np.linalg.norm(B_minus)
        ) / (2.0 * h)

    return force


def magnetic_force_world(
    *,
    capsule_position_world: np.ndarray,
    capsule_moment_world: np.ndarray,
    geometry: DEMAGeometry,
    currents_A: np.ndarray,
    em_gain: float,
    gradient_step: float,
    align_moment_to_field: bool = False,
) -> np.ndarray:
    """Force model used by the allocator linearization.

    The default keeps the current measured capsule moment fixed so the local
    Jacobian matches the instantaneous MuJoCo plant. Set
    ``align_moment_to_field=True`` only for a quasi-static diagnostic in which
    the capsule is assumed to realign with the field before force is evaluated.
    """

    m_world = np.asarray(capsule_moment_world, dtype=float).reshape(3)
    if align_moment_to_field:
        return magnetic_force_world_field_aligned(
            capsule_position_world=capsule_position_world,
            capsule_moment_magnitude_A_m2=float(np.linalg.norm(m_world)),
            geometry=geometry,
            currents_A=currents_A,
            em_gain=em_gain,
            gradient_step=gradient_step,
        )

    return magnetic_force_world_fixed_moment(
        capsule_position_world=capsule_position_world,
        capsule_moment_world=m_world,
        geometry=geometry,
        currents_A=currents_A,
        em_gain=em_gain,
        gradient_step=gradient_step,
    )


def _jacobian_diagnostics(J: np.ndarray) -> tuple[np.ndarray, int, float]:
    singular_values = np.linalg.svd(J, compute_uv=False)
    rank = int(np.linalg.matrix_rank(J))
    if singular_values.size == 0 or singular_values[-1] <= 1e-15:
        condition = float("inf")
    else:
        condition = float(singular_values[0] / singular_values[-1])
    return singular_values, rank, condition


def compute_robot_current_force_jacobian(
    *,
    capsule_position_world: np.ndarray,
    capsule_moment_world: np.ndarray,
    geometry: DEMAGeometry,
    nominal_currents_A: np.ndarray,
    em_gain: float,
    gradient_step: float,
    translation_probe_m: float = 0.0005,
    gamma_probe: float = 0.01,
    align_moment_to_field: bool = False,
) -> ActuationJacobianResult:
    """Central-difference 3-D force Jacobian at the *current* geometry.

    The returned 3x4 matrix columns are

        dF_L/dx_D, dF_L/dy_D, dF_L/dz_D, dF_L/dgamma.

    The LCS is the current DEMA frame.  The function is therefore valid at
    arbitrary world positions; it does not encode a special horizontal
    direction.
    """

    h = float(translation_probe_m)
    hg = float(gamma_probe)
    if h <= 0.0:
        raise ValueError("translation_probe_m must be positive")
    if hg <= 0.0:
        raise ValueError("gamma_probe must be positive")

    currents0 = np.asarray(nominal_currents_A, dtype=float).reshape(2)
    R_lcs_world = np.asarray(geometry.lcs_pose.rotation, dtype=float).T

    def force_lcs(geom: DEMAGeometry, currents: np.ndarray) -> np.ndarray:
        fw = magnetic_force_world(
            capsule_position_world=capsule_position_world,
            capsule_moment_world=capsule_moment_world,
            geometry=geom,
            currents_A=currents,
            em_gain=em_gain,
            gradient_step=gradient_step,
            align_moment_to_field=align_moment_to_field,
        )
        return R_lcs_world @ fw

    baseline = force_lcs(geometry, currents0)
    J = np.zeros((3, 4), dtype=float)

    for axis in range(3):
        delta = np.zeros(3, dtype=float)
        delta[axis] = h
        fp = force_lcs(_translated_geometry_lcs(geometry, +delta), currents0)
        fm = force_lcs(_translated_geometry_lcs(geometry, -delta), currents0)
        J[:, axis] = (fp - fm) / (2.0 * h)

    fp = force_lcs(geometry, (1.0 + hg) * currents0)
    fm = force_lcs(geometry, (1.0 - hg) * currents0)
    J[:, 3] = (fp - fm) / (2.0 * hg)

    singular_values, rank, condition = _jacobian_diagnostics(J)
    return ActuationJacobianResult(
        jacobian_force_lcs=J,
        baseline_force_lcs_N=baseline,
        singular_values=singular_values,
        rank=rank,
        condition_number=condition,
    )


def compute_additive_gamma_force_linearization(
    *,
    capsule_position_world: np.ndarray,
    capsule_moment_world: np.ndarray,
    geometry: DEMAGeometry,
    base_currents_A: np.ndarray,
    gamma: float,
    em_gain: float,
    gradient_step: float,
    translation_probe_m: float = 0.0005,
    gamma_probe: float = 0.01,
    align_moment_to_field: bool = False,
) -> ActuationJacobianResult:
    """Linearize force for additive ``[dP_D, dgamma]`` changes.

    The Jacobian is recomputed from the current capsule position and current
    DEMA geometry every time this function is called.  This is the form used
    by the combined robot/current allocator.
    """

    h = float(translation_probe_m)
    hg = float(gamma_probe)
    if h <= 0.0:
        raise ValueError("translation_probe_m must be positive")
    if hg <= 0.0:
        raise ValueError("gamma_probe must be positive")

    base = np.asarray(base_currents_A, dtype=float).reshape(2)
    gamma = float(gamma)
    current = gamma * base
    R_lcs_world = np.asarray(geometry.lcs_pose.rotation, dtype=float).T

    def force_lcs(geom: DEMAGeometry, currents: np.ndarray) -> np.ndarray:
        fw = magnetic_force_world(
            capsule_position_world=capsule_position_world,
            capsule_moment_world=capsule_moment_world,
            geometry=geom,
            currents_A=currents,
            em_gain=em_gain,
            gradient_step=gradient_step,
            align_moment_to_field=align_moment_to_field,
        )
        return R_lcs_world @ fw

    baseline = force_lcs(geometry, current)
    J = np.zeros((3, 4), dtype=float)

    for axis in range(3):
        delta = np.zeros(3, dtype=float)
        delta[axis] = h
        fp = force_lcs(_translated_geometry_lcs(geometry, +delta), current)
        fm = force_lcs(_translated_geometry_lcs(geometry, -delta), current)
        J[:, axis] = (fp - fm) / (2.0 * h)

    fp = force_lcs(geometry, (gamma + hg) * base)
    fm = force_lcs(geometry, (gamma - hg) * base)
    J[:, 3] = (fp - fm) / (2.0 * hg)

    singular_values, rank, condition = _jacobian_diagnostics(J)
    return ActuationJacobianResult(
        jacobian_force_lcs=J,
        baseline_force_lcs_N=baseline,
        singular_values=singular_values,
        rank=rank,
        condition_number=condition,
    )
