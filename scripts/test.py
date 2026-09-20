"""
Local 5-DOF magnetic wrench Jacobian test.

Goal
----
Test whether the local magnetic actuation variables

    u = [x_D, y_D, alpha_D, beta_D, I_d]

can locally influence five independent capsule wrench components:

    y = [F_x, F_y, F_z, tau_t1, tau_t2]

where tau_t1 and tau_t2 are the two torque components perpendicular
to the capsule magnetic moment.  Torque about the magnetic moment is
not included because m x B cannot generate torque along m.

This makes the test orientation-independent: the capsule orientation
is represented only by the current magnetic-moment direction.

Default operating point
-----------------------
- AUBO at the validated working pose
- capsule position: [0, 0, z_ref] in the DEMA LCS
- capsule magnetic moment: +x_L
- Ic = 0
- Id = 15 A  ->  I1 = -15 A, I2 = +15 A

Examples
--------
Validated horizontal operating point:
    python scripts\\test_full_5dof_wrench_jacobian.py

Different capsule magnetic orientation:
    python scripts\\test_full_5dof_wrench_jacobian.py --moment-az-deg 20 --moment-el-deg 15

Offset capsule position:
    python scripts\\test_full_5dof_wrench_jacobian.py --x-mm 2 --y-mm -1 --z-mm 3
"""

from __future__ import annotations

import argparse
import numpy as np

from endoscopy_capsule_control.config import DEFAULT_CONFIG
from endoscopy_capsule_control.magnetic.wrench import magnetic_wrench
from endoscopy_capsule_control.simulation.dema_geometry import (
    get_dema_geometry,
)
from endoscopy_capsule_control.simulation.initialization import (
    initialize_aubo_work_pose,
)
from endoscopy_capsule_control.simulation.mujoco_system import (
    MujocoSystem,
)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))

    if n < 1e-12:
        raise RuntimeError("Cannot normalize a near-zero vector.")

    return v / n


def rot_x(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def rot_y(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the local 6x5 and reduced 5x5 magnetic "
            "wrench Jacobians."
        )
    )

    parser.add_argument(
        "--x-mm",
        type=float,
        default=0.0,
        help="Capsule local-x offset from the nominal hover point [mm].",
    )

    parser.add_argument(
        "--y-mm",
        type=float,
        default=0.0,
        help="Capsule local-y offset from the nominal hover point [mm].",
    )

    parser.add_argument(
        "--z-mm",
        type=float,
        default=0.0,
        help="Capsule local-z offset from z_ref [mm].",
    )

    parser.add_argument(
        "--moment-az-deg",
        type=float,
        default=0.0,
        help=(
            "Magnetic-moment azimuth in the DEMA LCS [deg]. "
            "0 deg means +x_L."
        ),
    )

    parser.add_argument(
        "--moment-el-deg",
        type=float,
        default=0.0,
        help=(
            "Magnetic-moment elevation in the DEMA LCS [deg]. "
            "0 deg lies in the x_L-y_L plane."
        ),
    )

    parser.add_argument(
        "--id-a",
        type=float,
        default=15.0,
        help="Nominal differential current Id [A].",
    )

    parser.add_argument(
        "--h-pos-mm",
        type=float,
        default=0.5,
        help="Finite-difference step for DEMA x/y translation [mm].",
    )

    parser.add_argument(
        "--h-angle-deg",
        type=float,
        default=0.25,
        help="Finite-difference step for DEMA alpha/beta rotation [deg].",
    )

    parser.add_argument(
        "--h-current-a",
        type=float,
        default=0.1,
        help="Finite-difference step for Id [A].",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG

    # ========================================================
    # Current DEMA working pose
    # ========================================================
    system = MujocoSystem()

    initialize_aubo_work_pose(
        system
    )

    system.forward()

    dema = get_dema_geometry(
        system
    )

    p_lcs = np.asarray(
        dema.lcs_pose.position,
        dtype=float,
    ).reshape(3)

    R_lcs = np.asarray(
        dema.lcs_pose.rotation,
        dtype=float,
    ).reshape(3, 3)

    # Save each EM geometry in DEMA-local coordinates.
    p_em1_world = np.asarray(
        dema.em1.position_world,
        dtype=float,
    ).reshape(3)

    p_em2_world = np.asarray(
        dema.em2.position_world,
        dtype=float,
    ).reshape(3)

    a_em1_world = unit(
        dema.em1.axis_world
    )

    a_em2_world = unit(
        dema.em2.axis_world
    )

    r_em1_local = (
        R_lcs.T
        @ (
            p_em1_world - p_lcs
        )
    )

    r_em2_local = (
        R_lcs.T
        @ (
            p_em2_world - p_lcs
        )
    )

    a_em1_local = (
        R_lcs.T
        @ a_em1_world
    )

    a_em2_local = (
        R_lcs.T
        @ a_em2_world
    )

    # ========================================================
    # Capsule state
    #
    # Position is defined in DEMA-local coordinates.
    # Orientation is represented by the magnetic-moment direction.
    #
    # This avoids assuming that the capsule must always start with
    # one specific body orientation.
    # ========================================================
    p_capsule_local = np.array(
        [
            args.x_mm * 1e-3,
            args.y_mm * 1e-3,
            float(cfg.z_control.z_ref)
            + args.z_mm * 1e-3,
        ],
        dtype=float,
    )

    p_capsule_world = (
        p_lcs
        + R_lcs
        @ p_capsule_local
    )

    az = np.deg2rad(
        args.moment_az_deg
    )

    el = np.deg2rad(
        args.moment_el_deg
    )

    m_hat_local = unit(
        np.array(
            [
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az),
                np.sin(el),
            ],
            dtype=float,
        )
    )

    moment_mag = float(
        np.linalg.norm(
            np.asarray(
                cfg.capsule.moment_body,
                dtype=float,
            )
        )
    )

    moment_local = (
        moment_mag
        * m_hat_local
    )

    moment_world = (
        R_lcs
        @ moment_local
    )

    # Two orthonormal torque directions perpendicular to m.
    # Their exact labels are arbitrary; together they span the
    # physically controllable magnetic-torque plane.
    if abs(
        float(
            np.dot(
                m_hat_local,
                np.array(
                    [0.0, 0.0, 1.0]
                ),
            )
        )
    ) < 0.90:
        reference = np.array(
            [0.0, 0.0, 1.0]
        )
    else:
        reference = np.array(
            [0.0, 1.0, 0.0]
        )

    t1_local = unit(
        np.cross(
            m_hat_local,
            reference,
        )
    )

    t2_local = unit(
        np.cross(
            m_hat_local,
            t1_local,
        )
    )

    # ========================================================
    # DEMA perturbation model
    #
    # Inputs:
    #   x_D, y_D   : translations along current DEMA local x/y
    #   alpha_D    : rotation about current DEMA local x
    #   beta_D     : rotation about current DEMA local y
    #   Id         : differential current, Ic = 0
    #
    # Translation and rotation are applied to the full DEMA rigidly.
    # Capsule pose is held fixed during each local derivative.
    # ========================================================
    def evaluate(u: np.ndarray):
        u = np.asarray(
            u,
            dtype=float,
        ).reshape(5)

        x_d, y_d, alpha_d, beta_d, id_a = u

        delta_p_local = np.array(
            [
                x_d,
                y_d,
                0.0,
            ],
            dtype=float,
        )

        R_delta_local = (
            rot_x(alpha_d)
            @ rot_y(beta_d)
        )

        R_dema_perturbed = (
            R_lcs
            @ R_delta_local
        )

        p_lcs_perturbed = (
            p_lcs
            + R_lcs
            @ delta_p_local
        )

        p_em1 = (
            p_lcs_perturbed
            + R_dema_perturbed
            @ r_em1_local
        )

        p_em2 = (
            p_lcs_perturbed
            + R_dema_perturbed
            @ r_em2_local
        )

        a_em1 = unit(
            R_dema_perturbed
            @ a_em1_local
        )

        a_em2 = unit(
            R_dema_perturbed
            @ a_em2_local
        )

        currents = np.array(
            [
                -id_a,
                +id_a,
            ],
            dtype=float,
        )

        F_world, tau_world, B_world = (
            magnetic_wrench(
                p_capsule=(
                    p_capsule_world
                ),
                capsule_moment_world=(
                    moment_world
                ),
                em_positions=[
                    p_em1,
                    p_em2,
                ],
                em_axes=[
                    a_em1,
                    a_em2,
                ],
                currents=currents,
                em_gain=(
                    cfg.electromagnet
                    .magnetic_gain
                ),
                gradient_step=(
                    cfg.electromagnet
                    .magnetic_gradient_step
                ),
            )
        )

        F_local = (
            R_lcs.T
            @ F_world
        )

        tau_local = (
            R_lcs.T
            @ tau_world
        )

        B_local = (
            R_lcs.T
            @ B_world
        )

        wrench6 = np.concatenate(
            [
                F_local,
                tau_local,
            ]
        )

        wrench5 = np.array(
            [
                F_local[0],
                F_local[1],
                F_local[2],
                float(
                    np.dot(
                        tau_local,
                        t1_local,
                    )
                ),
                float(
                    np.dot(
                        tau_local,
                        t2_local,
                    )
                ),
            ],
            dtype=float,
        )

        return (
            wrench6,
            wrench5,
            B_local,
        )

    # ========================================================
    # Nominal point
    # ========================================================
    u0 = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            args.id_a,
        ],
        dtype=float,
    )

    w6_0, w5_0, B0_local = evaluate(
        u0
    )

    # ========================================================
    # Numerical Jacobians
    # ========================================================
    h = np.array(
        [
            args.h_pos_mm * 1e-3,
            args.h_pos_mm * 1e-3,
            np.deg2rad(
                args.h_angle_deg
            ),
            np.deg2rad(
                args.h_angle_deg
            ),
            args.h_current_a,
        ],
        dtype=float,
    )

    J6 = np.zeros(
        (6, 5),
        dtype=float,
    )

    J5 = np.zeros(
        (5, 5),
        dtype=float,
    )

    for j in range(5):
        du = np.zeros(
            5,
            dtype=float,
        )

        du[j] = h[j]

        w6_p, w5_p, _ = evaluate(
            u0 + du
        )

        w6_m, w5_m, _ = evaluate(
            u0 - du
        )

        J6[:, j] = (
            w6_p - w6_m
        ) / (
            2.0 * h[j]
        )

        J5[:, j] = (
            w5_p - w5_m
        ) / (
            2.0 * h[j]
        )

    # ========================================================
    # Human-scale finite perturbation map
    #
    # One column corresponds to:
    #   1 mm, 1 mm, 1 deg, 1 deg, 1 A
    # ========================================================
    input_display_scale = np.diag(
        [
            1e-3,
            1e-3,
            np.deg2rad(1.0),
            np.deg2rad(1.0),
            1.0,
        ]
    )

    delta6_display = (
        J6
        @ input_display_scale
    )

    # Force -> mN, torque -> uN*m.
    output6_display_scale = np.diag(
        [
            1e3,
            1e3,
            1e3,
            1e6,
            1e6,
            1e6,
        ]
    )

    delta6_display = (
        output6_display_scale
        @ delta6_display
    )

    # ========================================================
    # Numerically scaled 5x5 map for rank / conditioning.
    #
    # Scaling is ONLY for numerical interpretation:
    # inputs:  [1 mm, 1 mm, 1 deg, 1 deg, 1 A]
    # outputs: [1 mN, 1 mN, 1 mN, 10 uN*m, 10 uN*m]
    # ========================================================
    Su = np.diag(
        [
            1e-3,
            1e-3,
            np.deg2rad(1.0),
            np.deg2rad(1.0),
            1.0,
        ]
    )

    Sy_inv = np.diag(
        [
            1.0 / 1e-3,
            1.0 / 1e-3,
            1.0 / 1e-3,
            1.0 / 1e-5,
            1.0 / 1e-5,
        ]
    )

    J5_scaled = (
        Sy_inv
        @ J5
        @ Su
    )

    U, singular_values, Vt = (
        np.linalg.svd(
            J5_scaled,
            full_matrices=True,
        )
    )

    if singular_values[0] > 0.0:
        relative_singular_values = (
            singular_values
            / singular_values[0]
        )
    else:
        relative_singular_values = (
            singular_values.copy()
        )

    rank_tol = (
        singular_values[0]
        * 1e-6
        if singular_values[0] > 0.0
        else 1e-12
    )

    numerical_rank = int(
        np.sum(
            singular_values
            > rank_tol
        )
    )

    condition_number = (
        float(
            singular_values[0]
            / singular_values[-1]
        )
        if singular_values[-1] > 1e-12
        else np.inf
    )

    weakest_input_direction = (
        Vt[-1, :]
    )

    weakest_output_direction = (
        U[:, -1]
    )

    # ========================================================
    # Print
    # ========================================================
    np.set_printoptions(
        precision=8,
        suppress=True,
        linewidth=160,
    )

    print(
        "============================================================"
    )
    print(
        "FULL 5-DOF LOCAL MAGNETIC WRENCH JACOBIAN"
    )
    print(
        "============================================================"
    )

    print(
        "DEMA LCS world [m]              =",
        p_lcs,
    )

    print(
        "Capsule local position [m]      =",
        p_capsule_local,
    )

    print(
        "Capsule world position [m]      =",
        p_capsule_world,
    )

    print(
        "Moment direction local          =",
        m_hat_local,
    )

    print(
        "Moment az/el [deg]              =",
        [
            args.moment_az_deg,
            args.moment_el_deg,
        ],
    )

    print(
        "Torque tangent t1 local         =",
        t1_local,
    )

    print(
        "Torque tangent t2 local         =",
        t2_local,
    )

    print(
        "Nominal Id [A]                  =",
        args.id_a,
    )

    print()

    print(
        "Nominal F local [mN]            =",
        1e3 * w6_0[:3],
    )

    print(
        "Nominal tau local [uN*m]        =",
        1e6 * w6_0[3:],
    )

    print(
        "Nominal B local [mT]            =",
        1e3 * B0_local,
    )

    print(
        "Reduced wrench5 [mN,mN,mN,uN*m,uN*m] =",
        np.array(
            [
                1e3 * w5_0[0],
                1e3 * w5_0[1],
                1e3 * w5_0[2],
                1e6 * w5_0[3],
                1e6 * w5_0[4],
            ]
        ),
    )

    print()
    print(
        "------------------------------------------------------------"
    )
    print(
        "EFFECT OF +[1 mm, 1 mm, 1 deg, 1 deg, 1 A]"
    )
    print(
        "Rows: Fx,Fy,Fz [mN], tau_x,tau_y,tau_z [uN*m]"
    )
    print(
        "Cols: xD_L, yD_L, alphaD_L, betaD_L, Id"
    )
    print(
        "------------------------------------------------------------"
    )
    print(
        delta6_display
    )

    print()
    print(
        "------------------------------------------------------------"
    )
    print(
        "RAW 5x5 JACOBIAN"
    )
    print(
        "Rows: [Fx_L, Fy_L, Fz_L, tau_t1, tau_t2]"
    )
    print(
        "Cols: [xD_L, yD_L, alphaD_L, betaD_L, Id]"
    )
    print(
        "Units by column:"
    )
    print(
        "  xD,yD : [N/m, N/m, N/m, N*m/m, N*m/m]"
    )
    print(
        "  alpha,beta : [N/rad, N/rad, N/rad, N*m/rad, N*m/rad]"
    )
    print(
        "  Id : [N/A, N/A, N/A, N*m/A, N*m/A]"
    )
    print(
        "------------------------------------------------------------"
    )
    print(
        J5
    )

    print()
    print(
        "------------------------------------------------------------"
    )
    print(
        "SCALED 5x5 JACOBIAN"
    )
    print(
        "Input scales : [1 mm, 1 mm, 1 deg, 1 deg, 1 A]"
    )
    print(
        "Output scales: [1 mN, 1 mN, 1 mN, 10 uN*m, 10 uN*m]"
    )
    print(
        "------------------------------------------------------------"
    )
    print(
        J5_scaled
    )

    print()
    print(
        "Singular values                =",
        singular_values,
    )

    print(
        "Relative singular values       =",
        relative_singular_values,
    )

    print(
        "Numerical rank (tol=1e-6*smax)=",
        numerical_rank,
        "/ 5",
    )

    print(
        "Scaled condition number        =",
        condition_number,
    )

    print()

    print(
        "Weakest input direction"
    )
    print(
        "[xD, yD, alphaD, betaD, Id]    =",
        weakest_input_direction,
    )

    print(
        "Weakest output direction"
    )
    print(
        "[Fx,Fy,Fz,tau_t1,tau_t2]       =",
        weakest_output_direction,
    )

    print()

    if numerical_rank == 5:
        print(
            "RESULT: local 5x5 magnetic wrench map is FULL RANK."
        )
        print(
            "This operating point is locally invertible under the "
            "chosen five actuation coordinates."
        )
    else:
        print(
            "RESULT: local 5x5 magnetic wrench map is RANK DEFICIENT."
        )
        print(
            "At least one desired 5-DOF wrench direction cannot be "
            "independently generated with these five inputs at this pose."
        )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()
