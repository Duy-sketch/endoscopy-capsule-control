"""
Static local coupling test for AUBO-XY / DEMA-current-Z architecture.

Purpose
-------
Keep the capsule fixed at the validated hover equilibrium and numerically
translate the entire DEMA in world X/Y while holding DEMA orientation,
Z, capsule pose, and differential current fixed.

This identifies the local mapping

    [delta x_D, delta y_D]
        -> [Fx, Fy, Fz, tau_x, tau_y, tau_z]

before designing the closed-loop XY controller.

Run:
    python scripts\test_dema_xy_coupling.py
"""

from __future__ import annotations

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
from endoscopy_capsule_control.simulation.perturbation import (
    rotation_y,
)


def main():
    cfg = DEFAULT_CONFIG

    # --------------------------------------------------------
    # Validated AUBO / DEMA working pose
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Validated capsule hover equilibrium
    #
    # p_C^L = [0, 0, z_ref]
    # R_WB   = R_WL Ry(90 deg)
    # --------------------------------------------------------
    z_ref = float(
        cfg.z_control.z_ref
    )

    p_capsule = (
        p_lcs
        + R_lcs
        @ np.array(
            [
                0.0,
                0.0,
                z_ref,
            ],
            dtype=float,
        )
    )

    R_capsule = (
        R_lcs
        @ rotation_y(
            np.deg2rad(90.0)
        )
    )

    moment_body = np.asarray(
        cfg.capsule.moment_body,
        dtype=float,
    ).reshape(3)

    moment_world = (
        R_capsule
        @ moment_body
    )

    # Nominal differential-current hover point:
    # Ic = 0, Id = 15 A
    currents = np.array(
        [
            -15.0,
            +15.0,
        ],
        dtype=float,
    )

    em_positions_nom = [
        np.asarray(
            dema.em1.position_world,
            dtype=float,
        ).copy(),
        np.asarray(
            dema.em2.position_world,
            dtype=float,
        ).copy(),
    ]

    em_axes = [
        np.asarray(
            dema.em1.axis_world,
            dtype=float,
        ).copy(),
        np.asarray(
            dema.em2.axis_world,
            dtype=float,
        ).copy(),
    ]

    def evaluate(
        dx_dema: float,
        dy_dema: float,
    ):
        """
        Translate the entire DEMA in WORLD X/Y.

        Capsule is intentionally kept fixed in world coordinates.
        """
        delta_world = np.array(
            [
                dx_dema,
                dy_dema,
                0.0,
            ],
            dtype=float,
        )

        em_positions = [
            em_positions_nom[0]
            + delta_world,

            em_positions_nom[1]
            + delta_world,
        ]

        F_world, tau_world, B_world = (
            magnetic_wrench(
                p_capsule=(
                    p_capsule
                ),
                capsule_moment_world=(
                    moment_world
                ),
                em_positions=(
                    em_positions
                ),
                em_axes=(
                    em_axes
                ),
                currents=(
                    currents
                ),
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

        return (
            F_local,
            tau_local,
            B_local,
        )

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------
    F0, tau0, B0 = evaluate(
        0.0,
        0.0,
    )

    print(
        "=============================================="
    )
    print(
        "DEMA XY STATIC COUPLING TEST"
    )
    print(
        "=============================================="
    )

    print(
        "LCS world [m] =",
        p_lcs,
    )

    print(
        "Capsule equilibrium world [m] =",
        p_capsule,
    )

    print(
        "Currents [A] =",
        currents,
    )

    print()

    print(
        "Center F local [mN] =",
        1e3 * F0,
    )

    print(
        "Center tau local [uN*m] =",
        1e6 * tau0,
    )

    print(
        "Center B local [mT] =",
        1e3 * B0,
    )

    # --------------------------------------------------------
    # Direct +/- 1 mm sanity checks
    # --------------------------------------------------------
    probe = 1e-3

    print()
    print(
        "----------------------------------------------"
    )
    print(
        "+/- 1 mm DEMA translation"
    )
    print(
        "----------------------------------------------"
    )

    for name, dx, dy in [
        ("+X", +probe, 0.0),
        ("-X", -probe, 0.0),
        ("+Y", 0.0, +probe),
        ("-Y", 0.0, -probe),
    ]:
        F, tau, _ = evaluate(
            dx,
            dy,
        )

        print(
            f"{name:>2} DEMA:"
        )

        print(
            "  F local [mN]      =",
            1e3 * F,
        )

        print(
            "  tau local [uN*m]  =",
            1e6 * tau,
        )

    # --------------------------------------------------------
    # Local Jacobian wrt DEMA WORLD X/Y translation
    # --------------------------------------------------------
    h = 0.5e-3

    Fx_p, Tx_p, _ = evaluate(
        +h,
        0.0,
    )

    Fx_m, Tx_m, _ = evaluate(
        -h,
        0.0,
    )

    Fy_p, Ty_p, _ = evaluate(
        0.0,
        +h,
    )

    Fy_m, Ty_m, _ = evaluate(
        0.0,
        -h,
    )

    dF_dxd = (
        Fx_p - Fx_m
    ) / (
        2.0 * h
    )

    dTau_dxd = (
        Tx_p - Tx_m
    ) / (
        2.0 * h
    )

    dF_dyd = (
        Fy_p - Fy_m
    ) / (
        2.0 * h
    )

    dTau_dyd = (
        Ty_p - Ty_m
    ) / (
        2.0 * h
    )

    J_force = np.column_stack(
        [
            dF_dxd,
            dF_dyd,
        ]
    )

    J_torque = np.column_stack(
        [
            dTau_dxd,
            dTau_dyd,
        ]
    )

    print()
    print(
        "----------------------------------------------"
    )
    print(
        "LOCAL JACOBIAN wrt DEMA world [x_D, y_D]"
    )
    print(
        "----------------------------------------------"
    )

    print(
        "J_force [N/m]:"
    )

    print(
        J_force
    )

    print()

    print(
        "rows = [Fx_L, Fy_L, Fz_L]"
    )

    print(
        "cols = [x_D^W, y_D^W]"
    )

    print()

    print(
        "J_torque [N*m/m]:"
    )

    print(
        J_torque
    )

    print()

    print(
        "rows = [tau_x_L, tau_y_L, tau_z_L]"
    )

    print(
        "cols = [x_D^W, y_D^W]"
    )

    print()

    print(
        "Key terms:"
    )

    print(
        "  dFx_L/dxD_W =",
        J_force[0, 0],
        "N/m",
    )

    print(
        "  dFy_L/dyD_W =",
        J_force[1, 1],
        "N/m",
    )

    print(
        "  dtau_y_L/dxD_W =",
        J_torque[1, 0],
        "N",
    )

    print(
        "  dtau_z_L/dyD_W =",
        J_torque[2, 1],
        "N",
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()
