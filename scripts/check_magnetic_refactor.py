import numpy as np

from endoscopy_capsule_control.magnetic import (
    dipole_field,
    magnetic_wrench,
    moment_world_from_body,
)


def main():
    # --------------------------------------------------------
    # Test 1: signed current
    # --------------------------------------------------------

    p_em = np.array([
        0.0,
        0.0,
        0.0,
    ])

    axis_em = np.array([
        0.0,
        0.0,
        1.0,
    ])

    p_query = np.array([
        0.02,
        0.0,
        0.10,
    ])

    em_gain = 100.0

    B_plus = dipole_field(
        p_query=p_query,
        p_em=p_em,
        axis_em=axis_em,
        current=+10.0,
        em_gain=em_gain,
    )

    B_minus = dipole_field(
        p_query=p_query,
        p_em=p_em,
        axis_em=axis_em,
        current=-10.0,
        em_gain=em_gain,
    )

    print("B(+10 A) =", B_plus)
    print("B(-10 A) =", B_minus)

    print(
        "B(+I) + B(-I) =",
        B_plus + B_minus,
    )

    # --------------------------------------------------------
    # Test 2: capsule moment
    # --------------------------------------------------------

    R_world_body = np.eye(3)

    moment_body = np.array([
        0.05,
        0.0,
        0.0,
    ])

    m_world = moment_world_from_body(
        R_world_body,
        moment_body,
    )

    # --------------------------------------------------------
    # Test 3: complete magnetic wrench
    # --------------------------------------------------------

    em_positions = [
        np.array([
            +0.0475,
            0.0,
            0.0,
        ]),
        np.array([
            -0.0475,
            0.0,
            0.0,
        ]),
    ]

    em_axes = [
        np.array([
            0.0,
            0.0,
            1.0,
        ]),
        np.array([
            0.0,
            0.0,
            1.0,
        ]),
    ]

    currents = np.array([
        -15.0,
        +15.0,
    ])

    force, torque, B = magnetic_wrench(
        p_capsule=p_query,
        capsule_moment_world=m_world,
        em_positions=em_positions,
        em_axes=em_axes,
        currents=currents,
        em_gain=em_gain,
        gradient_step=1e-5,
    )

    print("\nB total =", B)
    print("Force =", force)
    print("Torque =", torque)


if __name__ == "__main__":
    main()