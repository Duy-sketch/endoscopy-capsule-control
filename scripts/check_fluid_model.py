import numpy as np

from endoscopy_capsule_control.dynamics import (
    buoyancy_force,
    capsule_volume,
    dynamic_viscosity,
    rotational_drag,
    stokes_drag_coefficient,
    translational_drag,
)


def main():
    # --------------------------------------------------------
    # Parameters used in the current capsule simulation
    # --------------------------------------------------------

    length = 0.030
    diameter = 0.0114

    fluid_density = 965.0
    kinematic_viscosity = 100e-6

    gravity = np.array([
        0.0,
        0.0,
        -9.81,
    ])

    rotational_damping = 3.0e-5

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    volume = capsule_volume(
        length=length,
        diameter=diameter,
    )

    print(
        "Capsule volume [m^3] =",
        volume,
    )

    print(
        "Capsule volume [cm^3] =",
        volume * 1e6,
    )

    # --------------------------------------------------------
    # Viscosity
    # --------------------------------------------------------

    mu = dynamic_viscosity(
        fluid_density=fluid_density,
        kinematic_viscosity=kinematic_viscosity,
    )

    print(
        "\nDynamic viscosity [Pa*s] =",
        mu,
    )

    drag_coefficient = (
        stokes_drag_coefficient(
            dynamic_viscosity_value=mu,
            diameter=diameter,
        )
    )

    print(
        "Drag coefficient [N*s/m] =",
        drag_coefficient,
    )

    # --------------------------------------------------------
    # Buoyancy
    # --------------------------------------------------------

    F_buoyancy = buoyancy_force(
        fluid_density=fluid_density,
        volume=volume,
        gravity_vector=gravity,
    )

    print(
        "\nBuoyancy force [N] =",
        F_buoyancy,
    )

    print(
        "Buoyancy magnitude [mN] =",
        np.linalg.norm(F_buoyancy) * 1000.0,
    )

    # --------------------------------------------------------
    # Translational drag
    # --------------------------------------------------------

    velocity = np.array([
        0.1,
        -0.05,
        0.02,
    ])

    F_drag = translational_drag(
        velocity_world=velocity,
        drag_coefficient=drag_coefficient,
    )

    print(
        "\nVelocity [m/s] =",
        velocity,
    )

    print(
        "Translational drag [N] =",
        F_drag,
    )

    # --------------------------------------------------------
    # Rotational drag
    # --------------------------------------------------------

    omega = np.array([
        1.0,
        -2.0,
        0.5,
    ])

    tau_drag = rotational_drag(
        angular_velocity_world=omega,
        rotational_damping=rotational_damping,
    )

    print(
        "\nAngular velocity [rad/s] =",
        omega,
    )

    print(
        "Rotational drag [N*m] =",
        tau_drag,
    )


if __name__ == "__main__":
    main()