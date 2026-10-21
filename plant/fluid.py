"""
Fluid-force utilities for the magnetic capsule.

This module contains only fluid-related physics:
- capsule volume
- viscosity conversion
- buoyancy
- translational viscous drag
- rotational damping

No MuJoCo-specific code is used here.
"""

from __future__ import annotations

import numpy as np


def capsule_volume(
    length: float,
    diameter: float,
) -> float:
    """
    Compute the volume of a capsule-shaped body.

    The geometry consists of:
    - one cylindrical section
    - two hemispherical ends, equivalent to one sphere

    Parameters
    ----------
    length
        Total capsule length [m].

    diameter
        Capsule diameter [m].

    Returns
    -------
    float
        Capsule volume [m^3].
    """
    length = float(length)
    diameter = float(diameter)

    if length <= 0.0:
        raise ValueError(
            "length must be positive."
        )

    if diameter <= 0.0:
        raise ValueError(
            "diameter must be positive."
        )

    if length < diameter:
        raise ValueError(
            "For a capsule geometry, total length "
            "must be greater than or equal to diameter."
        )

    radius = (
        diameter
        / 2.0
    )

    cylinder_length = (
        length
        - diameter
    )

    cylinder_volume = (
        np.pi
        * radius**2
        * cylinder_length
    )

    sphere_volume = (
        4.0
        / 3.0
        * np.pi
        * radius**3
    )

    return float(
        cylinder_volume
        + sphere_volume
    )


def dynamic_viscosity(
    fluid_density: float,
    kinematic_viscosity: float,
) -> float:
    """
    Convert kinematic viscosity to dynamic viscosity.

        mu = rho * nu

    Parameters
    ----------
    fluid_density
        Fluid density [kg/m^3].

    kinematic_viscosity
        Kinematic viscosity [m^2/s].

    Returns
    -------
    float
        Dynamic viscosity [Pa*s].
    """
    fluid_density = float(
        fluid_density
    )

    kinematic_viscosity = float(
        kinematic_viscosity
    )

    if fluid_density <= 0.0:
        raise ValueError(
            "fluid_density must be positive."
        )

    if kinematic_viscosity < 0.0:
        raise ValueError(
            "kinematic_viscosity cannot be negative."
        )

    return (
        fluid_density
        * kinematic_viscosity
    )


def stokes_drag_coefficient(
    dynamic_viscosity_value: float,
    diameter: float,
) -> float:
    """
    Compute the translational viscous-drag coefficient.

        c = 3 * pi * mu * D

    This follows the low-speed viscous-drag model used in the
    simulation.

    Parameters
    ----------
    dynamic_viscosity_value
        Dynamic viscosity [Pa*s].

    diameter
        Capsule diameter [m].

    Returns
    -------
    float
        Translational drag coefficient [N*s/m].
    """
    mu = float(
        dynamic_viscosity_value
    )

    diameter = float(
        diameter
    )

    if mu < 0.0:
        raise ValueError(
            "dynamic viscosity cannot be negative."
        )

    if diameter <= 0.0:
        raise ValueError(
            "diameter must be positive."
        )

    return float(
        3.0
        * np.pi
        * mu
        * diameter
    )


def buoyancy_force(
    fluid_density: float,
    volume: float,
    gravity_vector: np.ndarray,
) -> np.ndarray:
    """
    Compute buoyancy force in world coordinates.

    If gravity acceleration is

        g_vec = [0, 0, -9.81]

    then buoyancy is

        F_b = -rho * V * g_vec

    so it points opposite to gravity.

    Parameters
    ----------
    fluid_density
        Fluid density [kg/m^3].

    volume
        Displaced fluid volume [m^3].

    gravity_vector
        Gravity acceleration vector [m/s^2].

    Returns
    -------
    np.ndarray, shape (3,)
        Buoyancy force [N].
    """
    fluid_density = float(
        fluid_density
    )

    volume = float(
        volume
    )

    gravity_vector = np.asarray(
        gravity_vector,
        dtype=float,
    ).reshape(3)

    if fluid_density <= 0.0:
        raise ValueError(
            "fluid_density must be positive."
        )

    if volume < 0.0:
        raise ValueError(
            "volume cannot be negative."
        )

    return (
        -fluid_density
        * volume
        * gravity_vector
    )


def translational_drag(
    velocity_world: np.ndarray,
    drag_coefficient: float,
) -> np.ndarray:
    """
    Compute linear viscous drag.

        F_d = -c * v

    Parameters
    ----------
    velocity_world
        Capsule translational velocity [m/s].

    drag_coefficient
        Linear drag coefficient [N*s/m].

    Returns
    -------
    np.ndarray, shape (3,)
        Drag force [N].
    """
    velocity_world = np.asarray(
        velocity_world,
        dtype=float,
    ).reshape(3)

    drag_coefficient = float(
        drag_coefficient
    )

    if drag_coefficient < 0.0:
        raise ValueError(
            "drag_coefficient cannot be negative."
        )

    return (
        -drag_coefficient
        * velocity_world
    )


def rotational_drag(
    angular_velocity_world: np.ndarray,
    rotational_damping: float,
) -> np.ndarray:
    """
    Compute rotational viscous damping.

        tau_d = -C_rot * omega

    Parameters
    ----------
    angular_velocity_world
        Capsule angular velocity [rad/s].

    rotational_damping
        Rotational damping coefficient [N*m*s].

    Returns
    -------
    np.ndarray, shape (3,)
        Damping torque [N*m].
    """
    angular_velocity_world = np.asarray(
        angular_velocity_world,
        dtype=float,
    ).reshape(3)

    rotational_damping = float(
        rotational_damping
    )

    if rotational_damping < 0.0:
        raise ValueError(
            "rotational_damping cannot be negative."
        )

    return (
        -rotational_damping
        * angular_velocity_world
    )