import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from endoscopy_capsule_control.config import (
    DEFAULT_CONFIG,
)

from endoscopy_capsule_control.dynamics import (
    capsule_volume,
    dynamic_viscosity,
    stokes_drag_coefficient,
)

from endoscopy_capsule_control.magnetic import (
    magnetic_wrench,
)

from endoscopy_capsule_control.simulation import (
    MujocoSystem,
    apply_capsule_external_wrench,
    compute_capsule_external_wrench,
    get_capsule_state,
    get_dema_geometry,
    initialize_hover_operating_point,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the MuJoCo passive viewer.",
    )

    parser.add_argument(
        "--time",
        type=float,
        default=1.0,
        help="Simulation duration [s].",
    )

    args = parser.parse_args()

    cfg = DEFAULT_CONFIG

    # ========================================================
    # MUJOCO
    # ========================================================

    system = MujocoSystem()

    initialize_hover_operating_point(
        system
    )

    # ========================================================
    # FLUID CONSTANTS
    # ========================================================

    volume = capsule_volume(
        length=cfg.capsule.length,
        diameter=cfg.capsule.diameter,
    )

    mu = dynamic_viscosity(
        fluid_density=cfg.fluid.density,
        kinematic_viscosity=(
            cfg.fluid.kinematic_viscosity
        ),
    )

    drag_coefficient = (
        stokes_drag_coefficient(
            dynamic_viscosity_value=mu,
            diameter=cfg.capsule.diameter,
        )
    )

    # ========================================================
    # FIXED OPEN-LOOP CURRENT
    # ========================================================

    currents = np.array(
        [
            -15.0,
            +15.0,
        ],
        dtype=float,
    )

    moment_body = np.array(
        cfg.capsule.moment_body,
        dtype=float,
    )

    # ========================================================
    # INITIAL STATE
    # ========================================================

    geometry = get_dema_geometry(
        system
    )

    initial_state = get_capsule_state(
        system=system,
        lcs_pose=geometry.lcs_pose,
        magnetic_moment_body=moment_body,
    )

    initial_position_lcs = (
        initial_state.position_lcs.copy()
    )

    # ========================================================
    # ONE PHYSICS STEP
    # ========================================================

    def physics_step():
        # Geometry can change if AUBO moves.
        geometry = get_dema_geometry(
            system
        )

        state = get_capsule_state(
            system=system,
            lcs_pose=geometry.lcs_pose,
            magnetic_moment_body=moment_body,
        )

        em_positions = [
            geometry.em1.position_world,
            geometry.em2.position_world,
        ]

        em_axes = [
            geometry.em1.axis_world,
            geometry.em2.axis_world,
        ]

        (
            magnetic_force_world,
            magnetic_torque_world,
            _,
        ) = magnetic_wrench(
            p_capsule=state.position_world,
            capsule_moment_world=(
                state.magnetic_moment_world
            ),
            em_positions=em_positions,
            em_axes=em_axes,
            currents=currents,
            em_gain=(
                cfg.electromagnet.magnetic_gain
            ),
            gradient_step=(
                cfg.electromagnet
                .magnetic_gradient_step
            ),
        )

        wrench = (
            compute_capsule_external_wrench(
                magnetic_force_world=(
                    magnetic_force_world
                ),
                magnetic_torque_world=(
                    magnetic_torque_world
                ),
                linear_velocity_world=(
                    state.linear_velocity_world
                ),
                angular_velocity_world=(
                    state.angular_velocity_world
                ),
                fluid_density=(
                    cfg.fluid.density
                ),
                capsule_volume_value=volume,
                gravity_world=(
                    system.model.opt.gravity
                ),
                translational_drag_coefficient=(
                    drag_coefficient
                ),
                rotational_damping=(
                    cfg.fluid.rotational_damping
                ),
            )
        )

        apply_capsule_external_wrench(
            system=system,
            wrench=wrench,
        )

        system.step()

    # ========================================================
    # SIMULATION
    # ========================================================

    if args.viewer:
        with mujoco.viewer.launch_passive(
            system.model,
            system.data,
        ) as viewer:

            while (
                system.time < args.time
                and viewer.is_running()
            ):
                wall_start = time.time()

                physics_step()

                viewer.sync()

                elapsed = (
                    time.time()
                    - wall_start
                )

                sleep_time = (
                    system.timestep
                    - elapsed
                )

                if sleep_time > 0.0:
                    time.sleep(
                        sleep_time
                    )

    else:
        while system.time < args.time:
            physics_step()

    # ========================================================
    # FINAL STATE
    # ========================================================

    geometry = get_dema_geometry(
        system
    )

    final_state = get_capsule_state(
        system=system,
        lcs_pose=geometry.lcs_pose,
        magnetic_moment_body=moment_body,
    )

    print(
        "======================================"
    )
    print(
        "OPEN-LOOP HOVER RESULT"
    )
    print(
        "======================================"
    )

    print(
        "Simulation time [s] =",
        system.time,
    )

    print(
        "Initial position LCS [m] =",
        initial_position_lcs,
    )

    print(
        "Final position LCS [m] =",
        final_state.position_lcs,
    )

    print(
        "Displacement [mm] =",
        1000.0
        * (
            final_state.position_lcs
            - initial_position_lcs
        ),
    )

    print(
        "Final linear velocity LCS [m/s] =",
        final_state.linear_velocity_lcs,
    )

    print(
        "Final angular velocity LCS [rad/s] =",
        final_state.angular_velocity_lcs,
    )


if __name__ == "__main__":
    main()