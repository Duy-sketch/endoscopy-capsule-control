"""Temporary DEMA rotation test: X/Y/Z, no path planning."""
import argparse
import csv
import time

import mujoco
import numpy as np

from endoscopy_capsule_control.simulation.follower_options import parse_options
from endoscopy_capsule_control.plant.z_hover_plant import ZHoverPlant
from endoscopy_capsule_control.control.common_current_z import CommonCurrentZ
from endoscopy_capsule_control.simulation.aubo_ik import AUBODifferentialIK
from endoscopy_capsule_control.simulation.safety import RobotEnvironmentGuard
from endoscopy_capsule_control.simulation.viewer import configure_follower_camera
from endoscopy_capsule_control.magnetic import magnetic_wrench


def angle(a, b):
    """Unsigned angle between vectors [deg]."""
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-20:
        return float("nan")
    cosine = np.clip(np.dot(a, b) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotation_angle(R, R0):
    """Full orientation change, including rotation about Z [0..180 deg]."""
    cosine = (np.trace(R @ R0.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def magnetic_state(plant):
    state = plant.true_state()
    geometry = plant.actual_geometry()
    force, torque, B = magnetic_wrench(
        state.position_world,
        state.magnetic_moment_world,
        [geometry.em1.position_world, geometry.em2.position_world],
        [geometry.em1.axis_world, geometry.em2.axis_world],
        plant.power_supply.last_sample.actual_A,
        plant.config.electromagnet.magnetic_gain,
    )
    return state, geometry, force, B


def draw_vectors(viewer, plant):
    state, geometry, force, B = magnetic_state(plant)
    p = state.position_world

    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0

        def arrow(start, vector, length, color):
            norm = np.linalg.norm(vector)
            if norm < 1e-12 or scene.ngeom >= scene.maxgeom:
                return

            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array(color, dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.0012,
                start,
                start + length * vector / norm,
            )
            scene.ngeom += 1

        # Arrow lengths are normalized for visibility.
        arrow(p, B, 0.045, [0.0, 1.0, 0.15, 1.0])  # Green: B
        arrow(
            p, state.magnetic_moment_world,
            0.028, [0.1, 0.4, 1.0, 1.0],          # Blue: capsule moment
        )
        arrow(p, force, 0.045, [1.0, 0.5, 0.0, 1.0])  # Orange: force

        origin = geometry.lcs_pose.position
        R = geometry.lcs_pose.rotation
        arrow(origin, R[:, 0], 0.05, [1.0, 0.1, 0.1, 1.0])  # Red: DEMA X
        arrow(origin, R[:, 2], 0.065, [0.8, 0.2, 1.0, 1.0])  # Purple: DEMA Z


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tilt-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--tilt-deg", type=float, default=90.0)
    parser.add_argument("--ramp-s", type=float, default=10.0)
    extra, remaining = parser.parse_known_args()

    if (
        not np.isfinite([extra.tilt_deg, extra.ramp_s]).all()
        or extra.ramp_s <= 0
        or abs(extra.tilt_deg) > 170
    ):
        parser.error("Require finite angle within +/-170 deg and ramp-s > 0")

    opts, cfg, noise = parse_options([
        "--mode", "hold",
        "--x0-mm", "0",
        "--time", "20",
        "--camera", "capsule",
        "--csv", "results/dema_rotation.csv",
        *remaining,
    ])

    if opts.mode != "hold" or opts.scenario or opts.open_loop:
        parser.error("Omit --mode, --scenario and --open-loop")

    plant = ZHoverPlant(
        config=cfg,
        noise=noise,
        seed=opts.seed,
        vertical=True,
        initial_current_A=(0.0, 0.0),
        x_offset_m=opts.x0_mm / 1000,
        y_offset_m=opts.y0_mm / 1000,
        z_offset_m=opts.z0_mm / 1000,
        theta_y_offset_rad=np.deg2rad(opts.theta0_deg),
    )
    controller = CommonCurrentZ(plant)
    ik = AUBODifferentialIK(system=plant.system)
    guard = RobotEnvironmentGuard(plant, opts.table_clearance_mm)

    origin, R0 = ik.site_pose()
    initial_state, _, _, B0 = magnetic_state(plant)
    m0 = initial_state.magnetic_moment_world.copy()
    p0 = initial_state.position_world.copy()

    # Fixed rotation axis: initial DEMA local axis expressed in world.
    axis = R0[:, "xyz".index(extra.tilt_axis)]
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])

    dt = plant.physics_dt
    period = cfg.simulation.control_dt
    every = int(round(period / dt))
    if not np.isclose(every * dt, period):
        raise ValueError("Control dt must be a multiple of physics dt")

    render_every = max(1, round(1.0 / (30.0 * dt)))
    rows = []
    reason = "duration completed"

    print("Axis: INITIAL DEMA local", extra.tilt_axis.upper())
    print("Target rotation [deg]:", extra.tilt_deg)
    print("Noise:", noise.enabled_names() or ("none",))
    print("2 s settle -> smooth rotation -> hold")
    print("Capsule rotation is free. Current Z PID+FF remains active.")
    print("Green=B; blue=moment; orange=force; red=DEMA X; purple=DEMA Z")
    print(" t[s] | command | EE rotation | m change | B change | m-to-B")

    def reference():
        u = np.clip((plant.time_s - 2.0) / extra.ramp_s, 0.0, 1.0)
        degrees = extra.tilt_deg * (3*u*u - 2*u*u*u)
        theta = np.deg2rad(degrees)

        target_R = (
            np.eye(3)
            + np.sin(theta) * K
            + (1.0 - np.cos(theta)) * (K @ K)
        ) @ R0

        # Angular velocity feedforward for the smooth rotation.
        rate = np.deg2rad(extra.tilt_deg) * 6*u*(1-u) / extra.ramp_s
        return degrees, target_R, axis * rate

    def step(k):
        nonlocal reason

        if k % every == 0:
            clearance, pair = guard.check()
            if clearance <= opts.table_clearance_mm / 1000:
                reason = f"STOP: robot/environment clearance: {pair}"
                return False

            _, target_R, omega_ff = reference()
            position, R = ik.site_pose()

            rotation_error = 0.5 * sum(
                np.cross(R[:, j], target_R[:, j])
                for j in range(3)
            )
            velocity = 3.0 * (origin - position)
            velocity *= min(1.0, 0.01 / max(np.linalg.norm(velocity), 1e-12))

            omega = omega_ff + 3.0 * rotation_error
            omega *= min(1.0, 0.30 / max(np.linalg.norm(omega), 1e-12))

            ik.command_twist(
                twist_world=np.r_[velocity, omega],
                dt=period,
            )

            try:
                controller.update()
            except RuntimeError as exc:
                reason = f"STOP: current controller: {exc}"
                return False

        plant.step(controller.currents)

        if (k + 1) % every == 0:
            state, geometry, _, B = magnetic_state(plant)
            _, R_actual = ik.site_pose()
            displacement = (state.position_world - p0) * 1000

            row = [
                plant.time_s,
                reference()[0],
                rotation_angle(R_actual, R0),
                angle(state.magnetic_moment_world, m0),
                angle(B, B0),
                angle(state.magnetic_moment_world, B),
                *displacement,
                state.position_lcs[2] * 1000,
            ]
            rows.append(row)

            if len(rows) % max(1, round(1.0 / period)) == 0:
                print(" ".join(f"{value:10.3f}" for value in row[:6]))

            if not np.isfinite(state.position_world).all():
                reason = "STOP: nonfinite capsule state"
                return False

            p = state.position_lcs
            if np.linalg.norm(p[:2]) > 0.08 or not 0.04 < p[2] < 0.18:
                reason = "STOP: capsule left diagnostic ROI"
                return False

        return True

    def run(viewer=None):
        nonlocal reason
        for k in range(int(np.ceil(opts.time / dt))):
            if viewer is not None and not viewer.is_running():
                reason = "viewer closed"
                break

            started = time.perf_counter()
            keep_running = step(k)

            if viewer is not None:
                if k % render_every == 0 or not keep_running:
                    draw_vectors(viewer, plant)
                    viewer.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - started)))

            if not keep_running:
                break

    def save():
        opts.csv.parent.mkdir(parents=True, exist_ok=True)
        with opts.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_s",
                "command_deg",
                "EE_rotation_deg",
                "moment_change_deg",
                "B_change_deg",
                "m_to_B_deg",
                "world_dx_mm",
                "world_dy_mm",
                "world_dz_mm",
                "moving_LCS_z_mm",
            ])
            writer.writerows(rows)

        print(reason)
        print("CSV:", opts.csv)
        if rows:
            print("Final EE / moment / B / alignment [deg]:", rows[-1][2:6])
            print("Final world displacement XYZ [mm]:", rows[-1][6:9])

    try:
        if opts.headless:
            run()
        else:
            from mujoco import viewer as mj_viewer

            with mj_viewer.launch_passive(
                plant.system.model, plant.system.data
            ) as viewer:
                configure_follower_camera(
                    viewer, plant.system, opts.camera, guard.table_present
                )
                run(viewer)
                draw_vectors(viewer, plant)
                viewer.sync()
                save()

                print("PHYSICS PAUSED: displayed pose is frozen.")
                print("Close viewer to exit.")

                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
    finally:
        save()


if __name__ == "__main__":
    main()