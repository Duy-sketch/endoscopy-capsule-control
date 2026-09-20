"""
Run the new 3-DOF capsule controller.

Architecture
------------
X, Y -> AUBO moves DEMA
Z    -> differential current Id

Example:
    python scripts\run_xyz.py --dx-mm 5 --dy-mm 5 --dz-mm 0 ^
        --time 12 --settling-time 6 --ideal-sensing ^
        --viewer --camera overview
"""

from __future__ import annotations

import argparse
import numpy as np

from endoscopy_capsule_control.simulation.xyz_simulation import (
    run_xyz_simulation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="3-DOF capsule control: AUBO-XY + magnetic-Z"
    )

    parser.add_argument(
        "--dx-mm",
        type=float,
        default=5.0,
        help="Capsule target offset along world X [mm].",
    )

    parser.add_argument(
        "--dy-mm",
        type=float,
        default=5.0,
        help="Capsule target offset along world Y [mm].",
    )

    parser.add_argument(
        "--dz-mm",
        type=float,
        default=0.0,
        help="Capsule target offset along world Z [mm].",
    )

    parser.add_argument(
        "--time",
        type=float,
        default=12.0,
        help="Simulation duration [s].",
    )

    parser.add_argument(
        "--settling-time",
        type=float,
        default=6.0,
        help="Time from which RMS/mean statistics are collected [s].",
    )

    parser.add_argument(
        "--ideal-sensing",
        action="store_true",
        help="Disable measurement noise and delay.",
    )

    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open MuJoCo viewer.",
    )

    parser.add_argument(
        "--camera",
        choices=["overview", "capsule"],
        default="overview",
        help="Viewer camera mode.",
    )

    parser.add_argument(
        "--xy-lead-gain",
        type=float,
        default=0.30,
        help="Outer-loop XY feedback lead gain.",
    )

    parser.add_argument(
        "--xy-max-lead-mm",
        type=float,
        default=3.0,
        help="Maximum DEMA XY feedback lead [mm].",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    target_offset_world = 1e-3 * np.array(
        [
            args.dx_mm,
            args.dy_mm,
            args.dz_mm,
        ],
        dtype=float,
    )

    print()
    print("======================================", flush=True)
    print("START XYZ CAPSULE SIMULATION", flush=True)
    print("======================================", flush=True)
    print(
        f"Target offset [mm] = "
        f"[{args.dx_mm:.3f}, {args.dy_mm:.3f}, {args.dz_mm:.3f}]",
        flush=True,
    )
    print(
        f"Simulation time    = {args.time:.3f} s",
        flush=True,
    )
    print(
        f"Settling time      = {args.settling_time:.3f} s",
        flush=True,
    )
    print(
        f"Ideal sensing      = {args.ideal_sensing}",
        flush=True,
    )
    print(
        f"Viewer             = {args.viewer}",
        flush=True,
    )

    if args.viewer:
        print(
            f"Camera             = {args.camera}",
            flush=True,
        )

    print("--------------------------------------", flush=True)
    print("Running simulation...", flush=True)

    result = run_xyz_simulation(
        simulation_time=args.time,
        settling_time=args.settling_time,
        target_offset_world=target_offset_world,
        use_ideal_sensing=args.ideal_sensing,
        capsule_xy_lead_gain=args.xy_lead_gain,
        capsule_xy_max_lead=args.xy_max_lead_mm * 1e-3,
        viewer_enabled=args.viewer,
        camera=args.camera,
    )

    print()
    print("======================================", flush=True)
    print("XYZ CAPSULE CONTROL RESULT", flush=True)
    print("======================================", flush=True)

    print(
        "Initial capsule world [m] =",
        result.initial_capsule_position_world,
        flush=True,
    )

    print(
        "Target capsule world [m]  =",
        result.target_capsule_position_world,
        flush=True,
    )

    print(
        "Final capsule world [m]   =",
        result.final_capsule_position_world,
        flush=True,
    )

    print()
    print(
        "Final error XYZ [mm]      =",
        1000.0 * result.final_error_world,
        flush=True,
    )

    print(
        "RMS error XYZ [mm]        =",
        1000.0 * result.rms_error_world,
        flush=True,
    )

    print(
        "Mean error XYZ [mm]       =",
        1000.0 * result.mean_error_world,
        flush=True,
    )

    print(
        "Max |error| XYZ [mm]      =",
        1000.0 * result.max_abs_error_world,
        flush=True,
    )

    print()
    print(
        "Final DEMA LCS world [m]  =",
        result.final_dema_position_world,
        flush=True,
    )

    print(
        "Max |Id| [A]              =",
        result.max_abs_differential_current,
        flush=True,
    )

    print(
        "Max |coil current| [A]    =",
        result.max_abs_coil_current,
        flush=True,
    )

    print(
        "Current saturation        =",
        result.current_saturated_any,
        flush=True,
    )

    print()
    print(
        "Post-settling samples     =",
        result.post_settling_samples,
        flush=True,
    )

    print(
        "Controller updates        =",
        result.controller_updates,
        flush=True,
    )

    print(
        "Physics steps             =",
        result.physics_steps,
        flush=True,
    )

    print("======================================", flush=True)
    print("SIMULATION FINISHED", flush=True)
    print("======================================", flush=True)


if __name__ == "__main__":
    main()
