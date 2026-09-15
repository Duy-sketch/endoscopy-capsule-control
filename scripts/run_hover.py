"""
Command-line entry point for the DEMA-MCE hover simulation.
"""

from __future__ import annotations

import argparse

import numpy as np

from endoscopy_capsule_control.simulation import (
    HoverSimulation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run DEMA-MCE closed-loop hover simulation."
        )
    )

    parser.add_argument(
        "--time",
        type=float,
        default=10.0,
        help="Simulation duration [s].",
    )

    parser.add_argument(
        "--settling-time",
        type=float,
        default=5.0,
        help=(
            "Ignore samples before this time "
            "when computing RMS/STD metrics [s]."
        ),
    )

    parser.add_argument(
        "--x0-mm",
        type=float,
        default=0.0,
        help="Initial local-X perturbation [mm].",
    )

    parser.add_argument(
        "--z0-mm",
        type=float,
        default=0.0,
        help="Initial local-Z perturbation [mm].",
    )

    parser.add_argument(
        "--theta0-deg",
        type=float,
        default=0.0,
        help=(
            "Initial local-Y orientation perturbation [deg]."
        ),
    )

    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open MuJoCo viewer.",
    )

    parser.add_argument(
        "--camera",
        choices=[
            "capsule",
            "overview",
        ],
        default="capsule",
        help="Viewer camera mode.",
    )

    parser.add_argument(
        "--ideal-sensing",
        action="store_true",
        help=(
            "Disable measurement noise and delay."
        ),
    )

    parser.add_argument(
        "--position-noise-mm",
        type=float,
        default=0.36,
        help=(
            "Position measurement sigma "
            "[mm per local axis]."
        ),
    )

    parser.add_argument(
        "--theta-noise-deg",
        type=float,
        default=0.20,
        help=(
            "theta_y measurement sigma [deg]."
        ),
    )

    parser.add_argument(
        "--delay-samples",
        type=int,
        default=1,
        help=(
            "Measurement delay in controller samples. "
            "At 100 Hz, one sample = 10 ms."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Measurement noise random seed.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    simulation = HoverSimulation(
        simulation_time=(
            args.time
        ),
        settling_time=(
            args.settling_time
        ),
        x_offset=(
            args.x0_mm
            * 1e-3
        ),
        z_offset=(
            args.z0_mm
            * 1e-3
        ),
        theta_y_offset=np.deg2rad(
            args.theta0_deg
        ),
        sensing_enabled=(
            not args.ideal_sensing
        ),
        position_noise_std=(
            args.position_noise_mm
            * 1e-3
        ),
        theta_y_noise_std=np.deg2rad(
            args.theta_noise_deg
        ),
        measurement_delay_samples=(
            args.delay_samples
        ),
        measurement_seed=(
            args.seed
        ),
    )

    result = simulation.run(
        viewer=(
            args.viewer
        ),
        camera_mode=(
            args.camera
        ),
    )

    result.print_summary()


if __name__ == "__main__":
    main()