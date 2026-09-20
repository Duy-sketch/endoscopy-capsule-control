"""Command-line entry point for the refactored Z-hover experiment."""

from __future__ import annotations

import argparse

import numpy as np

from endoscopy_capsule_control.plant.noise import NoiseSwitches
from endoscopy_capsule_control.simulation.z_hover_simulation import ZHoverSimulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DEMA-MCE Z-hover with selectable plant uncertainties."
    )
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument("--settling-time", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--x0-mm", type=float, default=0.0)
    parser.add_argument("--z0-mm", type=float, default=0.0)
    parser.add_argument("--theta0-deg", type=float, default=0.0)

    parser.add_argument(
        "--noise",
        nargs="+",
        default=["none"],
        choices=["none", "current", "localization", "robot", "all"],
        help=(
            "Plant uncertainty sources. Examples: '--noise none', "
            "'--noise current localization', or '--noise all'."
        ),
    )

    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--camera",
        choices=["capsule", "overview"],
        default="capsule",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    noise = NoiseSwitches.from_names(args.noise)

    simulation = ZHoverSimulation(
        noise=noise,
        seed=args.seed,
        simulation_time_s=args.time,
        settling_time_s=args.settling_time,
        x_offset_m=args.x0_mm * 1e-3,
        z_offset_m=args.z0_mm * 1e-3,
        theta_y_offset_rad=np.deg2rad(args.theta0_deg),
    )
    result = simulation.run(
        viewer=args.viewer,
        camera_mode=args.camera,
    )
    result.print_summary()


if __name__ == "__main__":
    main()
