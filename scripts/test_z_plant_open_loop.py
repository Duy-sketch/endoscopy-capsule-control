"""Open-loop Z plant test before enabling the controller."""

from __future__ import annotations

import argparse

import numpy as np

from endoscopy_capsule_control.plant.noise import NoiseSwitches
from endoscopy_capsule_control.plant.z_hover_plant import ZHoverPlant


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--time", type=float, default=2.0)
    p.add_argument("--i1", type=float, default=-15.0)
    p.add_argument("--i2", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--noise",
        nargs="+",
        default=["none"],
        choices=["none", "current", "localization", "robot", "all"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    noise = NoiseSwitches.from_names(args.noise)
    plant = ZHoverPlant(noise=noise, seed=args.seed)

    command = np.array([args.i1, args.i2], dtype=float)
    initial = plant.true_state()
    n = int(round(args.time / plant.physics_dt))

    sum_sq_i_noise = np.zeros(2)
    localization_residuals = []

    control_every = int(round(0.010 / plant.physics_dt))
    for k in range(n):
        snapshot = plant.step(command)
        sum_sq_i_noise += snapshot.power_supply.noise_A**2
        if (k + 1) % control_every == 0:
            m = plant.sample_localization()
            localization_residuals.append(m.position_residual_m.copy())

    final = plant.true_state()
    rms_i = np.sqrt(sum_sq_i_noise / max(n, 1))

    print("Noise:", noise.enabled_names() or ("none",))
    print("AUBO pose bias XYZ [mm]:", 1000 * plant.robot_error.translation_world_m)
    print("Initial true LCS position [mm]:", 1000 * initial.position_lcs)
    print("Final true LCS position [mm]:", 1000 * final.position_lcs)
    print("Delta true LCS position [mm]:", 1000 * (final.position_lcs - initial.position_lcs))
    print("Current noise RMS [mA]:", 1000 * rms_i)

    if localization_residuals:
        e = np.asarray(localization_residuals)
        print(
            "Localization residual RMS XYZ [mm]:",
            1000 * np.sqrt(np.mean(e**2, axis=0)),
        )


if __name__ == "__main__":
    main()
