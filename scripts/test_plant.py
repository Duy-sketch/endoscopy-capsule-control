"""Standalone statistical checks for the three plant uncertainty models."""

from __future__ import annotations

import argparse

import numpy as np

from endoscopy_capsule_control.config import DEFAULT_CONFIG
from endoscopy_capsule_control.plant.localization import RFLocalizationModel
from endoscopy_capsule_control.plant.power_supply import DualChannelPowerSupply
from endoscopy_capsule_control.plant.robot_uncertainty import AuboI10PoseUncertainty


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--component",
        choices=["all", "current", "localization", "robot"],
        default="all",
    )
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def test_current(samples: int, seed: int) -> None:
    model = DualChannelPowerSupply(
        config=DEFAULT_CONFIG.power_supply,
        dt=DEFAULT_CONFIG.simulation.physics_dt,
        rng=np.random.default_rng(seed),
        noise_enabled=True,
    )
    model.reset([10.0, 10.0])
    noise = np.array([
        model.step([10.0, 10.0]).noise_A
        for _ in range(samples)
    ])
    rms = np.sqrt(np.mean(noise**2, axis=0))
    print("\n[CURRENT SOURCE]")
    print("Equivalent source:", DEFAULT_CONFIG.power_supply.model_name)
    print("Target RMS noise [mA]:", 1000 * DEFAULT_CONFIG.power_supply.current_noise_rms_A)
    print("Measured RMS [mA] ch1/ch2:", 1000 * rms)


def test_localization(samples: int, seed: int) -> None:
    model = RFLocalizationModel(
        config=DEFAULT_CONFIG.localization,
        rng=np.random.default_rng(seed),
        noise_enabled=True,
    )
    true_p = np.array([0.0, 0.0, 0.10])
    residual = np.array([
        model.sample(
            true_position_lcs_m=true_p,
            true_theta_y_rad=0.0,
        ).position_residual_m
        for _ in range(samples)
    ])
    axis_rms = np.sqrt(np.mean(residual**2, axis=0))
    norm_rmse = np.sqrt(np.mean(np.sum(residual**2, axis=1)))
    print("\n[RF LOCALIZATION]")
    print("Target sigma/axis [mm]:", 1000 * DEFAULT_CONFIG.localization.position_noise_std_m)
    print("Measured RMS/axis [mm]:", 1000 * axis_rms)
    print("Measured 3-D 2-norm RMSE [mm]:", 1000 * norm_rmse)
    print("Paper-derived target 3-D RMSE [mm]: 1.61")


def test_robot(samples: int, seed: int) -> None:
    model = AuboI10PoseUncertainty(
        config=DEFAULT_CONFIG.aubo_i10,
        rng=np.random.default_rng(seed),
        enabled=True,
    )
    bias = np.array([
        model.reset().translation_world_m
        for _ in range(samples)
    ])
    norm = np.linalg.norm(bias, axis=1)
    print("\n[AUBO i10]")
    print("Repeatability bound used [mm]:", 1000 * DEFAULT_CONFIG.aubo_i10.position_repeatability_m)
    print("Configured sigma/axis [mm]:", 1000 * DEFAULT_CONFIG.aubo_i10.position_bias_sigma_m)
    print("Observed RMS/axis [mm]:", 1000 * np.sqrt(np.mean(bias**2, axis=0)))
    print("Observed max bias norm [mm]:", 1000 * np.max(norm))


def main() -> None:
    args = parse_args()
    if args.component in {"all", "current"}:
        test_current(args.samples, args.seed)
    if args.component in {"all", "localization"}:
        test_localization(args.samples, args.seed + 1)
    if args.component in {"all", "robot"}:
        test_robot(args.samples, args.seed + 2)


if __name__ == "__main__":
    main()
