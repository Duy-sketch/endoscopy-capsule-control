from __future__ import annotations

import unittest

import numpy as np

from endoscopy_capsule_control.config import DEFAULT_CONFIG
from endoscopy_capsule_control.plant.localization import RFLocalizationModel
from endoscopy_capsule_control.plant.noise import NoiseSwitches
from endoscopy_capsule_control.plant.power_supply import DualChannelPowerSupply
from endoscopy_capsule_control.plant.robot_uncertainty import AuboI10PoseUncertainty


class NoiseSwitchTests(unittest.TestCase):
    def test_all(self):
        n = NoiseSwitches.from_names(["all"])
        self.assertTrue(n.current and n.localization and n.robot)

    def test_none(self):
        n = NoiseSwitches.from_names(["none"])
        self.assertFalse(n.current or n.localization or n.robot)


class PowerSupplyTests(unittest.TestCase):
    def test_noise_disabled_is_exact(self):
        supply = DualChannelPowerSupply(
            config=DEFAULT_CONFIG.power_supply,
            dt=0.001,
            rng=np.random.default_rng(1),
            noise_enabled=False,
        )
        supply.reset()
        sample = supply.step([10.0, -10.0])
        np.testing.assert_allclose(sample.actual_A, [10.0, -10.0])
        np.testing.assert_allclose(sample.noise_A, [0.0, 0.0])

    def test_rms_noise_matches_config(self):
        supply = DualChannelPowerSupply(
            config=DEFAULT_CONFIG.power_supply,
            dt=0.001,
            rng=np.random.default_rng(2),
            noise_enabled=True,
        )
        supply.reset()
        values = np.array([
            supply.step([0.0, 0.0]).noise_A
            for _ in range(50000)
        ])
        rms = np.sqrt(np.mean(values**2, axis=0))
        target = DEFAULT_CONFIG.power_supply.current_noise_rms_A
        self.assertTrue(np.all(np.abs(rms - target) < 0.0002))


class LocalizationTests(unittest.TestCase):
    def test_noise_disabled_is_exact(self):
        model = RFLocalizationModel(
            config=DEFAULT_CONFIG.localization,
            rng=np.random.default_rng(3),
            noise_enabled=False,
        )
        p = np.array([0.01, -0.02, 0.10])
        sample = model.sample(true_position_lcs_m=p, true_theta_y_rad=0.0)
        np.testing.assert_allclose(sample.position_lcs_m, p)

    def test_dynamic_rmse_target(self):
        model = RFLocalizationModel(
            config=DEFAULT_CONFIG.localization,
            rng=np.random.default_rng(4),
            noise_enabled=True,
        )
        p = np.array([0.0, 0.0, 0.10])
        residuals = np.array([
            model.sample(
                true_position_lcs_m=p,
                true_theta_y_rad=0.0,
            ).position_residual_m
            for _ in range(30000)
        ])
        norm_rmse = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
        self.assertLess(abs(norm_rmse - 0.00161), 0.00005)


class AuboI10UncertaintyTests(unittest.TestCase):
    def test_disabled_is_zero(self):
        model = AuboI10PoseUncertainty(
            config=DEFAULT_CONFIG.aubo_i10,
            rng=np.random.default_rng(5),
            enabled=False,
        )
        err = model.reset()
        np.testing.assert_allclose(err.translation_world_m, np.zeros(3))

    def test_bias_is_bounded(self):
        model = AuboI10PoseUncertainty(
            config=DEFAULT_CONFIG.aubo_i10,
            rng=np.random.default_rng(6),
            enabled=True,
        )
        limit = DEFAULT_CONFIG.aubo_i10.position_repeatability_m
        for _ in range(10000):
            err = model.reset()
            self.assertLessEqual(
                np.linalg.norm(err.translation_world_m),
                limit + 1e-15,
            )


if __name__ == "__main__":
    unittest.main()
