"""Equivalent dual-channel power-supply model for the DEMA coils."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from endoscopy_capsule_control.config import PowerSupplyConfig


@dataclass(frozen=True)
class PowerSupplySample:
    """Result of one power-supply update."""

    command_A: np.ndarray
    clipped_command_A: np.ndarray
    clean_output_A: np.ndarray
    noise_A: np.ndarray
    actual_A: np.ndarray


class DualChannelPowerSupply:
    """Two independent current-source channels.

    The baseline noise level is configured from a Kepco BOP 20-20-
    equivalent source.  Noise is independently sampled for the two DEMA
    channels, which is important because the paper uses two independent
    power units.

    A first-order response can be enabled through the config, but the
    default is instantaneous response because the DEMA paper does not
    provide coil L/R parameters or identify the actual power supply.
    """

    def __init__(
        self,
        *,
        config: PowerSupplyConfig,
        dt: float,
        rng: np.random.Generator,
        noise_enabled: bool,
    ) -> None:
        self.config = config
        self.dt = float(dt)
        self.rng = rng
        self.noise_enabled = bool(noise_enabled)

        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.config.current_abs_max <= 0.0:
            raise ValueError("current_abs_max must be positive")
        if self.config.current_noise_rms_A < 0.0:
            raise ValueError("current_noise_rms_A must be non-negative")
        if self.config.response_time_constant_s < 0.0:
            raise ValueError("response_time_constant_s must be non-negative")

        self._clean_output = np.zeros(2, dtype=float)
        self.last_sample = PowerSupplySample(
            command_A=np.zeros(2),
            clipped_command_A=np.zeros(2),
            clean_output_A=np.zeros(2),
            noise_A=np.zeros(2),
            actual_A=np.zeros(2),
        )

    def reset(self, initial_current_A=(0.0, 0.0)) -> None:
        current = np.asarray(initial_current_A, dtype=float).reshape(2)
        current = np.clip(
            current,
            -self.config.current_abs_max,
            +self.config.current_abs_max,
        )
        self._clean_output = current.copy()
        self.last_sample = PowerSupplySample(
            command_A=current.copy(),
            clipped_command_A=current.copy(),
            clean_output_A=current.copy(),
            noise_A=np.zeros(2),
            actual_A=current.copy(),
        )

    def _advance_clean_output(self, command_A: np.ndarray) -> np.ndarray:
        tau = float(self.config.response_time_constant_s)
        if tau <= 0.0:
            self._clean_output = command_A.copy()
            return self._clean_output.copy()

        alpha = 1.0 - np.exp(-self.dt / tau)
        self._clean_output = (
            self._clean_output
            + alpha * (command_A - self._clean_output)
        )
        return self._clean_output.copy()

    def step(self, command_A) -> PowerSupplySample:
        command = np.asarray(command_A, dtype=float).reshape(2)
        clipped = np.clip(
            command,
            -self.config.current_abs_max,
            +self.config.current_abs_max,
        )

        clean = self._advance_clean_output(clipped)

        if self.noise_enabled and self.config.current_noise_rms_A > 0.0:
            noise = self.rng.normal(
                loc=0.0,
                scale=self.config.current_noise_rms_A,
                size=2,
            )
        else:
            noise = np.zeros(2, dtype=float)

        actual = np.clip(
            clean + noise,
            -self.config.current_abs_max,
            +self.config.current_abs_max,
        )

        sample = PowerSupplySample(
            command_A=command.copy(),
            clipped_command_A=clipped.copy(),
            clean_output_A=clean.copy(),
            noise_A=noise.copy(),
            actual_A=actual.copy(),
        )
        self.last_sample = sample
        return sample
