"""
Outer-loop capsule X/Y controller.

The AUBO moves the DEMA. The nominal DEMA target follows the requested
capsule displacement and a small feedback lead is added in the direction
of capsule X/Y error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CapsuleXYGains:
    lead_gain: float = 0.30
    max_lead: float = 0.003  # m


@dataclass(frozen=True)
class CapsuleXYControlOutput:
    capsule_error_xy_world: np.ndarray
    nominal_dema_xy_world: np.ndarray
    feedback_lead_xy_world: np.ndarray
    desired_dema_xy_world: np.ndarray


def _clip_norm(vector: np.ndarray, maximum_norm: float) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(2).copy()
    norm = float(np.linalg.norm(value))

    if maximum_norm > 0.0 and norm > maximum_norm:
        value *= maximum_norm / norm

    return value


class CapsuleXYController:
    """Capsule X/Y position outer loop."""

    def __init__(
        self,
        *,
        gains: CapsuleXYGains | None = None,
    ):
        self.gains = gains if gains is not None else CapsuleXYGains()
        self._initialized = False
        self._initial_capsule_xy = np.zeros(2, dtype=float)
        self._initial_dema_xy = np.zeros(2, dtype=float)

    def reset(
        self,
        *,
        initial_capsule_position_world: np.ndarray,
        initial_dema_position_world: np.ndarray,
    ) -> None:
        self._initial_capsule_xy = np.asarray(
            initial_capsule_position_world,
            dtype=float,
        ).reshape(3)[:2].copy()

        self._initial_dema_xy = np.asarray(
            initial_dema_position_world,
            dtype=float,
        ).reshape(3)[:2].copy()

        self._initialized = True

    def compute(
        self,
        *,
        capsule_position_world: np.ndarray,
        target_capsule_position_world: np.ndarray,
    ) -> CapsuleXYControlOutput:
        if not self._initialized:
            raise RuntimeError(
                "CapsuleXYController.reset() must be called before compute()."
            )

        capsule_xy = np.asarray(
            capsule_position_world,
            dtype=float,
        ).reshape(3)[:2]

        target_xy = np.asarray(
            target_capsule_position_world,
            dtype=float,
        ).reshape(3)[:2]

        capsule_error = target_xy - capsule_xy

        target_displacement = (
            target_xy - self._initial_capsule_xy
        )

        nominal_dema_xy = (
            self._initial_dema_xy
            + target_displacement
        )

        feedback_lead = _clip_norm(
            self.gains.lead_gain * capsule_error,
            self.gains.max_lead,
        )

        desired_dema_xy = (
            nominal_dema_xy
            + feedback_lead
        )

        return CapsuleXYControlOutput(
            capsule_error_xy_world=capsule_error.copy(),
            nominal_dema_xy_world=nominal_dema_xy.copy(),
            feedback_lead_xy_world=feedback_lead.copy(),
            desired_dema_xy_world=desired_dema_xy.copy(),
        )
