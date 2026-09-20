"""AUBO i10 end-effector / DEMA pose uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from endoscopy_capsule_control.config import AuboI10Config


@dataclass(frozen=True)
class AuboI10PoseError:
    """Fixed per-episode pose error currently represented as translation."""

    translation_world_m: np.ndarray


class AuboI10PoseUncertainty:
    """Episode-wise AUBO i10 DEMA-mount position uncertainty.

    Repeatability is not modeled as 1-kHz white jitter.  Instead, one
    small Cartesian placement bias is sampled at reset and held fixed for
    the episode.  This is a better first approximation for a stationary
    AUBO during the Z-hover experiment.
    """

    def __init__(
        self,
        *,
        config: AuboI10Config,
        rng: np.random.Generator,
        enabled: bool,
    ) -> None:
        self.config = config
        self.rng = rng
        self.enabled = bool(enabled)
        self.error = AuboI10PoseError(np.zeros(3, dtype=float))

    def reset(self) -> AuboI10PoseError:
        if not self.enabled:
            translation = np.zeros(3, dtype=float)
        else:
            translation = self.rng.normal(
                0.0,
                self.config.position_bias_sigma_m,
                size=3,
            )
            magnitude = float(np.linalg.norm(translation))
            limit = float(self.config.position_repeatability_m)
            if magnitude > limit > 0.0:
                translation = translation * (limit / magnitude)

        self.error = AuboI10PoseError(translation_world_m=translation.copy())
        return self.error

    def apply_to_geometry(self, geometry: Any) -> Any:
        """Translate the complete DEMA geometry by the sampled pose bias.

        ``dataclasses.replace`` keeps this module independent of MuJoCo and
        of the concrete geometry dataclass definitions.
        """

        shift = self.error.translation_world_m
        if not np.any(shift):
            return geometry

        lcs_pose = replace(
            geometry.lcs_pose,
            position=np.asarray(geometry.lcs_pose.position) + shift,
        )
        em1 = replace(
            geometry.em1,
            position_world=np.asarray(geometry.em1.position_world) + shift,
        )
        em2 = replace(
            geometry.em2,
            position_world=np.asarray(geometry.em2.position_world) + shift,
        )
        return replace(geometry, lcs_pose=lcs_pose, em1=em1, em2=em2)
