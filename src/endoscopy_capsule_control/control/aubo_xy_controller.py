"""
Cartesian XY controller for the AUBO-i10.

The controller moves the DEMA reference frame in world X-Y while:

    - holding its world Z position,
    - holding its orientation.

It produces a desired Cartesian twist:

    [vx, vy, vz, wx, wy, wz]

The conversion from Cartesian twist to AUBO joint motion is handled
separately by the differential-IK module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AUBOXYGains:
    """
    Cartesian servo gains.

    kp_xy
        XY position gain [1/s].

    kp_z_hold
        Z holding gain [1/s].

    kp_rotation
        Orientation holding gain [1/s].

    max_linear_speed
        Maximum Cartesian speed [m/s].

    max_angular_speed
        Maximum angular speed [rad/s].
    """

    kp_xy: float = 2.0
    kp_z_hold: float = 2.0
    kp_rotation: float = 2.0

    max_linear_speed: float = 0.03
    max_angular_speed: float = 0.50


@dataclass(frozen=True)
class AUBOXYControlOutput:
    """
    Result of one Cartesian-controller update.
    """

    twist_world: np.ndarray

    position_error_world: np.ndarray
    orientation_error_world: np.ndarray


def _clip_vector_norm(
    vector: np.ndarray,
    maximum_norm: float,
) -> np.ndarray:
    vector = np.asarray(
        vector,
        dtype=float,
    ).copy()

    norm = float(
        np.linalg.norm(
            vector
        )
    )

    if (
        maximum_norm > 0.0
        and norm > maximum_norm
    ):
        vector *= (
            maximum_norm
            / norm
        )

    return vector


def rotation_error_world(
    current_rotation: np.ndarray,
    target_rotation: np.ndarray,
) -> np.ndarray:
    """
    Orientation error expressed as a world-frame rotation vector.

    The returned vector approximately represents the angular
    displacement required to rotate the current orientation into
    the target orientation.
    """

    R_current = np.asarray(
        current_rotation,
        dtype=float,
    ).reshape(
        3,
        3,
    )

    R_target = np.asarray(
        target_rotation,
        dtype=float,
    ).reshape(
        3,
        3,
    )

    R_error = (
        R_target
        @ R_current.T
    )

    cos_angle = float(
        np.clip(
            (
                np.trace(
                    R_error
                )
                - 1.0
            )
            / 2.0,
            -1.0,
            1.0,
        )
    )

    angle = float(
        np.arccos(
            cos_angle
        )
    )

    if angle < 1e-10:
        return np.zeros(
            3,
            dtype=float,
        )

    sin_angle = float(
        np.sin(
            angle
        )
    )

    if abs(
        sin_angle
    ) < 1e-8:
        # This controller operates only around the AUBO work pose,
        # so a near-pi orientation error is not expected.
        return np.zeros(
            3,
            dtype=float,
        )

    axis = np.array(
        [
            R_error[2, 1]
            - R_error[1, 2],

            R_error[0, 2]
            - R_error[2, 0],

            R_error[1, 0]
            - R_error[0, 1],
        ],
        dtype=float,
    )

    axis /= (
        2.0
        * sin_angle
    )

    return (
        angle
        * axis
    )


class AUBOXYController:
    """
    Cartesian pose servo used for XY positioning of the DEMA.

    X and Y are task variables.

    Z and orientation are held at their initial/reference values.
    """

    def __init__(
        self,
        *,
        gains: AUBOXYGains | None = None,
    ):
        self.gains = (
            gains
            if gains is not None
            else AUBOXYGains()
        )

    def compute(
        self,
        *,
        current_position_world: np.ndarray,
        current_rotation_world: np.ndarray,
        target_position_world: np.ndarray,
        target_rotation_world: np.ndarray,
    ) -> AUBOXYControlOutput:
        current_position = np.asarray(
            current_position_world,
            dtype=float,
        ).reshape(3)

        target_position = np.asarray(
            target_position_world,
            dtype=float,
        ).reshape(3)

        position_error = (
            target_position
            - current_position
        )

        orientation_error = (
            rotation_error_world(
                current_rotation=(
                    current_rotation_world
                ),
                target_rotation=(
                    target_rotation_world
                ),
            )
        )

        linear_velocity = np.array(
            [
                self.gains.kp_xy
                * position_error[0],

                self.gains.kp_xy
                * position_error[1],

                self.gains.kp_z_hold
                * position_error[2],
            ],
            dtype=float,
        )

        angular_velocity = (
            self.gains.kp_rotation
            * orientation_error
        )

        linear_velocity = (
            _clip_vector_norm(
                linear_velocity,
                self.gains.max_linear_speed,
            )
        )

        angular_velocity = (
            _clip_vector_norm(
                angular_velocity,
                self.gains.max_angular_speed,
            )
        )

        twist_world = np.concatenate(
            [
                linear_velocity,
                angular_velocity,
            ]
        )

        return AUBOXYControlOutput(
            twist_world=(
                twist_world
            ),
            position_error_world=(
                position_error
            ),
            orientation_error_world=(
                orientation_error
            ),
        )