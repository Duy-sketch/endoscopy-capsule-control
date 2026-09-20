"""
Differential inverse kinematics for the AUBO-i10.

The module converts a desired Cartesian twist of the DEMA LCS frame
into AUBO joint-velocity commands using damped least squares.

The AUBO position actuators then receive integrated joint-position
commands.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .initialization import (
    AUBO_ACTUATOR_NAMES,
    AUBO_JOINT_NAMES,
)

from .mujoco_system import (
    MujocoSystem,
)


class AUBODifferentialIK:
    """
    Differential IK servo for the six AUBO joints.
    """

    def __init__(
        self,
        *,
        system: MujocoSystem,
        site_name: str = "LCS_origin",
        damping: float = 1e-3,
        max_joint_speed: float = 0.8,
    ):
        if damping <= 0.0:
            raise ValueError(
                "damping must be positive."
            )

        if max_joint_speed <= 0.0:
            raise ValueError(
                "max_joint_speed must be positive."
            )

        self.system = system

        self.damping = float(
            damping
        )

        self.max_joint_speed = float(
            max_joint_speed
        )

        # ====================================================
        # DEMA REFERENCE SITE
        # ====================================================

        self.site_id = mujoco.mj_name2id(
            self.system.model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        if self.site_id < 0:
            raise ValueError(
                f"Site not found: {site_name}"
            )

        # ====================================================
        # AUBO JOINT INDICES
        # ====================================================

        self.joint_ids = []

        self.qpos_indices = []

        self.dof_indices = []

        for joint_name in AUBO_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.system.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if joint_id < 0:
                raise ValueError(
                    f"AUBO joint not found: {joint_name}"
                )

            self.joint_ids.append(
                joint_id
            )

            self.qpos_indices.append(
                int(
                    self.system
                    .model
                    .jnt_qposadr[
                        joint_id
                    ]
                )
            )

            self.dof_indices.append(
                int(
                    self.system
                    .model
                    .jnt_dofadr[
                        joint_id
                    ]
                )
            )

        self.qpos_indices = np.array(
            self.qpos_indices,
            dtype=int,
        )

        self.dof_indices = np.array(
            self.dof_indices,
            dtype=int,
        )

        # ====================================================
        # AUBO ACTUATORS
        # ====================================================

        self.actuator_ids = []

        for actuator_name in AUBO_ACTUATOR_NAMES:
            actuator_id = mujoco.mj_name2id(
                self.system.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )

            if actuator_id < 0:
                raise ValueError(
                    f"AUBO actuator not found: "
                    f"{actuator_name}"
                )

            self.actuator_ids.append(
                actuator_id
            )

        self.actuator_ids = np.array(
            self.actuator_ids,
            dtype=int,
        )

        if len(
            self.actuator_ids
        ) != 6:
            raise RuntimeError(
                "Expected six AUBO actuators."
            )

        # ====================================================
        # INTERNAL JOINT COMMAND
        # ====================================================

        self.q_command = (
            self.current_joint_position()
        )

        self.apply_joint_command()

    # ========================================================
    # STATE
    # ========================================================

    def current_joint_position(
        self,
    ) -> np.ndarray:
        return (
            self.system
            .data
            .qpos[
                self.qpos_indices
            ]
            .copy()
        )

    def site_pose(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Return DEMA LCS-origin pose in the world frame.
        """

        position = (
            self.system
            .data
            .site_xpos[
                self.site_id
            ]
            .copy()
        )

        rotation = (
            self.system
            .data
            .site_xmat[
                self.site_id
            ]
            .reshape(
                3,
                3,
            )
            .copy()
        )

        return (
            position,
            rotation,
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset_command_to_current(
        self,
    ) -> None:
        """
        Make actuator commands equal to the current AUBO configuration.
        """

        self.q_command = (
            self.current_joint_position()
        )

        self.apply_joint_command()

    # ========================================================
    # JACOBIAN
    # ========================================================

    def jacobian(
        self,
    ) -> np.ndarray:
        """
        Return 6x6 Jacobian of LCS_origin with respect to AUBO joints.

        Row convention:

            [linear velocity]
            [angular velocity]
        """

        jac_position = np.zeros(
            (
                3,
                self.system.model.nv,
            ),
            dtype=float,
        )

        jac_rotation = np.zeros(
            (
                3,
                self.system.model.nv,
            ),
            dtype=float,
        )

        mujoco.mj_jacSite(
            self.system.model,
            self.system.data,
            jac_position,
            jac_rotation,
            self.site_id,
        )

        J = np.vstack(
            [
                jac_position[
                    :,
                    self.dof_indices
                ],
                jac_rotation[
                    :,
                    self.dof_indices
                ],
            ]
        )

        return J

    # ========================================================
    # DIFFERENTIAL IK
    # ========================================================

    def solve_joint_velocity(
        self,
        twist_world: np.ndarray,
    ) -> np.ndarray:
        """
        Solve qdot from desired Cartesian twist.
        """

        twist = np.asarray(
            twist_world,
            dtype=float,
        ).reshape(6)

        J = self.jacobian()

        regularization = (
            self.damping**2
            * np.eye(
                6,
                dtype=float,
            )
        )

        qdot = (
            J.T
            @ np.linalg.solve(
                J
                @ J.T
                + regularization,
                twist,
            )
        )

        qdot = np.clip(
            qdot,
            -self.max_joint_speed,
            +self.max_joint_speed,
        )

        return qdot

    # ========================================================
    # COMMAND
    # ========================================================

    def command_twist(
        self,
        *,
        twist_world: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Convert desired Cartesian twist to a joint-position command.
        """

        if dt <= 0.0:
            raise ValueError(
                "dt must be positive."
            )

        qdot = (
            self.solve_joint_velocity(
                twist_world
            )
        )

        self.q_command = (
            self.q_command
            + qdot
            * float(
                dt
            )
        )

        # ----------------------------------------------------
        # Joint-limit protection
        # ----------------------------------------------------

        for i, joint_id in enumerate(
            self.joint_ids
        ):
            if bool(
                self.system
                .model
                .jnt_limited[
                    joint_id
                ]
            ):
                lower = float(
                    self.system
                    .model
                    .jnt_range[
                        joint_id,
                        0,
                    ]
                )

                upper = float(
                    self.system
                    .model
                    .jnt_range[
                        joint_id,
                        1,
                    ]
                )

                self.q_command[i] = (
                    np.clip(
                        self.q_command[i],
                        lower,
                        upper,
                    )
                )

        self.apply_joint_command()

        return qdot

    def apply_joint_command(
        self,
    ) -> None:
        """
        Send q_command to the six AUBO position actuators.
        """

        for i, actuator_id in enumerate(
            self.actuator_ids
        ):
            command = float(
                self.q_command[i]
            )

            if bool(
                self.system
                .model
                .actuator_ctrllimited[
                    actuator_id
                ]
            ):
                lower = float(
                    self.system
                    .model
                    .actuator_ctrlrange[
                        actuator_id,
                        0,
                    ]
                )

                upper = float(
                    self.system
                    .model
                    .actuator_ctrlrange[
                        actuator_id,
                        1,
                    ]
                )

                command = float(
                    np.clip(
                        command,
                        lower,
                        upper,
                    )
                )

                self.q_command[i] = (
                    command
                )

            self.system.data.ctrl[
                actuator_id
            ] = command