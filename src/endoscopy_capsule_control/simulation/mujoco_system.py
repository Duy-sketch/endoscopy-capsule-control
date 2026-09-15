"""
Generic MuJoCo interface for the DEMA-MCE simulation.

Responsibilities:
- locate and load the MuJoCo XML model,
- hold MjModel and MjData,
- reset / forward / step the simulation,
- query MuJoCo objects by name,
- read body and site poses.

Magnetic physics, fluid dynamics and controllers do not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class Pose:
    """
    Pose expressed in the MuJoCo world frame.

    position
        Position vector [m], shape (3,).

    rotation
        Rotation matrix R_world_object, shape (3, 3).
    """

    position: np.ndarray
    rotation: np.ndarray


def default_model_path() -> Path:
    """
    Return the packaged AUBO i10 MuJoCo model path.
    """

    package_root = Path(
        __file__
    ).resolve().parents[1]

    return (
        package_root
        / "models"
        / "aubo_i10.xml"
    )


class MujocoSystem:
    """
    Thin wrapper around MuJoCo model and data.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
    ):
        if model_path is None:
            model_path = default_model_path()

        self.model_path = Path(
            model_path
        ).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MuJoCo XML not found: "
                f"{self.model_path}"
            )

        self.model = (
            mujoco.MjModel.from_xml_path(
                str(self.model_path)
            )
        )

        self.data = mujoco.MjData(
            self.model
        )

        # Compute the initial kinematics.
        mujoco.mj_forward(
            self.model,
            self.data,
        )


    @property
    def time(self) -> float:
        """
        Current MuJoCo simulation time [s].
        """

        return float(
            self.data.time
        )


    @property
    def timestep(self) -> float:
        """
        MuJoCo physics timestep [s].
        """

        return float(
            self.model.opt.timestep
        )


    def reset(self) -> None:
        """
        Reset simulation state to the XML initial state.
        """

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )


    def forward(self) -> None:
        """
        Recompute MuJoCo kinematics/dynamics without advancing time.

        Use this after manually modifying qpos or qvel.
        """

        mujoco.mj_forward(
            self.model,
            self.data,
        )


    def step(self) -> None:
        """
        Advance the simulation by one MuJoCo physics step.
        """

        mujoco.mj_step(
            self.model,
            self.data,
        )


    # ========================================================
    # Name -> ID helpers
    # ========================================================

    def _name_to_id(
        self,
        object_type,
        name: str,
    ) -> int:
        object_id = mujoco.mj_name2id(
            self.model,
            object_type,
            name,
        )

        if object_id < 0:
            raise KeyError(
                f"MuJoCo object not found: {name!r}"
            )

        return int(
            object_id
        )


    def body_id(
        self,
        name: str,
    ) -> int:
        return self._name_to_id(
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )


    def site_id(
        self,
        name: str,
    ) -> int:
        return self._name_to_id(
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )


    def joint_id(
        self,
        name: str,
    ) -> int:
        return self._name_to_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )


    def actuator_id(
        self,
        name: str,
    ) -> int:
        return self._name_to_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )


    # ========================================================
    # Pose queries
    # ========================================================

    def get_body_pose(
        self,
        name: str,
    ) -> Pose:
        """
        Return a body pose in world coordinates.
        """

        body_id = self.body_id(
            name
        )

        position = np.asarray(
            self.data.xpos[
                body_id
            ],
            dtype=float,
        ).copy()

        rotation = np.asarray(
            self.data.xmat[
                body_id
            ],
            dtype=float,
        ).reshape(
            3,
            3,
        ).copy()

        return Pose(
            position=position,
            rotation=rotation,
        )


    def get_site_pose(
        self,
        name: str,
    ) -> Pose:
        """
        Return a site pose in world coordinates.
        """

        site_id = self.site_id(
            name
        )

        position = np.asarray(
            self.data.site_xpos[
                site_id
            ],
            dtype=float,
        ).copy()

        rotation = np.asarray(
            self.data.site_xmat[
                site_id
            ],
            dtype=float,
        ).reshape(
            3,
            3,
        ).copy()

        return Pose(
            position=position,
            rotation=rotation,
        )


    # ========================================================
    # Model inspection
    # ========================================================

    def _object_names(
        self,
        object_type,
        count: int,
    ) -> list[str]:
        names = []

        for object_id in range(
            count
        ):
            name = mujoco.mj_id2name(
                self.model,
                object_type,
                object_id,
            )

            if name is not None:
                names.append(
                    name
                )

        return names


    def body_names(
        self,
    ) -> list[str]:
        return self._object_names(
            mujoco.mjtObj.mjOBJ_BODY,
            self.model.nbody,
        )


    def joint_names(
        self,
    ) -> list[str]:
        return self._object_names(
            mujoco.mjtObj.mjOBJ_JOINT,
            self.model.njnt,
        )


    def site_names(
        self,
    ) -> list[str]:
        return self._object_names(
            mujoco.mjtObj.mjOBJ_SITE,
            self.model.nsite,
        )


    def actuator_names(
        self,
    ) -> list[str]:
        return self._object_names(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            self.model.nu,
        )