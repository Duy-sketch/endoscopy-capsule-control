"""Static sign/authority sanity check for the DEMA magnetic Z model."""

from __future__ import annotations

import numpy as np

from endoscopy_capsule_control.config import DEFAULT_CONFIG
from endoscopy_capsule_control.magnetic import magnetic_wrench
from endoscopy_capsule_control.simulation.capsule_state import get_capsule_state
from endoscopy_capsule_control.simulation.dema_geometry import (
    get_dema_geometry,
    world_to_lcs_vector,
)
from endoscopy_capsule_control.simulation.initialization import (
    initialize_hover_operating_point,
)
from endoscopy_capsule_control.simulation.mujoco_system import MujocoSystem


def main() -> None:
    cfg = DEFAULT_CONFIG
    system = MujocoSystem()
    initialize_hover_operating_point(system)
    geometry = get_dema_geometry(system)
    state = get_capsule_state(
        system=system,
        lcs_pose=geometry.lcs_pose,
        magnetic_moment_body=np.asarray(cfg.capsule.moment_body, dtype=float),
    )

    patterns = [
        (10.0, 10.0),
        (10.0, -10.0),
        (-10.0, 10.0),
        (-10.0, -10.0),
        (-15.0, 15.0),
    ]

    print("Capsule LCS position [m]:", state.position_lcs)
    print("\nI1, I2 [A] -> F_LCS [mN], Tau_LCS [mN*m]")
    for i1, i2 in patterns:
        force_w, torque_w, _ = magnetic_wrench(
            p_capsule=state.position_world,
            capsule_moment_world=state.magnetic_moment_world,
            em_positions=[geometry.em1.position_world, geometry.em2.position_world],
            em_axes=[geometry.em1.axis_world, geometry.em2.axis_world],
            currents=np.array([i1, i2]),
            em_gain=cfg.electromagnet.magnetic_gain,
            gradient_step=cfg.electromagnet.magnetic_gradient_step,
        )
        force_l = world_to_lcs_vector(force_w, geometry.lcs_pose)
        torque_l = world_to_lcs_vector(torque_w, geometry.lcs_pose)
        print(
            f"({i1:+6.1f}, {i2:+6.1f}) -> "
            f"F={1000*force_l}, Tau={1000*torque_l}"
        )


if __name__ == "__main__":
    main()
