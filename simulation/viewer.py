"""
MuJoCo viewer utilities.

Visualization is kept separate from the simulation physics and
control logic.

Two camera modes are provided:

1. Capsule tracking
   The camera automatically follows the MCE body.

2. DEMA overview
   The camera looks at the DEMA local-coordinate-system origin.
"""

from __future__ import annotations

import mujoco


def configure_capsule_tracking_camera(
    viewer,
    system,
    *,
    distance: float = 0.18,
    azimuth: float = 90.0,
    elevation: float = -20.0,
) -> None:
    """
    Configure the MuJoCo viewer to track the capsule.

    Parameters
    ----------
    viewer
        Passive MuJoCo viewer handle.

    system
        MujocoSystem instance.

    distance
        Camera distance from the tracked capsule [m].

    azimuth
        Camera azimuth [deg].

    elevation
        Camera elevation [deg].
    """

    mce_body_id = system.body_id(
        "MCE"
    )

    with viewer.lock():
        viewer.cam.type = (
            mujoco.mjtCamera.mjCAMERA_TRACKING
        )

        viewer.cam.trackbodyid = (
            mce_body_id
        )

        viewer.cam.distance = float(
            distance
        )

        viewer.cam.azimuth = float(
            azimuth
        )

        viewer.cam.elevation = float(
            elevation
        )


def configure_dema_overview_camera(
    viewer,
    system,
    *,
    distance: float = 0.45,
    azimuth: float = 90.0,
    elevation: float = -25.0,
) -> None:
    """
    Configure a free camera looking at the DEMA LCS origin.

    This mode is useful when both the electromagnets and capsule
    should remain visible.
    """

    lcs_site_id = system.site_id(
        "LCS_origin"
    )

    lcs_position = (
        system.data.site_xpos[
            lcs_site_id
        ].copy()
    )

    with viewer.lock():
        viewer.cam.type = (
            mujoco.mjtCamera.mjCAMERA_FREE
        )

        viewer.cam.lookat[:] = (
            lcs_position
        )

        viewer.cam.distance = float(
            distance
        )

        viewer.cam.azimuth = float(
            azimuth
        )

        viewer.cam.elevation = float(
            elevation
        )