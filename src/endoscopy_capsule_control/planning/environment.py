"""Known XY map setup and validation, shared with the MuJoCo scene builder."""

import numpy as np
from .astar import XYPlan
from endoscopy_capsule_control.simulation.scene_builder import (
    install_scene,
    planning_bounds_from_tank,
)


def prepare_path(plant, options, origin, rotation):
    settings = dict(
        obstacles_x_y_radius_mm=[[25, 0, 5]],
        walls_cx_cy_width_height_mm=[],
        center_bounds_mm=[-60, 80, -60, 60],
        goal_xy_mm=[50, 0],
        grid_mm=2,
        margin_mm=2,
        speed_mm_s=3,
        acceleration_mm_s2=2,
        warmup_s=2,
        hold_s=5,
        goal_tolerance_mm=2,
        goal_speed_mm_s=0.5,
        goal_hold_s=1,
    )
    extra = options.path or {}
    if not isinstance(extra, dict) or set(extra) - set(settings):
        raise ValueError("Unknown path configuration keys.")
    settings.update(extra)
    if options.target_mm is not None:
        settings["goal_xy_mm"] = options.target_mm
    if options.speed_mm_s is not None:
        settings["speed_mm_s"] = options.speed_mm_s
    for key in ("hold_s", "goal_tolerance_mm", "goal_speed_mm_s", "goal_hold_s"):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if settings["goal_hold_s"] > settings["hold_s"]:
        raise ValueError("hold_s must cover goal_hold_s.")
    if options.z0_mm != 0:
        raise ValueError(
            "XY planning starts at the maintained Z reference; use z0-mm 0."
        )
    if settings["speed_mm_s"] > plant.config.aubo_motion.max_linear_speed_mps * 1000:
        raise ValueError("Trajectory speed exceeds robot configured speed limit.")
    goal = np.asarray(settings["goal_xy_mm"], float)
    if goal.shape != (2,):
        raise ValueError("goal_xy_mm must contain X,Y.")
    shell = plant.system.model.geom("MCE_shell").id
    radius = float(sum(plant.system.model.geom_size[shell, :2]) * 1000)
    if not np.isclose(radius, plant.config.capsule.length * 500):
        raise ValueError("Capsule XML/config length mismatch.")
    z = plant.config.z_control.z_ref * 1000
    bounds = planning_bounds_from_tank(
        plant,
        origin,
        rotation,
        z,
        radius,
        settings["margin_mm"],
        settings["center_bounds_mm"],
    )
    plan = XYPlan(
        [options.x0_mm, options.y0_mm, z],
        [*goal, z],
        settings["obstacles_x_y_radius_mm"],
        bounds,
        settings["grid_mm"],
        radius,
        settings["margin_mm"],
        settings["speed_mm_s"],
        settings["acceleration_mm_s2"],
        plant.config.simulation.control_dt,
        settings["warmup_s"],
        rectangles=settings["walls_cx_cy_width_height_mm"],
    )
    install_scene(plant, plan, origin, rotation, options.csv.with_suffix(".scene.xml"))
    options.time = plan.motion_end + settings["hold_s"]
    options.move_time = plan.motion_end
    if options.settling_time >= options.time:
        raise ValueError("settling-time exceeds generated duration.")
    print("A* waypoints XY [mm]:", plan.path.tolist())
    print(
        f"Path length {plan.length_mm:.2f} mm; total time {options.time:.2f} s (automatic)"
    )
    return plan, settings
