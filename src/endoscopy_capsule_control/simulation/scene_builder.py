"""Physical posts and visual path built from the same map as the planner."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def install_scene(plant, plan, origin, rotation, scene_path):
    """Add static physical cylinders; preserve initialized generalized state.

    No capsule pose is overwritten during simulation. The one-time model
    rebuild adds only fixed geoms, keeping nq/nv/nu and existing dynamics.
    """
    import mujoco
    from endoscopy_capsule_control.simulation.mujoco_system import MujocoSystem

    old = plant.system
    root = ET.parse(old.model_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for key in ("meshdir", "texturedir"):
        compiler.set(
            key, str((old.model_path.parent / compiler.get(key, ".")).resolve())
        )
    for asset in root.findall("./asset/*"):
        if asset.tag not in ("mesh", "texture") and asset.get("file"):
            asset.set(
                "file", str((old.model_path.parent / asset.get("file")).resolve())
            )
    world = root.find("worldbody")
    oil = old.data.geom_xpos[plant.oil_geom_id]
    half = old.model.geom_size[plant.oil_geom_id]
    # Posts fixed to tank bottom, ending 5 mm below the oil surface.
    zbottom = oil[2] - half[2]
    ztop = oil[2] + half[2] - 0.005
    for index, (x, y, radius) in enumerate(plan.obstacles):
        center = origin + rotation @ np.array([x, y, plan.start[2]]) / 1000
        if np.any(np.abs(center[:2] - oil[:2]) + radius / 1000 >= half[:2]):
            raise ValueError("An obstacle does not fit inside the tank.")
        ET.SubElement(
            world,
            "geom",
            name=f"path_obstacle_{index}",
            type="cylinder",
            pos=" ".join(map(str, [center[0], center[1], (zbottom + ztop) / 2])),
            size=f"{radius/1000} {(ztop-zbottom)/2}",
            rgba=".75 .20 .12 1",
            contype="4",
            conaffinity="3",
            friction=".4 .01 .001",
            group="0",
        )

    # Real flat walls: axis-aligned boxes in the fixed map frame.
    for index, (x, y, width, depth) in enumerate(plan.rectangles):
        center = origin + rotation @ np.array([x, y, plan.start[2]]) / 1000
        extent = np.abs(rotation[:2, :2]) @ np.array([width, depth]) / 2000
        if np.any(np.abs(center[:2] - oil[:2]) + extent >= half[:2]):
            raise ValueError("A wall does not fit inside the tank.")
        ET.SubElement(world, "geom", name=f"path_obstacle_wall_{index}",
            type="box", pos=" ".join(map(str, [*center[:2], (zbottom+ztop)/2])),
            size=f"{width/2000} {depth/2000} {(ztop-zbottom)/2}",
            rgba=".2 .55 .55 1", contype="4", conaffinity="3",
            friction=".4 .01 .001", group="0")

    def line(name, p, q, color, width=0.00035):
        if np.linalg.norm(q - p) < 1e-10:
            return
        ET.SubElement(
            world,
            "geom",
            name=name,
            type="capsule",
            fromto=" ".join(map(str, np.r_[p, q])),
            size=str(width),
            contype="0",
            conaffinity="0",
            rgba=color,
            group="2",
            mass="0",
        )

    pts = np.array(
        [origin + rotation @ np.r_[p, plan.start[2]] / 1000 for p in plan.path]
    )
    for i in range(len(pts) - 1):
        line(f"path_reference_{i}", pts[i], pts[i + 1], "0 .8 .3 .8")
    for index, (obstacle, radius) in enumerate(zip(plan.obstacles, plan.inflated)):
        theta = np.linspace(0, 2 * np.pi, 49)
        circle = [
            origin
            + rotation
            @ np.array(
                [
                    obstacle[0] + radius * np.cos(t),
                    obstacle[1] + radius * np.sin(t),
                    plan.start[2],
                ]
            )
            / 1000
            for t in theta
        ]
        for j in range(len(circle) - 1):
            line(
                f"path_margin_{index}_{j}",
                circle[j],
                circle[j + 1],
                "1 .65 0 .55",
                0.00015,
            )
    from endoscopy_capsule_control.planning.collision import rounded_rectangle_outline
    for index, wall in enumerate(plan.rectangles):
        outline = rounded_rectangle_outline(wall, plan.capsule_radius + plan.margin)
        pts = [origin + rotation @ np.r_[xy, plan.start[2]] / 1000 for xy in outline]
        for k, (p, q) in enumerate(zip(pts[:-1], pts[1:])):
            line(f"wall_inflation_{index}_{k}", p, q, "1 .55 0 .55", .00015)
    for name, p, color in [
        ("start", plan.start, "0 .8 .2 1"),
        ("goal", plan.goal, ".7 0 .9 1"),
    ]:
        ET.SubElement(
            world,
            "site",
            name="path_" + name,
            type="sphere",
            size=".0015",
            pos=" ".join(map(str, origin + rotation @ p / 1000)),
            rgba=color,
            group="2",
        )
    scene_path = Path(scene_path).resolve()
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(scene_path, encoding="utf-8", xml_declaration=True)
    new = MujocoSystem(scene_path)
    if (new.model.nq, new.model.nv, new.model.nu) != (
        old.model.nq,
        old.model.nv,
        old.model.nu,
    ):
        raise RuntimeError("Adding static obstacles changed dynamic DOFs unexpectedly.")
    for key in ("qpos", "qvel", "act", "ctrl"):
        getattr(new.data, key)[:] = getattr(old.data, key)
    new.data.time = old.data.time
    mujoco.mj_forward(new.model, new.data)
    plant.system = new
    plant.oil_geom_id = new.model.geom("oil_volume").id
    return scene_path


def planning_bounds_from_tank(
    plant, origin, rotation, z_mm, body_radius_mm, margin_mm, requested
):
    """Restrict center rectangle to fully submerged bounding sphere in actual oil box."""
    if not np.allclose(np.abs(rotation), np.eye(3), atol=1e-6):
        raise ValueError(
            "This first XY integration requires the baseline vertical DEMA frame."
        )
    oil = plant.system.data.geom_xpos[plant.oil_geom_id]
    half = plant.system.model.geom_size[plant.oil_geom_id]
    reserve = (body_radius_mm + margin_mm) / 1000
    world_z = (origin + rotation @ np.array([0.0, 0.0, z_mm / 1000]))[2]
    if not oil[2] - half[2] + reserve < world_z < oil[2] + half[2] - reserve:
        raise ValueError(
            "Requested Z does not leave the capsule envelope fully inside the oil."
        )
    low = (oil - half + reserve - origin) * 1000
    high = (oil + half - reserve - origin) * 1000
    limits = []
    for axis in (0, 1):
        v = sorted(
            [low[axis] / rotation[axis, axis], high[axis] / rotation[axis, axis]]
        )
        limits.extend(v)
    b = np.asarray(requested, float).copy()
    b[0] = max(b[0], limits[0])
    b[1] = min(b[1], limits[1])
    b[2] = max(b[2], limits[2])
    b[3] = min(b[3], limits[3])
    return b
