"""Sampled robot/environment guard, unchanged from the baseline follower.
Values at the query cap are lower bounds. This is not robot self-collision checking.
"""

import mujoco


class RobotEnvironmentGuard:
    def __init__(self, plant, margin_mm):
        self.plant = plant
        self.model = model = plant.system.model
        self.environment = [
            i
            for i in range(model.ngeom)
            if model.geom(i).name == "ground"
            or model.geom(i).name.startswith(("lab_table_", "tank_", "path_obstacle_"))
        ]
        self.moving = [
            i
            for i in range(model.ngeom)
            if model.geom_contype[i] & 2
            and model.body_weldid[model.geom_bodyid[i]] != 0
        ]
        self.clearance_history = []
        self.clearance_query_m = margin_mm / 1000 + 1e-6
        self.table_present = any(
            model.geom(i).name == "lab_table_top" for i in self.environment
        )

    def check(self):
        if not self.table_present or not self.moving:
            return (float("inf"), "")
        closest = (float("inf"), "")
        for i in self.moving:
            for j in self.environment:
                distance = mujoco.mj_geomDistance(
                    self.model,
                    self.plant.system.data,
                    i,
                    j,
                    self.clearance_query_m,
                    None,
                )
                if distance < closest[0]:
                    closest = (
                        distance,
                        self.model.geom(i).name + " / " + self.model.geom(j).name,
                    )
        self.clearance_history.append(closest[0])
        return closest


class CapsulePathMonitor:
    """Conservative XY envelope check at physics ticks; real capsule remains 3D."""

    def __init__(self, plant, plan, settings, origin, rotation):
        self.plant, self.plan, self.settings = plant, plan, settings
        self.origin, self.rotation = origin, rotation
        self.minimum_clearance = float("inf")
        self.margin_violations = 0
        self.goal_since = None
        self.goal_hold = 0.0
        self.previous = None
        self.previous_time = None
        self.speed = 0.0
        self.shell = plant.system.model.geom("MCE_shell").id

    def check(self):
        import numpy as np

        plant, plan = self.plant, self.plan
        state = plant.true_state()
        p = self.rotation.T @ (state.position_world - self.origin) * 1000
        if not np.isfinite(p).all():
            return "non-finite capsule state"
        clearance = plan.body_clearance_mm(p[:2])
        self.minimum_clearance = min(self.minimum_clearance, clearance)
        if clearance <= plan.margin:
            self.margin_violations += 1
        if clearance <= 0:
            return "capsule bounding envelope reached obstacle"
        x0, x1, y0, y1 = plan.bounds
        if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
            return "capsule left allowed XY region"
        oil = plant.system.data.geom_xpos[plant.oil_geom_id]
        half = plant.system.model.geom_size[plant.oil_geom_id]
        if (
            abs(state.position_world[2] - oil[2]) + plan.capsule_radius / 1000
            >= half[2]
        ):
            return "capsule envelope left oil depth"
        for contact in plant.system.data.contact[: plant.system.data.ncon]:
            if contact.dist <= 0 and self.shell in (contact.geom1, contact.geom2):
                return "physical capsule contact"
        t = plant.time_s
        if self.previous is not None and t > self.previous_time:
            self.speed = float(
                np.linalg.norm(p[:2] - self.previous[:2]) / (t - self.previous_time)
            )
        self.previous, self.previous_time = p.copy(), t
        reached = (
            t >= plan.motion_end
            and np.linalg.norm(p[:2] - plan.goal[:2])
            <= self.settings["goal_tolerance_mm"]
            and self.speed <= self.settings["goal_speed_mm_s"]
        )
        if reached:
            if self.goal_since is None:
                self.goal_since = t
            self.goal_hold = t - self.goal_since
        else:
            self.goal_since = None
            self.goal_hold = 0.0
        return None

    def metadata(self, status):
        import numpy as np

        held = self.goal_hold >= self.settings["goal_hold_s"]
        p = (
            self.rotation.T
            @ (self.plant.true_state().position_world - self.origin)
            * 1000
        )
        return dict(
            success=status == "duration completed"
            and held
            and self.margin_violations == 0,
            goal_held=held,
            goal_hold_s=self.goal_hold,
            final_xy_error_mm=float(np.linalg.norm(p[:2] - self.plan.goal[:2])),
            min_obstacle_envelope_clearance_mm=(
                self.minimum_clearance if np.isfinite(self.minimum_clearance) else None
            ),
            margin_violation_physics_steps=self.margin_violations,
            notes="Success evaluates XY goal and conservative obstacle margin; Z quality is reported separately. Sampled checks, not continuous collision certification.",
        )
