"""Static XY A*, exact segment-circle checks and stop-at-waypoint quintics.

Public planning coordinates are millimetres in the initial DEMA frame,
fixed in world. MuJoCo/controller reference outputs are metres.
Obstacle tuples are (x_mm, y_mm, physical_radius_mm).
"""

from .collision import segment_safe, segment_rectangle_distance, rectangle_signed_distance
from .trajectory import sample_quintic
import heapq
import math
import numpy as np


class XYPlan:
    def __init__(
        self,
        start,
        goal,
        obstacles,
        bounds,
        step,
        capsule_radius,
        margin,
        speed,
        acceleration,
        control_dt,
        warmup=2.0,
        rectangles=(),
    ):
        self.start = np.asarray(start, float)
        self.goal = np.asarray(goal, float)
        self.obstacles = np.asarray(obstacles, float).reshape(-1, 3)
        self.rectangles = np.asarray(rectangles, float).reshape(-1, 4)
        if not np.isfinite(self.rectangles).all() or np.any(self.rectangles[:, 2:] <= 0):
            raise ValueError("Walls require finite cx,cy,width,height and positive dimensions.")
        self.bounds = np.asarray(bounds, float)
        self.capsule_radius = float(capsule_radius)
        self.margin = float(margin)
        self.warmup = float(warmup)
        values = [
            *self.start,
            *self.goal,
            *self.bounds,
            step,
            capsule_radius,
            margin,
            speed,
            acceleration,
            control_dt,
            warmup,
            *self.obstacles.ravel(),
        ]
        if not np.isfinite(values).all():
            raise ValueError("All planning inputs must be finite.")
        if (
            min(step, capsule_radius, speed, acceleration, control_dt) <= 0
            or min(margin, warmup) < 0
        ):
            raise ValueError(
                "Positive resolution/radius/speed/acceleration/dt required; margin/warmup >= 0."
            )
        if np.any(self.obstacles[:, 2] <= 0):
            raise ValueError("Obstacle radii must be positive.")
        if not np.isclose(self.start[2], self.goal[2], atol=1e-8, rtol=0):
            raise ValueError(
                "First integration is XY at fixed capsule Z: start Z must equal target Z."
            )
        xmin, xmax, ymin, ymax = self.bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Bounds must be xmin < xmax, ymin < ymax.")
        self.inflated = self.obstacles[:, 2] + capsule_radius + margin
        self.xs = xmin + np.arange(int(np.floor((xmax - xmin) / step)) + 1) * step
        self.ys = ymin + np.arange(int(np.floor((ymax - ymin) / step)) + 1) * step
        if len(self.xs) * len(self.ys) > 200000:
            raise ValueError("Grid too large (>200000 nodes); increase --grid-mm.")
        for label, p in [("start", self.start), ("target", self.goal)]:
            if not self.segment_safe(p[:2], p[:2]):
                raise ValueError(
                    f"{label} is outside center bounds or inside an inflated obstacle."
                )
        self.raw = self._astar()
        if self.raw is None:
            raise ValueError(
                "No collision-free path found on this grid. No simulation started."
            )
        self.path = self._simplify(self.raw)
        lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        durations = np.maximum(
            1.875 * lengths / speed,
            np.sqrt((10 * math.sqrt(3) / 3) * lengths / acceleration),
        )
        self.durations = np.ceil(durations / control_dt) * control_dt
        self.ends = self.warmup + np.cumsum(self.durations)
        self.motion_end = float(self.ends[-1]) if len(self.ends) else self.warmup
        self.length_mm = float(lengths.sum())
        self.peak_speed_mm_s = (
            float(np.max(1.875 * lengths / self.durations)) if len(lengths) else 0.0
        )

    def segment_safe(self, a, b):
        if not segment_safe(self.bounds, self.obstacles, self.inflated, a, b):
            return False
        return bool(np.all(segment_rectangle_distance(a, b, self.rectangles)
                           > self.capsule_radius + self.margin + 1e-8))

    def _astar(self):
        start, goal = self.start[:2], self.goal[:2]
        if np.linalg.norm(goal - start) < 1e-9:
            return np.array([start])
        # Exact start/goal remain graph vertices; never silently snap them.
        if self.segment_safe(start, goal):
            return np.array([start, goal])

        def point(node):
            if node == (-1, -1):
                return goal
            return np.array([self.xs[node[0]], self.ys[node[1]]])

        nodes = [(i, j) for i in range(len(self.xs)) for j in range(len(self.ys))]
        starts = sorted(nodes, key=lambda n: np.linalg.norm(point(n) - start))
        goals = sorted(nodes, key=lambda n: np.linalg.norm(point(n) - goal))

        def connectors(p, candidates):
            result = {}
            for node in candidates:
                q = point(node)
                if self.segment_safe(p, q):
                    result[node] = float(np.linalg.norm(q - p))
                    if len(result) == 12:
                        break
            return result

        gs = connectors(start, starts)
        goal_links = connectors(goal, goals)
        parent = {n: None for n in gs}
        queue = [
            (g + float(np.linalg.norm(point(n) - goal)), g, n) for n, g in gs.items()
        ]
        heapq.heapify(queue)
        moves = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost > gs[node] + 1e-10:
                continue
            if node == (-1, -1):
                rev = [goal]
                node = parent[node]
                while node is not None:
                    rev.append(point(node))
                    node = parent[node]
                rev.append(start)
                result = np.array(rev[::-1])
                keep = np.r_[
                    True, np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-9
                ]
                return result[keep]
            candidates = []
            for dx, dy in moves:
                nxt = (node[0] + dx, node[1] + dy)
                if 0 <= nxt[0] < len(self.xs) and 0 <= nxt[1] < len(self.ys):
                    if self.segment_safe(point(node), point(nxt)):
                        candidates.append(
                            (nxt, float(np.linalg.norm(point(nxt) - point(node))))
                        )
            if node in goal_links:
                candidates.append(((-1, -1), goal_links[node]))
            for nxt, edge in candidates:
                new = cost + edge
                if new < gs.get(nxt, float("inf")) - 1e-10:
                    gs[nxt] = new
                    parent[nxt] = node
                    heapq.heappush(
                        queue,
                        (new + float(np.linalg.norm(point(nxt) - goal)), new, nxt),
                    )
        return None

    def _simplify(self, raw):
        result = [raw[0]]
        i = 0
        while i < len(raw) - 1:
            j = len(raw) - 1
            while j > i and not self.segment_safe(raw[i], raw[j]):
                j -= 1
            if i == j:
                raise ValueError("Unsafe A* edge.")
            result.append(raw[j])
            i = j
        return np.array(result)

    def reference(self, t):
        return sample_quintic(self, t)

    def body_clearance_mm(self, xy):
        distances = list(np.linalg.norm(self.obstacles[:, :2] - xy, axis=1)
                         - self.obstacles[:, 2] - self.capsule_radius)
        distances.extend(rectangle_signed_distance(xy, self.rectangles) - self.capsule_radius)
        return float(min(distances, default=float("inf")))

    def metadata(self):
        return dict(
            frame="initial nominal DEMA frame fixed in world; coordinates mm",
            obstacles_x_y_radius_mm=self.obstacles.tolist(),
            walls_cx_cy_width_height_mm=self.rectangles.tolist(),
            inflated_radius_mm=self.inflated.tolist(),
            capsule_bounding_sphere_radius_mm=self.capsule_radius,
            safety_margin_mm=self.margin,
            center_bounds_mm=self.bounds.tolist(),
            raw_path_xy_mm=self.raw.tolist(),
            path_xy_mm=self.path.tolist(),
            length_mm=self.length_mm,
            segment_durations_s=self.durations.tolist(),
            warmup_s=self.warmup,
            motion_end_s=self.motion_end,
            peak_speed_mm_s=self.peak_speed_mm_s,
            method="8-neighbor A* with exact endpoint connectors; analytic segment-circle and rounded-rectangle clearance; greedy shortcuts; segmentwise quintic",
        )
