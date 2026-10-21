"""Analytic point/segment checks against inflated circular obstacles in mm."""

import numpy as np


def segment_safe(bounds, obstacles, inflated, a, b):
    a, b = np.asarray(a), np.asarray(b)
    xmin, xmax, ymin, ymax = bounds
    for p in (a, b):
        if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax):
            return False
    d = b - a
    dd = float(d @ d)
    if not len(obstacles):
        return True
    t = (
        np.clip((obstacles[:, :2] - a) @ d / dd, 0, 1)
        if dd
        else np.zeros(len(obstacles))
    )
    distance = np.linalg.norm(a + t[:, None] * d - obstacles[:, :2], axis=1)
    return bool(np.all(distance > inflated + 1e-8))


def rectangle_signed_distance(point, rectangles):
    """Signed point-to-axis-aligned-box distance; rectangles = cx,cy,w,h."""
    rectangles = np.asarray(rectangles, float).reshape(-1, 4)
    q = np.abs(np.asarray(point) - rectangles[:, :2]) - rectangles[:, 2:] / 2
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def segment_rectangle_distance(a, b, rectangles):
    """Exact unsigned segment-to-rectangle distance, including crossing edges."""
    rects = np.asarray(rectangles, float).reshape(-1, 4)
    a, b = np.asarray(a, float), np.asarray(b, float)
    low, high = rects[:, :2] - rects[:, 2:] / 2, rects[:, :2] + rects[:, 2:] / 2
    d = b - a
    enter, leave = np.zeros(len(rects)), np.ones(len(rects))
    possible = np.ones(len(rects), dtype=bool)
    for k in range(2):
        if abs(d[k]) < 1e-15:
            possible &= (a[k] >= low[:, k]) & (a[k] <= high[:, k])
        else:
            t1, t2 = (low[:, k] - a[k]) / d[k], (high[:, k] - a[k]) / d[k]
            enter = np.maximum(enter, np.minimum(t1, t2))
            leave = np.minimum(leave, np.maximum(t1, t2))
    intersects = possible & (enter <= leave)
    result = np.minimum(np.maximum(rectangle_signed_distance(a, rects), 0),
                        np.maximum(rectangle_signed_distance(b, rects), 0))
    dd = float(d @ d)
    for ix, iy in ((0,0),(0,1),(1,0),(1,1)):
        corner = np.column_stack(((low, high)[ix][:, 0], (low, high)[iy][:, 1]))
        t = np.clip((corner-a) @ d / dd, 0, 1) if dd else np.zeros(len(rects))
        result = np.minimum(result, np.linalg.norm(corner-a-t[:, None]*d, axis=1))
    return np.where(intersects, 0.0, result)


def rounded_rectangle_outline(rectangle, radius, samples=13):
    """Polyline of the exact circular inflation boundary (display only)."""
    x,y,w,h = rectangle
    points = []
    for sx,sy,start in ((1,1,0),(-1,1,90),(-1,-1,180),(1,-1,270)):
        angles = np.deg2rad(np.linspace(start, start+90, samples))
        points.extend(np.column_stack((x+sx*w/2+radius*np.cos(angles),
                                       y+sy*h/2+radius*np.sin(angles))))
    return np.vstack([points, points[0]])
