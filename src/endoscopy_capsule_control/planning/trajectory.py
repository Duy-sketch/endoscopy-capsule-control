"""Leader line/circle references; these functions do not run physics."""

import numpy as np


def leader_offset(t, duration, trajectory, axis, distance_m, radius_m):
    """One smooth line or closed XY circle; returns to rest after duration.

    The circle starts at the current leader position, has center (-r,0)
    in the initial DEMA XY plane and completes one revolution.
    """
    u = float(np.clip(t / duration, 0.0, 1.0))
    blend = 3 * u * u - 2 * u * u * u
    if trajectory == "circle":
        phase = 2 * np.pi * blend
        return radius_m * np.array([np.cos(phase) - 1.0, np.sin(phase), 0.0])
    result = np.zeros(3)
    result["xy".index(axis)] = distance_m * blend
    return result


def leader_velocity(t, duration, trajectory, axis, distance_m, radius_m):
    """Analytic derivative of leader_offset, in initial LCS coordinates."""
    if t <= 0 or t >= duration:
        return np.zeros(3)
    u = t / duration
    rate = 6 * u * (1 - u) / duration
    if trajectory == "circle":
        phase = 2 * np.pi * (3 * u * u - 2 * u * u * u)
        return (
            radius_m * 2 * np.pi * rate * np.array([-np.sin(phase), np.cos(phase), 0.0])
        )
    result = np.zeros(3)
    result["xy".index(axis)] = distance_m * rate
    return result


def sample_quintic(plan, t):
    if t <= plan.warmup or len(plan.path) == 1:
        return plan.start / 1000, np.zeros(3), np.zeros(3)
    if t >= plan.motion_end:
        return plan.goal / 1000, np.zeros(3), np.zeros(3)
    i = int(np.searchsorted(plan.ends, t, side="right"))
    begin = plan.warmup if i == 0 else plan.ends[i - 1]
    T = plan.durations[i]
    u = (t - begin) / T
    a = np.r_[plan.path[i], plan.start[2]] / 1000
    d = np.r_[plan.path[i + 1] - plan.path[i], 0.0] / 1000
    return (
        a + (10 * u**3 - 15 * u**4 + 6 * u**5) * d,
        (30 * u**2 - 60 * u**3 + 30 * u**4) / T * d,
        (60 * u - 180 * u**2 + 120 * u**3) / T**2 * d,
    )
