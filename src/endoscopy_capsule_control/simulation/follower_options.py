"""CLI/configuration only. No dynamics or controller logic here."""

import argparse
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from endoscopy_capsule_control.config import DEFAULT_CONFIG
from endoscopy_capsule_control.plant.noise import NoiseSwitches


def parse_options(argv=None):
    parser = argparse.ArgumentParser(
        description="Passive XY follower with common-current Z PID+FF"
    )
    parser.add_argument(
        "--mode", choices=["scan", "hold", "follow", "path"], default="hold"
    )
    parser.add_argument("--axis", choices=["x", "y"], default="x")
    parser.add_argument("--trajectory", choices=["line", "circle"], default="line")
    parser.add_argument(
        "--radius-mm",
        type=float,
        default=5.0,
        help="Circle radius in initial DEMA XY plane",
    )
    for name, default in [
        ("distance-mm", 5.0),
        ("move-time", 5.0),
        ("time", 8.0),
        ("x0-mm", 1.0),
        ("y0-mm", 0.0),
        ("z0-mm", 0.0),
        ("theta0-deg", 0.0),
    ]:
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save trajectory/error/angle PNG beside CSV; requires matplotlib",
    )
    parser.add_argument(
        "--camera", choices=["overview", "tank", "capsule"], default="overview"
    )
    parser.add_argument(
        "--table-clearance-mm",
        type=float,
        default=10.0,
        help="Stop if robot/environment clearance reaches this margin",
    )
    parser.add_argument(
        "--no-leader-feedforward",
        action="store_true",
        help="Disable robot trajectory velocity feedforward for comparison",
    )
    parser.add_argument(
        "--noise",
        nargs="+",
        default=["none"],
        choices=["none", "current", "localization", "robot", "all"],
    )
    parser.add_argument(
        "--current-tau-ms",
        type=float,
        default=DEFAULT_CONFIG.power_supply.response_time_constant_s * 1000,
        help="Assumed effective current-source first-order time constant; not an identified coil L/R model",
    )
    parser.add_argument(
        "--localization-delay-ms",
        type=float,
        default=DEFAULT_CONFIG.localization.delay_samples
        * DEFAULT_CONFIG.simulation.control_dt
        * 1000,
        help="Fixed measurement delay, a multiple of control_dt (baseline 10 ms); independent of noise switches",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--settling-time",
        type=float,
        default=0.0,
        help="Start time for RMS diagnostics",
    )
    parser.add_argument(
        "--open-loop", action="store_true", help="Disable Z PID; fixed currents"
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("results/passive_follower.csv")
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        help="Optional JSON defaults; explicit CLI flags override them",
    )
    parser.add_argument("--target-mm", nargs=2, type=float, help="Path goal X Y [mm]")
    parser.add_argument("--speed-mm-s", type=float, help="Path speed limit")
    parser.set_defaults(path=None)
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.scenario is not None:
        try:
            defaults = json.loads(preliminary.scenario.read_text(encoding="utf-8"))
            actions = {
                action.dest: action
                for action in parser._actions
                if action.dest not in ("help", "scenario")
            }
            if not isinstance(defaults, dict):
                raise ValueError("Scenario must be a JSON object")
            path_settings = defaults.pop("path", None)
            if path_settings is not None and not isinstance(path_settings, dict):
                raise ValueError("path must be an object")
            parser.set_defaults(path=path_settings)
            if set(defaults) - set(actions):
                raise ValueError(
                    "Unknown scenario keys; use argparse names such as move_time, noise, camera"
                )
            for key, value in defaults.items():
                action = actions[key]
                if isinstance(action, argparse._StoreTrueAction):
                    if not isinstance(value, bool):
                        raise ValueError(f"{key} requires a boolean")
                elif isinstance(action.nargs, int):
                    if not isinstance(value, list) or len(value) != action.nargs:
                        raise ValueError(f"Invalid {key} length")
                    defaults[key] = [action.type(v) for v in value]
                elif action.nargs == "+":
                    if not isinstance(value, list) or not value:
                        raise ValueError(f"{key} requires a nonempty list")
                    if action.choices and any(v not in action.choices for v in value):
                        raise ValueError(f"Invalid {key}")
                else:
                    if isinstance(value, (list, dict, bool)) or value is None:
                        raise ValueError(f"Invalid {key}")
                    if action.type:
                        value = action.type(value)
                    if action.choices and value not in action.choices:
                        raise ValueError(f"Invalid {key}")
                    defaults[key] = value
            parser.set_defaults(**defaults)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
    a = parser.parse_args(argv)
    if a.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            parser.error("--plot requires: python -m pip install matplotlib")
    values = [
        a.table_clearance_mm,
        a.radius_mm,
        a.time,
        a.move_time,
        a.distance_mm,
        a.x0_mm,
        a.y0_mm,
        a.z0_mm,
        a.theta0_deg,
    ]
    if (
        not np.isfinite(values).all()
        or a.time <= 0
        or a.move_time <= 0
        or a.radius_mm <= 0
        or a.table_clearance_mm <= 0
    ):
        parser.error(
            "Finite parameters, positive durations and positive circle radius required"
        )
    if (
        a.seed < 0
        or not np.isfinite(a.settling_time)
        or not 0 <= a.settling_time < a.time
    ):
        parser.error("seed must be nonnegative and 0 <= settling-time < time")
    if ("none" in a.noise or "all" in a.noise) and len(a.noise) > 1:
        parser.error("Use none/all alone, or combine individual noise sources")
    if (
        not np.isfinite([a.current_tau_ms, a.localization_delay_ms]).all()
        or min(a.current_tau_ms, a.localization_delay_ms) < 0
    ):
        parser.error(
            "Current tau and localization delay must be finite and nonnegative"
        )
    sample_ms = DEFAULT_CONFIG.simulation.control_dt * 1000
    delay_samples = int(round(a.localization_delay_ms / sample_ms))
    if not np.isclose(
        a.localization_delay_ms, delay_samples * sample_ms, rtol=0, atol=1e-8
    ):
        parser.error(f"localization-delay-ms must be a multiple of {sample_ms:g} ms")
    config = replace(
        DEFAULT_CONFIG,
        power_supply=replace(
            DEFAULT_CONFIG.power_supply,
            response_time_constant_s=a.current_tau_ms / 1000,
        ),
        localization=replace(DEFAULT_CONFIG.localization, delay_samples=delay_samples),
    )
    noise = NoiseSwitches.from_names(a.noise)
    if a.mode == "scan" and noise.enabled_names():
        parser.error(
            "Static scan uses --noise none; use hold/follow for noise experiments"
        )
    if a.mode == "scan" and (a.current_tau_ms or delay_samples):
        parser.error(
            "Static scan has no time dynamics; use hold/follow for latency experiments"
        )
    if a.mode == "path" and a.open_loop:
        parser.error("Path following requires the Z controller; omit --open-loop")
    return a, config, noise
