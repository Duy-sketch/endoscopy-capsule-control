"""Follower CSV schema, diagnostics and PNG export. Never feeds truth into control."""

from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import mujoco
import numpy as np
from endoscopy_capsule_control.magnetic import magnetic_wrench

CSV_HEADER = "time_s,x_lcs_m,y_lcs_m,z_lcs_m,alignment_deg,capsule_world_x_m,capsule_world_y_m,capsule_world_z_m,dema_world_x_m,dema_world_y_m,dema_world_z_m,i1_command_A,i2_command_A,i1_actual_A,i2_actual_A,z_measured_last_m,capsule_ref_world_x_m,capsule_ref_world_y_m,capsule_ref_world_z_m,leader_ref_world_x_m,leader_ref_world_y_m,leader_ref_world_z_m,submerged_fraction,i1_clean_output_A,i2_clean_output_A,capsule_tilt_from_world_vertical_deg,field_tilt_from_world_vertical_deg,B_world_x_T,B_world_y_T,B_world_z_T"


class FollowerResults:
    def __init__(self, options, plant, controller, guard, origin, rotation, target_at):
        self.options = options
        self.plant = plant
        self.controller = controller
        self.guard = guard
        self.origin = origin
        self.rotation = rotation
        self.target_at = target_at
        self.rows = []
        self.status = "running"
        self.noise_sq = np.zeros(2)
        self.noise_count = 0

    def wrench(self, p, m, currents):
        g = self.plant.actual_geometry()
        return magnetic_wrench(
            p,
            m,
            [g.em1.position_world, g.em2.position_world],
            [g.em1.axis_world, g.em2.axis_world],
            currents,
            self.plant.config.electromagnet.magnetic_gain,
        )

    def record(self):
        if self.rows and abs(self.rows[-1][0] - self.plant.time_s) < 1e-12:
            return
        s = self.plant.true_state()
        g = self.plant.actual_geometry()
        actual = self.plant.power_supply.last_sample.actual_A
        _, _, b = self.wrench(s.position_world, s.magnetic_moment_world, actual)
        m = s.magnetic_moment_world
        angle = np.rad2deg(
            np.arccos(np.clip(m @ b / (np.linalg.norm(m) * np.linalg.norm(b)), -1, 1))
        )
        down = -self.plant.system.model.opt.gravity / np.linalg.norm(
            self.plant.system.model.opt.gravity
        )
        capsule_tilt = np.rad2deg(
            np.arccos(np.clip(abs(m @ down) / np.linalg.norm(m), 0, 1))
        )
        field_tilt = np.rad2deg(
            np.arccos(np.clip(abs(b @ down) / np.linalg.norm(b), 0, 1))
        )
        self.rows.append(
            [
                self.plant.time_s,
                *s.position_lcs,
                angle,
                *s.position_world,
                *g.lcs_pose.position,
                *self.controller.currents,
                *actual,
                self.controller.measurement.position_lcs_m[2],
                *self.target_at(self.plant.time_s)
                + self.rotation[:, 2] * self.plant.config.z_control.z_ref,
                *self.target_at(self.plant.time_s),
                self.plant.submerged_volume_state(s)[0],
                *self.plant.power_supply.last_sample.clean_output_A,
                capsule_tilt,
                field_tilt,
                *b,
            ]
        )

    def write_metadata(self, status):
        samples = np.asarray(self.rows)
        metadata = {
            "schema_version": 2,
            "status": status,
            "actual_time_s": self.plant.time_s,
            "arguments": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(self.options).items()
            },
            "config": asdict(self.plant.config),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "mujoco": mujoco.__version__,
            },
            "sha256": {
                "report_module": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "model_xml": hashlib.sha256(
                    self.plant.system.model_path.read_bytes()
                ).hexdigest(),
            },
            "initial_common_current_A": float(self.controller.initial_current),
            "clearance_query_cap_mm": self.guard.clearance_query_m * 1000,
            "robot_fixed_bias_world_m": self.plant.robot_error.translation_world_m.tolist(),
            "model_notes": [
                "Uncalibrated lab model; tank and magnetic parameters require measurement.",
                "Z PID+FF; passive capsule XY and rotation; robot follows prescribed path.",
                "Current source starts prebiased at nominal current; no power-on transient.",
                "Source tau is an effective first-order current response, not a voltage-limited coil model.",
                "Delay is applied to LCS measurements; startup holds the first available sample.",
                "Random-noise switches do not disable deterministic delay or current response.",
                "Clearance values at query cap are lower bounds, not exact distances.",
                "XML hash does not fingerprint external mesh assets.",
            ],
        }
        metadata["source_sha256"] = {
            str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        }
        if len(samples):
            metadata["metrics"] = {
                "max_abs_z_error_mm": float(
                    np.max(np.abs(samples[:, 3] - self.plant.config.z_control.z_ref))
                    * 1000
                ),
                "final_alignment_deg": float(samples[-1, 4]),
                "max_alignment_deg": float(np.max(samples[:, 4])),
                "max_capsule_tilt_from_world_vertical_deg": float(
                    np.max(samples[:, 25])
                ),
                "max_field_tilt_from_world_vertical_deg": float(np.max(samples[:, 26])),
                "minimum_sampled_clearance_mm": (
                    float(min(self.guard.clearance_history) * 1000)
                    if self.guard.clearance_history
                    else None
                ),
                "clean_current_tracking_rms_A": np.sqrt(
                    np.mean((samples[:, 11:13] - samples[:, 23:25]) ** 2, axis=0)
                ).tolist(),
            }
        plan = getattr(self, "plan", None)
        if plan is not None:
            metadata["planning"] = plan.metadata()
            metadata["path_metrics"] = self.path_monitor.metadata(status)
            fixed_errors = (samples[:, 5:8] - samples[:, 16:19]) @ self.rotation * 1000
            metadata["path_metrics"]["xy_rms_mm"] = float(
                np.sqrt(np.mean(np.sum(fixed_errors[:, :2] ** 2, axis=1)))
            )
            metadata["path_metrics"]["fixed_z_rms_mm"] = float(
                np.sqrt(np.mean(fixed_errors[:, 2] ** 2))
            )
        self.options.csv.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def save_plot(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        samples = np.asarray(self.rows)
        t = samples[:, 0]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        xy = (samples[:, 5:8] - self.origin) @ self.rotation * 1000
        ref = (samples[:, 16:19] - self.origin) @ self.rotation * 1000
        ax = axes[0, 0]
        ax.plot(ref[:, 0], ref[:, 1], "--", label="Commanded capsule path")
        ax.plot(xy[:, 0], xy[:, 1], label="True capsule path")
        ax.set(
            xlabel="Initial LCS X [mm]",
            ylabel="Initial LCS Y [mm]",
            title="XY path",
            aspect="equal",
        )
        points = np.vstack([xy[:, :2], ref[:, :2]])
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        half = max(float(np.ptp(points, axis=0).max()) * 0.55, 0.5)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        plan = getattr(self, "plan", None)
        if plan is not None:
            from matplotlib.patches import Circle, Rectangle, Polygon
            from endoscopy_capsule_control.planning.collision import rounded_rectangle_outline

            for obstacle, radius in zip(plan.obstacles, plan.inflated):
                ax.add_patch(Circle(obstacle[:2], radius, color="orange", alpha=0.2))
                ax.add_patch(Circle(obstacle[:2], obstacle[2], color="firebrick"))
            for wall in plan.rectangles:
                x, y, w, h = wall
                ax.add_patch(Polygon(rounded_rectangle_outline(wall, plan.capsule_radius+plan.margin),
                                     color="orange", alpha=.2))
                ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, color="teal"))
            ax.plot(plan.raw[:, 0], plan.raw[:, 1], ":", color="gray", label="Raw A*")
            ax.plot(
                plan.path[:, 0],
                plan.path[:, 1],
                "o--",
                color="green",
                label="Simplified A*",
            )
            ax.scatter(*plan.goal[:2], marker="*", s=100, color="purple", label="Goal")
            ax.set_title("XY path; orange = inflated forbidden region")
            if len(plan.obstacles):
                extents = np.vstack(
                    [
                        plan.obstacles[:, :2] - plan.inflated[:, None],
                        plan.obstacles[:, :2] + plan.inflated[:, None],
                        points,
                    ]
                )
                midpoint = (extents.max(axis=0) + extents.min(axis=0)) / 2
                width = max(float(np.ptp(extents, axis=0).max()) * 0.55, 0.5)
                ax.set_xlim(midpoint[0] - width, midpoint[0] + width)
                ax.set_ylim(midpoint[1] - width, midpoint[1] + width)
            if len(plan.rectangles):
                ax.set_xlim(plan.bounds[0]-20, plan.bounds[1]+20)
                ax.set_ylim(plan.bounds[2]-20, plan.bounds[3]+20)

        ax = axes[0, 1]
        ax.plot(
            t,
            (samples[:, 3] - self.plant.config.z_control.z_ref) * 1000,
            label="True Z error (current LCS)",
        )
        ax.plot(
            t,
            (samples[:, 15] - self.plant.config.z_control.z_ref) * 1000,
            alpha=0.4,
            label="Last measured Z error",
        )
        ax.set(xlabel="Time [s]", ylabel="Error [mm]", title="Z regulation")
        ax = axes[1, 0]
        ax.plot(t, samples[:, 25], label="Capsule vs vertical axis")
        ax.plot(t, samples[:, 26], "--", label="Local B vs vertical axis")
        ax.plot(t, samples[:, 4], label="Capsule vs local B")
        ax.set(
            xlabel="Time [s]",
            ylabel="Angle [deg]",
            title="Tilt and alignment are different",
        )
        ax = axes[1, 1]
        ax.plot(t, samples[:, 11], label="Common current command")
        ax.plot(t, samples[:, 13], alpha=0.65, label="Channel 1 actual")
        ax.plot(t, samples[:, 14], alpha=0.65, label="Channel 2 actual")
        ax.set(xlabel="Time [s]", ylabel="Current [A]", title="Actuation")
        for ax in axes.flat:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"Simulation only | {self.status} | seed {self.options.seed} | noise {'+'.join(self.options.noise)}"
        )
        fig.savefig(self.options.csv.with_suffix(".png"), dpi=160)
        plt.close(fig)

    def save_logs(self):
        self.record()
        np.savetxt(
            self.options.csv, self.rows, delimiter=",", comments="", header=CSV_HEADER
        )
        self.write_metadata(self.status)
        if self.options.plot:
            self.save_plot()

    def summary(self):
        if getattr(self, "path_monitor", None) is not None:
            print(
                "Path result:",
                json.dumps(self.path_monitor.metadata(self.status), indent=2),
            )
        print(self.status, "at", self.plant.time_s, "s")
        print("Final capsule LCS [mm]:", self.plant.true_state().position_lcs * 1000)
        print(
            "Actual DEMA displacement [mm]:",
            (self.plant.reported_geometry().lcs_pose.position - self.origin) * 1000,
        )
        samples = np.asarray(self.rows)
        print("Submerged fraction min/max:", samples[:, 22].min(), samples[:, 22].max())
        print(
            "Tank setup: assumed 300x300 mm interior, oil depth 180 mm; not calibrated hardware."
        )
        print("Max |Z error| [mm]:", np.max(np.abs(samples[:, 3] - 0.1)) * 1000)
        mask = samples[:, 0] >= self.options.settling_time
        if np.any(mask):
            print(
                "Post-settling true Z RMS [mm]:",
                np.sqrt(np.mean((samples[mask, 3] - 0.1) ** 2)) * 1000,
            )
            print(
                "Post-settling radial XY RMS [mm]:",
                np.sqrt(np.mean(np.sum(samples[mask, 1:3] ** 2, axis=1))) * 1000,
            )
        else:
            print("No samples after requested settling time")
        errors_world = samples[:, 5:8] - samples[:, 16:19]
        errors_initial = errors_world @ self.rotation
        moving = (
            (samples[:, 0] > 0)
            & (samples[:, 0] <= self.options.move_time)
            & (self.options.mode in ("follow", "path"))
        )
        for title, selected in [
            ("Whole run", np.ones(len(samples), dtype=bool)),
            ("During motion", moving),
        ]:
            if np.any(selected):
                error = errors_initial[selected]
                print(
                    title + " capsule commanded-path XY RMS [mm]:",
                    np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1))) * 1000,
                )
                print(
                    title + " capsule commanded-path 3D RMS [mm]:",
                    np.sqrt(np.mean(np.sum(error**2, axis=1))) * 1000,
                )
        robot_error = samples[:, 8:11] - samples[:, 19:22]
        print(
            "Leader commanded-path 3D RMS [mm]:",
            np.sqrt(np.mean(np.sum(robot_error**2, axis=1))) * 1000,
        )
        print("Max command current [A]:", np.max(np.abs(samples[:, 11:13])))
        print(
            "Current noise RMS [mA]:",
            np.sqrt(self.noise_sq / max(self.noise_count, 1)) * 1000,
        )
        print(
            "Robot fixed bias XYZ world [mm]:",
            self.plant.robot_error.translation_world_m * 1000,
        )
        print("Final alignment error [deg]:", samples[-1, 4])
        print("Maximum logged alignment error [deg]:", np.max(samples[:, 4]))
        print(
            "Maximum logged capsule tilt from vertical axis [deg]:",
            np.max(samples[:, 25]),
        )
        print(
            "Maximum logged field tilt from vertical axis [deg]:",
            np.max(samples[:, 26]),
        )
        if self.options.plot:
            print("Figure:", self.options.csv.with_suffix(".png"))
        if self.guard.clearance_history:
            print(
                "Minimum sampled robot/environment clearance [mm; capped]:",
                min(self.guard.clearance_history) * 1000,
            )
        print(
            "Clean current tracking RMS [A], channels 1/2:",
            np.sqrt(np.mean((samples[:, 11:13] - samples[:, 23:25]) ** 2, axis=0)),
        )
        print("CSV:", self.options.csv)
        print("Run metadata:", self.options.csv.with_suffix(".json"))
