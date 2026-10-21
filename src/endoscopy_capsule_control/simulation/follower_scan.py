"""Static axis-wise magnetic-force diagnostic, retained for --mode scan."""

import numpy as np


def run_scan(simulation):
    plant = simulation.plant
    a = simulation.options
    origin, R = simulation.origin, simulation.rotation
    currents = simulation.controller.currents
    effective_mass = simulation.controller.effective_mass
    m = R[:, 2] * plant.magnetic_moment_magnitude
    wrench = simulation.results.wrench
    rows = []
    for axis in range(3):
        for mode in range(2):
            samples = []
            for mm in [-5.0, -1.0, 0.0, 1.0, 5.0]:
                p = np.array([0.0, 0.0, 0.1])
                p[axis] += mm / 1000
                world = origin + R @ p
                _, _, b = wrench(world, m, currents)
                moment = m if mode == 0 else b / np.linalg.norm(b) * np.linalg.norm(m)
                f, _, _ = wrench(world, moment, currents)
                net = R.T @ (f + effective_mass * plant.system.model.opt.gravity)
                rows.append([axis, mode, mm, *net])
                samples.append(net[axis])
            slope = (samples[3] - samples[1]) / 0.002
            print(
                "XYZ"[axis],
                ["vertical", "field-aligned"][mode],
                f"dF/dposition={slope:.6f} N/m",
                "restoring" if slope < 0 else "destabilizing",
            )
    np.savetxt(
        a.csv,
        rows,
        delimiter=",",
        comments="",
        header="axis_0x_1y_2z,moment_0vertical_1aligned,offset_mm,Fnet_x_N,Fnet_y_N,Fnet_z_N",
    )
    print("Axis-wise slopes only; not a full dynamic stability proof. CSV:", a.csv)
    return
