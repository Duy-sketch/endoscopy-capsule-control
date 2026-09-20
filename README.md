# endoscopy-capsule-control

MuJoCo simulation of a magnetic capsule endoscope actuated by a dual-electromagnet actuator (DEMA) mounted on an **AUBO i10**.

The current development scope is intentionally narrow:

1. build a clean physical plant,
2. add three independently switchable uncertainty sources,
3. validate the plant,
4. test **Z hovering only**,
5. return to XYZ control later.

## Plant data flow

```text
current command
    -> equivalent dual-channel power supply
    -> actual coil currents
    -> magnetic field / force / torque
    -> fluid + MuJoCo rigid-body dynamics
    -> true capsule state
    -> RF localization
    -> measured capsule state
    -> Z controller
```

AUBO uncertainty is applied between reported robot kinematics and the physical DEMA pose used by the magnetic plant.

The controller must not read MuJoCo `qpos` directly. `qpos` is ground truth for simulation diagnostics only.

## Uncertainty models

### 1. Current-source noise

The DEMA paper states that the two electromagnets are energized by **two independent power units** and that the coils are designed for a maximum current of **20 A**, but it does not identify the power supplies.

The baseline simulation therefore uses a **Kepco BOP 20-20 equivalent**:

- range: ±20 A,
- current-mode ripple/noise: 0.03% of full-scale RMS,
- baseline current-noise RMS: `0.0003 * 20 A = 0.006 A = 6 mA`,
- two channels are sampled independently.

No artificial coil L/R time constant is enabled by default because the paper does not provide enough electrical parameters to identify one.

### 2. RF localization noise

The paper reports dynamic position-localization results of approximately:

- trial 1: `1.70 ± 0.74 mm`,
- trial 2: `1.52 ± 0.72 mm`.

The baseline uses their mean 3-D 2-norm RMSE, `1.61 mm`. Under an explicit isotropic independent-axis Gaussian assumption:

```text
sigma_axis = 1.61 mm / sqrt(3) = 0.9295 mm
```

The paper states that the software runs at 100 Hz. It does not identify a fixed end-to-end localization delay, so the baseline delay is zero samples.

### 3. AUBO i10 pose uncertainty

The current AUBO i10 model corresponds to the older ±175° joint-range generation. The matching AUBO i10 specification gives pose repeatability of approximately **±0.05 mm**.

For the stationary Z-hover experiment this is modeled as a **fixed Cartesian DEMA translation bias sampled once per episode**, not as 1-kHz white jitter.

The simulation uses `sigma = 0.05 mm / 3` and clips the total translation-bias magnitude to `0.05 mm`. This Gaussian interpretation is a simulation assumption; the manufacturer specifies repeatability, not a probability distribution.

## Install

```bash
pip install -e .
```

## 1. Test the three uncertainty models first

```bash
python scripts/test_plant.py
```

Expected values are approximately:

```text
current noise RMS       ~ 6 mA/channel
localization 3-D RMSE   ~ 1.61 mm
AUBO bias norm           <= 0.05 mm
```

Run one component only:

```bash
python scripts/test_plant.py --component current
python scripts/test_plant.py --component localization
python scripts/test_plant.py --component robot
```

## 2. Verify magnetic Z sign convention

```bash
python scripts/test_magnetic_z.py
```

This evaluates several `(I1, I2)` combinations and prints force and torque in the DEMA local frame. Do this before tuning any controller.

## 3. Test the plant open loop

Ideal plant:

```bash
python scripts/test_z_plant_open_loop.py --noise none
```

One uncertainty at a time:

```bash
python scripts/test_z_plant_open_loop.py --noise current
python scripts/test_z_plant_open_loop.py --noise localization
python scripts/test_z_plant_open_loop.py --noise robot
```

All uncertainties:

```bash
python scripts/test_z_plant_open_loop.py --noise all --seed 42
```

## 4. Run Z hovering

Ideal baseline first:

```bash
python scripts/run_z_hover.py --noise none --time 10 --settling-time 5
```

Then isolate each uncertainty:

```bash
python scripts/run_z_hover.py --noise current --seed 42
python scripts/run_z_hover.py --noise localization --seed 42
python scripts/run_z_hover.py --noise robot --seed 42
```

Combine selected sources:

```bash
python scripts/run_z_hover.py --noise current localization --seed 42
```

Full plant uncertainty:

```bash
python scripts/run_z_hover.py --noise all --seed 42
```

Viewer:

```bash
python scripts/run_z_hover.py --noise all --seed 42 --viewer --camera overview
```

## Project structure

```text
scripts/
    run_z_hover.py
    test_plant.py
    test_magnetic_z.py
    test_z_plant_open_loop.py

src/endoscopy_capsule_control/
    config.py

    plant/
        noise.py
        power_supply.py
        localization.py
        robot_uncertainty.py
        z_hover_plant.py

    control/
        pid_z.py
        allocator.py
        z_hover_controller.py

    dynamics/
        fluid.py

    magnetic/
        capsule_magnet.py
        dipole.py
        wrench.py

    simulation/
        mujoco_system.py
        initialization.py
        capsule_state.py
        capsule_wrench.py
        dema_geometry.py
        perturbation.py
        viewer.py
        aubo_ik.py
        z_hover_simulation.py

    models/
        aubo_i10.xml
```

## Important model limitations

- The current `magnetic_gain` is a simulation calibration value, not a paper-identified hardware parameter.
- The capsule magnetic moment magnitude is not numerically specified by the DEMA paper; the project currently uses a test value.
- The present capsule mass remains 7.0 g by project choice, while the paper mentions approximately 7.6 g.
- AUBO joint/servo dynamics are not yet modeled as hardware dynamics. For the stationary Z-hover experiment only the DEMA pose uncertainty is included.
- XYZ/AUBO motion control is intentionally not part of the active pipeline yet.
