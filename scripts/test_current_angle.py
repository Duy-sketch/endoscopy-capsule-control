"""Experimental current-only XZ-angle control, fixed world height, free capsule.
Place in scripts/. Uses the existing installed endoscopy_capsule_control package.
Angle zero = initial DEMA +Z; positive angle tilts toward initial DEMA +X.
This is planar orientation feedback, not full 3D attitude control.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from endoscopy_capsule_control.simulation.follower_options import parse_options
from endoscopy_capsule_control.plant.z_hover_plant import ZHoverPlant
from endoscopy_capsule_control.control.pid_z import ZPIDController, design_z_pid_gains
from endoscopy_capsule_control.simulation.safety import RobotEnvironmentGuard
from endoscopy_capsule_control.simulation.viewer import configure_follower_camera
from endoscopy_capsule_control.magnetic import magnetic_wrench


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def allocate(A, target, limit):
    """Bounded two-current least squares; signed currents, matching the existing power-supply model.
    Scale rows to prioritize vertical force when both objectives are infeasible.
    Enumerating four edges gives the optimum on the 2D current box.
    """
    W = np.array([5/.05, 1/.05])
    M, y = W[:, None]*A, W*target
    raw = np.linalg.lstsq(M, y, rcond=1e-10)[0]
    candidates = [np.clip(raw, -limit, limit)]
    for fixed in range(2):
        other = 1-fixed
        for value in [-limit, limit]:
            q = np.zeros(2)
            q[fixed] = value
            v = M[:, other]
            q[other] = np.clip(v @ (y-M[:, fixed]*value) / max(v@v, 1e-20), -limit, limit)
            candidates.append(q)
    current = min(candidates, key=lambda q: np.linalg.norm(M@q-y))
    feasible = np.linalg.norm(M@current-y) < 1e-5
    return current, feasible


class HeightAngleController:
    def __init__(self, plant, angle_kp=.3):
        self.plant = plant
        self.angle_kp = angle_kp
        self.dt = plant.config.simulation.control_dt
        g = plant.reported_geometry()
        self.R0 = g.lcs_pose.rotation.copy()
        self.o0 = g.lcs_pose.position.copy()
        self.up = -plant.system.model.opt.gravity
        self.up = self.up/np.linalg.norm(self.up)
        # Fixed reference from nominal geometry, NOT a ground-truth feedback signal.
        self.height_ref = self.up @ (self.o0 + self.R0[:, 2]*plant.config.z_control.z_ref)
        self.ff = (plant.config.capsule.mass-plant.config.fluid.density*plant.capsule_volume)*np.linalg.norm(plant.system.model.opt.gravity)
        c = plant.config.z_control
        gains = design_z_pid_gains(plant.config.capsule.mass, plant.drag_coefficient,
                                  c.natural_frequency, c.damping_ratio, c.integral_pole)
        self.pid = ZPIDController(gains, self.dt, c.derivative_filter_tau,
                                 c.integral_limit, c.pid_force_limit)
        self.pid.reset()
        self.limit = min(plant.config.power_supply.current_abs_max,
                         plant.config.electromagnet.current_abs_max)
        self.currents = np.zeros(2)
        self.feasible = True
        self.last = {}
        self.update(0., initialize=True)
        plant.power_supply.reset(initial_current_A=self.currents)
        plant.localization.reset()

    def update(self, reference, initialize=False):
        plant = self.plant
        # All controller feedback comes from noisy/delayed localization.
        measurement = plant.sample_localization()
        g = plant.reported_geometry()
        R = g.lcs_pose.rotation
        p = g.lcs_pose.position + R @ measurement.position_lcs_m
        # Existing sensor angle zero is +X: convert it to zero at local +Z.
        local_phi = wrap(measurement.theta_y_rad + np.pi/2)
        m = R @ np.array([np.sin(local_phi), 0., np.cos(local_phi)]) * plant.magnetic_moment_magnitude
        mr = self.R0.T @ m
        phi = np.arctan2(mr[0], mr[2])
        error = wrap(reference-phi)
        # Field-direction feedforward plus angle-error feedback; no forced alignment.
        field_phi = reference + np.clip(self.angle_kp*error, -.15, .15)
        n = self.R0 @ np.array([np.sin(field_phi), 0., np.cos(field_phi)])
        perpendicular = self.R0 @ np.array([np.cos(field_phi), 0., -np.sin(field_phi)])
        fields, forces = [], []
        for current in ([1., 0.], [0., 1.]):
            force, _, field = magnetic_wrench(p, m,
                [g.em1.position_world, g.em2.position_world],
                [g.em1.axis_world, g.em2.axis_world], current,
                plant.config.electromagnet.magnetic_gain)
            forces.append(force)
            fields.append(field)
        fields, forces = np.column_stack(fields), np.column_stack(forces)
        A = np.vstack([self.up @ forces, perpendicular @ fields])
        height = self.up @ p
        if initialize:
            self.pid.reset(initial_error=self.height_ref-height)
        candidate = self.pid.prepare(self.height_ref, height)
        target = np.array([self.ff+candidate.force_candidate, 0.])
        current, feasible = allocate(A, target, self.limit)
        aligned = n @ (fields @ current) > 1e-8
        unsaturated_force = (self.pid.gains.kp*candidate.error
            + self.pid.gains.ki*candidate.integral_candidate
            + self.pid.gains.kd*candidate.error_rate_filtered)
        correction = self.pid.finalize(feasible and aligned
            and abs(unsaturated_force-candidate.force_candidate) < 1e-10)
        target[0] = self.ff+correction
        current, feasible = allocate(A, target, self.limit)
        self.currents = current
        self.feasible = bool(feasible and n @ (fields @ current) > 1e-8)
        self.last = dict(measured_phi=phi, measured_height=height,
                         force_target=target[0], force_predicted=float(A[0]@current))


def draw(viewer, plant):
    state, g = plant.true_state(), plant.actual_geometry()
    F, _, B = magnetic_wrench(state.position_world, state.magnetic_moment_world,
        [g.em1.position_world,g.em2.position_world], [g.em1.axis_world,g.em2.axis_world],
        plant.power_supply.last_sample.actual_A,plant.config.electromagnet.magnetic_gain)
    with viewer.lock():
        scene=viewer.user_scn
        scene.ngeom=0
        for v,length,color in [(B,.045,[0,1,.1,1]),(state.magnetic_moment_world,.028,[.1,.4,1,1]),(F,.04,[1,.5,0,1])]:
            if np.linalg.norm(v)<1e-12:
                continue
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.0012,state.position_world,state.position_world+length*v/np.linalg.norm(v))
            scene.ngeom+=1


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--angle-deg',type=float,default=3.)
    parser.add_argument('--ramp-s',type=float,default=5.)
    parser.add_argument('--angle-kp',type=float,default=.3)
    extra,remaining=parser.parse_known_args()
    if not np.isfinite([extra.angle_deg,extra.ramp_s,extra.angle_kp]).all() or abs(extra.angle_deg)>45 or extra.ramp_s<=0 or extra.angle_kp<0:
        parser.error('Require |angle-deg| <= 45, ramp-s > 0, angle-kp >= 0')
    opts,cfg,noise=parse_options(['--mode','hold','--x0-mm','0','--time','15',
        '--camera','capsule','--noise','all','--current-tau-ms','10',
        '--localization-delay-ms','10','--csv','results/current_angle.csv',*remaining])
    if opts.mode!='hold' or opts.scenario or opts.open_loop:
        parser.error('This test supports hold only; omit scenario/open-loop')
    plant=ZHoverPlant(config=cfg,noise=noise,seed=opts.seed,vertical=True,
        initial_current_A=(0.,0.),x_offset_m=opts.x0_mm/1000,y_offset_m=opts.y0_mm/1000,
        z_offset_m=opts.z0_mm/1000,theta_y_offset_rad=np.deg2rad(opts.theta0_deg))
    control=HeightAngleController(plant,extra.angle_kp)
    guard=RobotEnvironmentGuard(plant,opts.table_clearance_mm)
    dt=plant.physics_dt
    every=int(round(cfg.simulation.control_dt/dt))
    if not np.isclose(every*dt,cfg.simulation.control_dt):
        raise ValueError('Control period must be a multiple of physics dt')
    rows=[]; reason='duration completed'; last_snapshot=None
    print('Fixed robot joint targets; free capsule. Height feedback in WORLD vertical.')
    print('Angle: initial DEMA +Z toward +X; XZ plane only.')
    print('Noise:',noise.enabled_names(),'seed:',opts.seed)
    print('tau / delay [ms]:',opts.current_tau_ms,opts.localization_delay_ms)
    print('GREEN B / BLUE moment / ORANGE force. Current limits: +/-',control.limit,'A')
    print(' time | target deg | actual deg | Z error mm | XY drift mm | I1 A | I2 A | feasible')

    def reference(t):
        u=np.clip((t-2.)/extra.ramp_s,0.,1.)
        return np.deg2rad(extra.angle_deg)*(3*u*u-2*u*u*u)

    def step(k):
        nonlocal reason,last_snapshot
        if k%every==0:
            clearance,pair=guard.check()
            if clearance<=opts.table_clearance_mm/1000:
                reason='STOP: robot/environment clearance '+pair
                return False
            control.update(reference(plant.time_s))
        last_snapshot=plant.step(control.currents)
        if (k+1)%every==0:
            state=plant.true_state()
            if not np.isfinite(state.position_world).all():
                reason='STOP: nonfinite state';return False
            m=control.R0.T@state.magnetic_moment_world
            phi=np.arctan2(m[0],m[2]); pos=control.R0.T@(state.position_world-control.o0)
            hz=control.up@state.position_world
            row=[plant.time_s,np.rad2deg(reference(plant.time_s)),np.rad2deg(phi),
                 np.rad2deg(control.last['measured_phi']),1000*(hz-control.height_ref),
                 1000*(control.last['measured_height']-control.height_ref),
                 *(1000*pos),*control.currents,*last_snapshot.power_supply.actual_A,
                 float(control.feasible),control.last['force_target'],control.last['force_predicted'],
                 *last_snapshot.external_wrench.magnetic_force_world,
                 float(np.linalg.norm(state.linear_velocity_world)),float(m[1]/np.linalg.norm(m))]
            rows.append(row)
            if len(rows)%max(1,round(1/cfg.simulation.control_dt))==0:
                print(f'{row[0]:5.1f} {row[1]:9.3f} {row[2]:9.3f} {row[4]:10.3f} {np.linalg.norm(pos[:2])*1000:10.3f} {row[9]:7.3f} {row[10]:7.3f} {bool(row[13])}')
            # ROI is diagnostic only: print warning, never freeze solely for ROI exit.
            if not .04<state.position_lcs[2]<.18 or np.linalg.norm(state.position_lcs[:2])>.08:
                if not getattr(step,'warned',False):
                    print('WARNING: left diagnostic ROI; physics continues.')
                    step.warned=True
            shell=plant.system.model.geom('MCE_shell').id
            if any(c.dist<=0 and shell in (c.geom1,c.geom2) for c in plant.system.data.contact[:plant.system.data.ncon]):
                reason='STOP: capsule physical contact (not hovering)';return False
            if np.linalg.norm(state.position_world-control.o0)>1.:
                reason='STOP: capsule escaped 1 m diagnostic domain';return False
        return True

    def run(viewer=None):
        nonlocal reason
        for k in range(int(np.ceil(opts.time/dt))):
            if viewer is not None and not viewer.is_running():
                reason='viewer closed';break
            started=time.perf_counter();keep=step(k)
            if viewer is not None:
                if k%max(1,round(1/(30*dt)))==0 or not keep:
                    draw(viewer,plant);viewer.sync()
                time.sleep(max(0.,dt-(time.perf_counter()-started)))
            if not keep:break

    def save():
        opts.csv.parent.mkdir(parents=True,exist_ok=True)
        names=['time_s','target_deg','true_angle_deg','measured_angle_deg','true_height_error_mm',
            'measured_height_error_mm','fixed_x_mm','fixed_y_mm','fixed_z_mm','command_I1_A','command_I2_A',
            'actual_I1_A','actual_I2_A','allocation_feasible','force_target_N','force_predicted_N',
            'true_Fx_N','true_Fy_N','true_Fz_N','speed_m_s','out_of_plane_m_fraction']
        with opts.csv.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(names);w.writerows(rows)
        summary=dict(status=reason,noise=list(noise.enabled_names()),seed=opts.seed,
            angle_target_deg=extra.angle_deg,height_ref_world_m=control.height_ref,
            current_tau_ms=opts.current_tau_ms,localization_delay_ms=opts.localization_delay_ms,
            current_range_A=[-control.limit, control.limit],
            model_path=str(plant.system.model_path),control='measured planar angle + world-height PID; free XYZ',
            warning='Not hardware calibrated. Angle/Z tracking does not establish XY equilibrium.')
        if rows:
            a=np.array(rows);tail=a[a[:,0]>=max(2+extra.ramp_s,a[-1,0]-3)]
            summary.update(final_angle_deg=float(a[-1,2]),final_height_error_mm=float(a[-1,4]),
                final_xy_mm=a[-1,6:8].tolist(),infeasible_fraction=float(np.mean(a[:,13]<.5)),
                max_out_of_plane_fraction=float(np.max(np.abs(a[:,-1]))))
            if len(tail):
                summary.update(tail_angle_rms_deg=float(np.sqrt(np.mean((tail[:,2]-tail[:,1])**2))),
                    tail_height_rms_mm=float(np.sqrt(np.mean(tail[:,4]**2))))
        opts.csv.with_suffix('.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))
        if opts.plot and rows:
            import matplotlib.pyplot as plt
            a=np.array(rows);fig,axs=plt.subplots(4,1,figsize=(9,10),sharex=True,layout='constrained')
            axs[0].plot(a[:,0],a[:,1],'--',label='target');axs[0].plot(a[:,0],a[:,2],label='true');axs[0].set_ylabel('Angle [deg]');axs[0].legend()
            axs[1].plot(a[:,0],a[:,4]);axs[1].set_ylabel('World Z error [mm]')
            axs[2].plot(a[:,0],a[:,6],label='X');axs[2].plot(a[:,0],a[:,7],label='Y');axs[2].set_ylabel('XY [mm]');axs[2].legend()
            axs[3].plot(a[:,0],a[:,11],label='I1');axs[3].plot(a[:,0],a[:,12],label='I2');axs[3].set_ylabel('Actual current [A]');axs[3].legend();axs[3].set_xlabel('Time [s]')
            for ax in axs:ax.grid(alpha=.3)
            fig.suptitle(reason);fig.savefig(opts.csv.with_suffix('.png'),dpi=150);plt.close(fig)
        print('Saved:',opts.csv)

    try:
        if opts.headless:run()
        else:
            from mujoco import viewer as mj_viewer
            with mj_viewer.launch_passive(plant.system.model,plant.system.data) as viewer:
                configure_follower_camera(viewer,plant.system,opts.camera,guard.table_present)
                run(viewer);draw(viewer,plant);viewer.sync();save()
                print('PHYSICS PAUSED:',reason,'-- frozen display, NOT evidence of hover. Close viewer.')
                while viewer.is_running():viewer.sync();time.sleep(.02)
    finally:save()


if __name__=='__main__':main()
