"""Experimental coordinated robot/current XZ-angle and world-position control.
Place in scripts/. Uses the existing installed endoscopy_capsule_control package.
Angle zero = initial DEMA +Z; positive angle tilts toward initial DEMA +X.
This is planar orientation feedback, not full 3D attitude control.
Requirements: installed project package, numpy, mujoco, scipy; matplotlib for --plot.
The robot translates through its actuators and differential IK. Capsule state is
never prescribed. Robot XY commands by default; --allow-robot-z adds slow Z.
A 40-mm diagnostic radial ROI is an assumption, not the paper's calibrated FMFR.
The allocator is an extension inspired by Fig.9, not a literal implementation
of the paper's gamma-only current scaling. Fast currents allocate X/Z force;
slow bounded optimization changes robot pose to obtain the requested field angle.
Sensor feedback remains the existing planar localization model.

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
    """Paper-inspired force/field allocation, not an exact Eq.17/20 reproduction.

    Slow bounded optimization predicts force AND field at a future robot pose.
    Fast two-current allocation protects X/Z force at the CURRENT measured geometry.
    No qpos reset, weld, prescribed capsule motion, or ground-truth feedback.
    """
    def __init__(self, plant, angle_kp=.3, robot_speed=5., robot_range=40., allow_z=False):
        from dataclasses import replace
        from endoscopy_capsule_control.simulation.aubo_cartesian_controller import AuboCartesianController
        from endoscopy_capsule_control.control.xyz_pid_ff import XYZPIDFFController, design_xyz_pid_gains
        self.plant=plant;self.dt=plant.config.simulation.control_dt
        g=plant.reported_geometry();self.R0=g.lcs_pose.rotation.copy();self.o0=g.lcs_pose.position.copy()
        self.up=-plant.system.model.opt.gravity.copy();self.up/=np.linalg.norm(self.up)
        self.target=self.o0+self.R0[:,2]*plant.config.z_control.z_ref
        self.height_ref=self.up@self.target
        self.ff=(plant.config.capsule.mass-plant.config.fluid.density*plant.capsule_volume)*np.linalg.norm(plant.system.model.opt.gravity)
        c=plant.config.xyz_control
        gains=design_xyz_pid_gains(mass=plant.config.capsule.mass,drag_coefficient=plant.drag_coefficient,
            natural_frequency_rad_s=(1.,1.,5.),damping_ratio=(1.,1.,1.),integral_pole_per_s=(.2,.2,1.))
        self.pid=XYZPIDFFController(target_position_world_m=self.target,feedforward_force_world_N=self.up*self.ff,
            gains=gains,dt=self.dt,derivative_filter_tau_s=.08,
            integral_limit_m_s=c.integral_limit_m_s,pid_force_limit_N=c.pid_force_limit_N)
        self.pid.reset(initial_position_world_m=self.target)
        self.limit=min(plant.config.power_supply.current_abs_max,plant.config.electromagnet.current_abs_max)
        self.robot=AuboCartesianController(system=plant.system,config=replace(plant.config.aubo_motion,
            max_linear_speed_mps=robot_speed/1000,position_kp_per_s=4.))
        self.robot.reset();self.allow_z=allow_z;self.robot_range=robot_range/1000
        self.axes=np.eye(3) if allow_z else np.column_stack([self.R0[:,0],self.R0[:,1]])
        self.nd=self.axes.shape[1]
        self.goal=self.o0.copy();self.base=np.array([4.22795935,4.22795935])
        self.currents=self.base.copy();self.feasible=True;self.last={};self.counter=0
        self.opt_error=0.;self.solver_ok=True;self.angle_kp=angle_kp
        self.update(0.)
        plant.power_supply.reset(initial_current_A=self.currents);plant.localization.reset()

    def update(self, reference, initialize=False):
        from scipy.optimize import least_squares
        plant=self.plant;g=plant.reported_geometry();R=g.lcs_pose.rotation
        measurement=plant.sample_localization()
        p=g.lcs_pose.position+R@measurement.position_lcs_m
        phi_local=wrap(measurement.theta_y_rad+np.pi/2)
        m=R@np.array([np.sin(phi_local),0.,np.cos(phi_local)])*plant.magnetic_moment_magnitude
        mr=self.R0.T@m;phi=np.arctan2(mr[0],mr[2])
        # Angle correction belongs to the slow force/field optimizer, not an
        # instantaneous current-ratio jump ahead of the moving robot.
        field_phi=reference+np.clip(self.angle_kp*wrap(reference-phi),-.05,.05)
        n=self.R0@np.array([np.sin(field_phi),0.,np.cos(field_phi)])
        perp=self.R0@np.array([np.cos(field_phi),0.,-np.sin(field_phi)])
        mslow=n*plant.magnetic_moment_magnitude
        origins=np.array([g.em1.position_world,g.em2.position_world])
        axes=[g.em1.axis_world,g.em2.axis_world]
        def wrench(moment, currents, displacement):
            return magnetic_wrench(p,moment,origins+displacement,axes,currents,
                plant.config.electromagnet.magnetic_gain,
                gradient_step=plant.config.electromagnet.magnetic_gradient_step)
        candidate=self.pid.prepare(measured_position_world_m=p)
        force=candidate.desired_force_candidate_world_N
        # Accept integral only if instantaneous force realization is close.
        previous_force=wrench(m,self.currents,np.zeros(3))[0]
        accept=np.abs(force-previous_force)<np.array([.0005,.0005,.001])
        force=self.pid.finalize(accept_integral_axes=accept).desired_force_world_N
        if self.counter%5==0:
            # Normalize pose to 10 mm and currents to 10 A.
            prev=np.r_[self.axes.T@(self.goal-self.o0)/.01,self.base/10.]
            bound=self.robot_range/.01
            lo=np.r_[np.full(self.nd,-bound),[-self.limit/10]*2]
            hi=-lo
            def residual(v):
                goal=self.o0+self.axes@(v[:self.nd]*.01)
                shift=goal-g.lcs_pose.position
                F,_,B=wrench(mslow,v[self.nd:]*10.,shift)
                physical=np.r_[(F-force)/.001, (perp@B)/.01,
                    (self.R0[:,1]@B)/.01, max(0.,.005-n@B)/.01]
                # Weak regularization selects a continuous local solution.
                return np.r_[physical,.002*(v-prev)]
            def jacobian(v):
                h=1e-3
                return np.column_stack([(residual(v+np.eye(len(v))[j]*h)-residual(v-np.eye(len(v))[j]*h))/(2*h) for j in range(len(v))])
            solution=least_squares(residual,np.clip(prev,lo+1e-9,hi-1e-9),bounds=(lo,hi),
                jac=jacobian,max_nfev=35,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            self.solver_ok=bool(solution.success and np.isfinite(solution.x).all())
            if np.isfinite(solution.x).all():
                self.goal=self.o0+self.axes@(solution.x[:self.nd]*.01)
                self.base=solution.x[self.nd:]*10.
            self.opt_error=float(np.linalg.norm(residual(solution.x)[:5]))
        self.robot.command_position(target_position_world_m=self.goal,dt=self.dt)
        # Current allocation compensates X/Z force at the real pose NOW. Future
        # robot displacement never counts as force already applied.
        columns=np.column_stack([wrench(m,i,np.zeros(3))[0] for i in ([1.,0.],[0.,1.])])
        A=np.vstack([self.up@columns,self.R0[:,0]@columns])
        requested=np.array([self.up@force,self.R0[:,0]@force])
        self.currents, feasible=allocate(A,requested,self.limit)
        actual_F,_,actual_B=wrench(m,self.currents,np.zeros(3))
        self.feasible=bool(feasible and n@actual_B>0.)
        self.last=dict(measured_phi=phi,measured_height=float(self.up@p),
            force_target=float(self.up@force),force_predicted=float(self.up@actual_F))
        self.counter+=1

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
    parser.add_argument('--ramp-s',type=float,default=20.)
    parser.add_argument('--robot-speed-mm-s',type=float,default=5.)
    parser.add_argument('--robot-range-mm',type=float,default=40.)
    parser.add_argument('--roi-mm',type=float,default=40.)
    parser.add_argument('--allow-robot-z',action='store_true')
    parser.add_argument('--angle-kp',type=float,default=.3)
    extra,remaining=parser.parse_known_args()
    if not np.isfinite([extra.angle_deg,extra.ramp_s,extra.angle_kp]).all() or abs(extra.angle_deg)>45 or extra.ramp_s<=0 or extra.angle_kp<0:
        parser.error('Require |angle-deg| <= 45, ramp-s > 0, angle-kp >= 0')
    if not np.isfinite([extra.robot_speed_mm_s,extra.robot_range_mm,extra.roi_mm]).all() or min(extra.robot_speed_mm_s,extra.robot_range_mm,extra.roi_mm)<=0:
        parser.error('robot-speed-mm-s, robot-range-mm and roi-mm must be positive and finite')
    opts,cfg,noise=parse_options(['--mode','hold','--x0-mm','0','--time','35',
        '--camera','capsule','--noise','all','--current-tau-ms','10',
        '--localization-delay-ms','10','--csv','results/coordinated_angle.csv',*remaining])
    if opts.time<=0 or not np.isfinite(opts.time):
        parser.error('time must be finite and positive')
    if opts.mode!='hold' or opts.scenario or opts.open_loop:
        parser.error('This test supports hold only; omit scenario/open-loop')
    plant=ZHoverPlant(config=cfg,noise=noise,seed=opts.seed,vertical=True,
        initial_current_A=(0.,0.),x_offset_m=opts.x0_mm/1000,y_offset_m=opts.y0_mm/1000,
        z_offset_m=opts.z0_mm/1000,theta_y_offset_rad=np.deg2rad(opts.theta0_deg))
    control=HeightAngleController(plant,extra.angle_kp,extra.robot_speed_mm_s,extra.robot_range_mm,extra.allow_robot_z)
    guard=RobotEnvironmentGuard(plant,opts.table_clearance_mm)
    dt=plant.physics_dt
    every=int(round(cfg.simulation.control_dt/dt))
    if not np.isclose(every*dt,cfg.simulation.control_dt):
        raise ValueError('Control period must be a multiple of physics dt')
    rows=[]; reason='duration completed'; last_snapshot=None
    print('Coordinated robot/current allocation. Free capsule; fixed WORLD position target.')
    print('Robot XYZ enabled:',extra.allow_robot_z,'speed limit mm/s:',extra.robot_speed_mm_s)
    print('Diagnostic ROI radius mm:',extra.roi_mm,'not paper-calibrated.')
    print('Dependencies: numpy, mujoco, scipy; matplotlib for --plot. Simulated actuator commands only.')
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
                 float(np.linalg.norm(state.linear_velocity_world)),float(m[1]/np.linalg.norm(m)),
                 *(1000*(plant.reported_geometry().lcs_pose.position-control.o0)),
                 *(1000*(control.goal-control.o0)),control.opt_error,float(control.solver_ok),
                 float(np.linalg.norm(state.position_lcs[:2])*1000),plant.last_submerged_fraction]
            rows.append(row)
            if len(rows)%max(1,round(1/cfg.simulation.control_dt))==0:
                print(f'{row[0]:5.1f} {row[1]:9.3f} {row[2]:9.3f} {row[4]:10.3f} {np.linalg.norm(pos[:2])*1000:10.3f} {row[9]:7.3f} {row[10]:7.3f} {bool(row[13])}')
            if not .04<state.position_lcs[2]<.18 or np.linalg.norm(state.position_lcs[:2])*1000>extra.roi_mm:
                reason='STOP: diagnostic ROI exceeded';return False
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
            'true_Fx_N','true_Fy_N','true_Fz_N','speed_m_s','out_of_plane_m_fraction',
            'robot_dx_mm','robot_dy_mm','robot_dz_mm','robot_target_dx_mm','robot_target_dy_mm','robot_target_dz_mm',
            'optimizer_residual_scaled','solver_success','radial_lcs_mm','submerged_fraction']
        with opts.csv.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(names);w.writerows(rows)
        summary=dict(status=reason,noise=list(noise.enabled_names()),seed=opts.seed,
            angle_target_deg=extra.angle_deg,height_ref_world_m=control.height_ref,
            current_tau_ms=opts.current_tau_ms,localization_delay_ms=opts.localization_delay_ms,
            current_range_A=[-control.limit, control.limit],
            model_path=str(plant.system.model_path),control='world XYZ PID+FF; slow force/field optimization + fast XZ current allocation',
            ramp_s=extra.ramp_s,angle_kp=extra.angle_kp,robot_range_mm=extra.robot_range_mm,
            robot_z_enabled=extra.allow_robot_z,robot_speed_mm_s=extra.robot_speed_mm_s,diagnostic_roi_mm=extra.roi_mm,
            warning='Paper-inspired, not exact reproduction. ROI and actuator limits are simulation assumptions. No global stability guarantee.')
        if rows:
            a=np.array(rows);tail=a[a[:,0]>=max(2+extra.ramp_s,a[-1,0]-3)]
            summary.update(final_angle_deg=float(a[-1,2]),final_height_error_mm=float(a[-1,4]),
                final_xy_mm=a[-1,6:8].tolist(),infeasible_fraction=float(np.mean(a[:,13]<.5)),
                max_out_of_plane_fraction=float(np.max(np.abs(a[:,20]))),
                max_radial_lcs_mm=float(np.max(a[:,29])),min_submerged_fraction=float(np.min(a[:,30])),
                final_robot_displacement_mm=a[-1,21:24].tolist(),
                max_abs_current_A=float(np.max(np.abs(a[:,11:13]))),
                max_world_xy_error_mm=float(np.max(np.linalg.norm(a[:,6:8],axis=1))),
                max_abs_height_error_mm=float(np.max(np.abs(a[:,4]))))
            if len(tail):
                summary.update(tail_angle_rms_deg=float(np.sqrt(np.mean((tail[:,2]-tail[:,1])**2))),
                    tail_height_rms_mm=float(np.sqrt(np.mean(tail[:,4]**2))),
                    tail_xy_rms_mm=float(np.sqrt(np.mean(np.sum(tail[:,6:8]**2,axis=1)))))
        opts.csv.with_suffix('.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))
        if opts.plot and rows:
            import matplotlib.pyplot as plt
            a=np.array(rows);fig,axs=plt.subplots(6,1,figsize=(9,14),sharex=True,layout='constrained')
            axs[0].plot(a[:,0],a[:,1],'--',label='target');axs[0].plot(a[:,0],a[:,2],label='true');axs[0].set_ylabel('Angle [deg]');axs[0].legend()
            axs[1].plot(a[:,0],a[:,4]);axs[1].set_ylabel('World Z error [mm]')
            axs[2].plot(a[:,0],a[:,6],label='X');axs[2].plot(a[:,0],a[:,7],label='Y');axs[2].set_ylabel('XY [mm]');axs[2].legend()
            axs[3].plot(a[:,0],a[:,11],label='I1');axs[3].plot(a[:,0],a[:,12],label='I2');axs[3].set_ylabel('Actual current [A]');axs[3].legend();axs[3].set_xlabel('Time [s]')
            for j in range(3):axs[4].plot(a[:,0],a[:,21+j],label='robot '+ 'XYZ'[j])
            axs[4].set_ylabel('Robot delta [mm]');axs[4].legend();axs[4].set_xlabel('Time [s]')
            axs[5].plot(a[:,0],a[:,29],label='capsule radial distance in moving LCS')
            axs[5].axhline(extra.roi_mm,ls='--',color='red',label='diagnostic ROI boundary')
            axs[5].set_ylabel('LCS radius [mm]');axs[5].set_xlabel('Time [s]');axs[5].legend()
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
