"""Headless physics checks for the vertical operating point and passive torque."""
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import unittest
from endoscopy_capsule_control.config import DEFAULT_CONFIG
from endoscopy_capsule_control.simulation.z_hover_simulation import ZHoverSimulation
from endoscopy_capsule_control.magnetic import magnetic_wrench


def simulation(tilt=0., dt=.001):
    config = replace(DEFAULT_CONFIG, simulation=replace(DEFAULT_CONFIG.simulation, physics_dt=dt))
    from endoscopy_capsule_control.simulation.mujoco_system import MujocoSystem
    def system():
        result = MujocoSystem()
        result.model.opt.timestep = dt
        return result
    with patch('endoscopy_capsule_control.plant.z_hover_plant.MujocoSystem', system):
        return ZHoverSimulation(config=config, theta_y_offset_rad=np.deg2rad(tilt),
                                simulation_time_s=2., settling_time_s=1.)


class VerticalAlignmentTests(unittest.TestCase):
    def test_vertical_equilibrium_and_restoring_torque(self):
        sim = simulation()
        state = sim.plant.true_state()
        geo = sim.plant.actual_geometry()
        axis = geo.lcs_pose.rotation[:, 2]
        assert np.dot(state.magnetic_moment_world / .05, axis) > 1 - 1e-12
        assert np.isclose(sim.current_command_A[0], sim.current_command_A[1], rtol=1e-8)
        assert np.all(sim.current_command_A > 0)
        force, torque, field = magnetic_wrench(
            state.position_world, state.magnetic_moment_world,
            [geo.em1.position_world, geo.em2.position_world],
            [geo.em1.axis_world, geo.em2.axis_world], sim.current_command_A,
            sim.config.electromagnet.magnetic_gain,
        )
        assert field @ axis > 0
        assert np.linalg.norm(torque) < 1e-12
        assert abs(force @ axis - sim.controller.feedforward_force_z) < 1e-10
        assert sim.current_command_A[0] < 20
    
    
    def test_free_capsule_aligns_without_angle_pid(self):
        for tilt in [-15., 15.]:
            sim = simulation(tilt)
            sim.run()
            samples = np.asarray(sim.alignment_log)
            assert abs(samples[0, 4] - abs(tilt)) < 1e-8
            assert samples[-1, 4] < .2
            assert samples[-1, 5] < .2
            assert abs(samples[-1, 3] - .1) < .0002
            assert np.isfinite(samples).all()
    
    
    def test_removing_magnetic_torque_removes_alignment(self):
        sim = simulation(15.)
        # Stop before loss of hover leads to floor contact (a separate torque).
        sim.number_of_physics_steps = 500
        def no_torque(*args, **kwargs):
            force, torque, field = magnetic_wrench(*args, **kwargs)
            return force, np.zeros(3), field
        with patch('endoscopy_capsule_control.plant.z_hover_plant.magnetic_wrench', no_torque):
            sim.run()
        # The pose is not being overwritten to manufacture alignment.
        assert abs(sim.alignment_log[-1][5] - 15.) < .02
    
    
    def test_smaller_timestep_preserves_alignment(self):
        sim = simulation(15., dt=.0005)
        sim.run()
        assert sim.alignment_log[-1][4] < .2
        assert sim.alignment_log[-1][5] < .2
