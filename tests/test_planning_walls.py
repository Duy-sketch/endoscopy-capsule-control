import unittest
import numpy as np
from endoscopy_capsule_control.planning.collision import segment_rectangle_distance
from endoscopy_capsule_control.planning.astar import XYPlan

class WallPlanningTests(unittest.TestCase):
    def test_segment_distance(self):
        for a,b,d in [([-20,0],[20,0],0),([-20,12],[20,12],2),([8,14],[8,14],5),([0,0],[0,1],0),([5,-20],[5,20],0)]:
            np.testing.assert_allclose(segment_rectangle_distance(a,b,[[0,0,10,20]]),d)

    def test_straight_empty_map(self):
        p=XYPlan([0,0,100],[50,0,100],[],[-10,60,-40,40],2,15,3,2,2,.01)
        self.assertEqual(len(p.path),2)

    def test_detour_tangency_and_clearance(self):
        p=XYPlan([-40,0,100],[40,0,100],[],[-60,60,-50,50],2,15,3,2,2,.01,rectangles=[[0,0,10,20]])
        self.assertFalse(p.segment_safe([-40,0],[40,0]))
        self.assertFalse(p.segment_safe([-40,28],[40,28]))
        self.assertTrue(p.segment_safe([-40,28.01],[40,28.01]))
        self.assertTrue(all(p.segment_safe(a,b) for a,b in zip(p.path[:-1],p.path[1:])))
        self.assertLess(p.body_clearance_mm([0,0]),0)

    def test_invalid_start(self):
        with self.assertRaises(ValueError):
            XYPlan([0,0,100],[40,0,100],[],[-60,60,-50,50],2,15,3,2,2,.01,rectangles=[[0,0,10,20]])

if __name__=='__main__': unittest.main()
