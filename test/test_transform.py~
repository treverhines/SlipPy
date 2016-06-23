#!/usr/bin/env python
import cosinv.transform
import numpy as np
import unittest

class Test(unittest.TestCase):
  def test_idenity(self):
    T  = cosinv.transform.point_translation([1.0,2.0,3.0])
    T *= cosinv.transform.point_stretch([1.5,2.5,3.5])
    T *= cosinv.transform.point_rotation_x(1.5)
    T *= cosinv.transform.point_rotation_y(2.5)
    T *= cosinv.transform.point_rotation_z(3.5)
    T *= cosinv.transform.basis_rotation_z(3.5)
    T *= cosinv.transform.basis_rotation_y(2.5)
    T *= cosinv.transform.basis_rotation_x(1.5)
    T *= cosinv.transform.basis_stretch([1.5,2.5,3.5])
    T *= cosinv.transform.basis_translation([1.0,2.0,3.0])
    self.assertTrue(np.all(np.isclose(T.get_M(),np.eye(4))))

  def test_translation(self):
    point = np.array([1.0,2.0,3.0])  
    trans = np.array([3.0,4.0,5.0]) 
    T = cosinv.transform.point_translation(trans)
    self.assertTrue(np.all(np.isclose(T(point),point+trans)))

  def test_stretch(self):
    point = np.array([1.0,2.0,3.0])  
    trans = np.array([3.0,4.0,5.0]) 
    T = cosinv.transform.point_stretch(trans)
    self.assertTrue(np.all(np.isclose(T(point),point*trans)))

  def test_rotate_x(self):
    point = np.array([0.0,1.0,0.0])  
    T = cosinv.transform.point_rotation_x(np.pi/2.0)
    soln = np.array([0.0,0.0,1.0]) 
    self.assertTrue(np.all(np.isclose(T(point),soln)))

  def test_rotate_y(self):
    point = np.array([0.0,0.0,1.0])  
    T = cosinv.transform.point_rotation_y(np.pi/2.0)
    soln = np.array([1.0,0.0,0.0]) 
    self.assertTrue(np.all(np.isclose(T(point),soln)))

  def test_rotate_z(self):
    point = np.array([1.0,0.0,0.0])  
    T = cosinv.transform.point_rotation_z(np.pi/2.0)
    soln = np.array([0.0,1.0,0.0]) 
    self.assertTrue(np.all(np.isclose(T(point),soln)))

unittest.main()
