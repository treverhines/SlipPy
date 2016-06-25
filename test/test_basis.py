#!/usr/bin/env python
import cosinv.basis
import numpy as np
import unittest

class Test(unittest.TestCase):
  def test_cardinal_basis(self):
    out = cosinv.basis.cardinal_basis((3,))
    self.assertTrue(np.all(np.isclose(out,np.eye(3))))

  def test_cardinal_basis_shape(self):
    out = cosinv.basis.cardinal_basis((3,))
    self.assertTrue(out.shape == (3,3))

    out = cosinv.basis.cardinal_basis((5,3))
    self.assertTrue(out.shape == (5,3,3))

    out = cosinv.basis.cardinal_basis((2,5,3))
    self.assertTrue(out.shape == (2,5,3,3))
    
  def test_cardinal_components(self):
    basis = np.array([[1.0,0.0],
                      [0.0,1.0]])
    comp = np.array([2.0,3.0])                  
    out = cosinv.basis.cardinal_components(comp,basis)
    self.assertTrue(np.all(np.isclose(out,comp)))
    
  def test_change_basis1(self):
    basis1 = np.array([[ 1.5,  0.3],
                       [-0.2,  0.8]])
    basis2 = cosinv.basis.cardinal_basis((2,))
                      
    point1 = np.array([1.1,1.5])
    # change from basis1 to basis2 
    point2 = cosinv.basis.change_basis(point1,basis1,basis2)
    # change from basis2 to basis1
    self.assertTrue(np.all(np.isclose(point2,basis1.T.dot(point1))))
    point3 = cosinv.basis.change_basis(point2,basis2,basis1)
    # change from basis1 to cardinal
    point4 = cosinv.basis.cardinal_components(point1,basis2)
    self.assertTrue(np.all(np.isclose(point1,point3)) & 
                    np.all(np.isclose(point1,point4)))

  def test_change_basis2(self):
    basis1 = np.array([[ 1.5,  0.3],
                       [-0.2,  0.8]])
    basis2 = np.array([[ 1.5,  0.3],
                       [-0.2,  0.8]])
    point1 = np.array([1.1,1.5])
    # change from basis1 to basis2 
    point2 = cosinv.basis.change_basis(point1,basis1,basis2)
    self.assertTrue(np.all(np.isclose(point2,point1)))

  def test_change_basis3(self):
    basis1 = np.array([[[ 1.5,  0.3],
                        [-0.2,  0.8]]])
    basis2 = np.array([[[ 1.5,  0.3],
                        [-0.2,  0.8]]])
    point1 = np.array([[1.1,1.5]])
    # change from basis1 to basis2 
    point2 = cosinv.basis.change_basis(point1,basis1,basis2)
    self.assertTrue(np.all(np.isclose(point2,point1)))
      

unittest.main()
