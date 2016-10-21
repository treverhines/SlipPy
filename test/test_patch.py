#!/usr/bin/env python
import slippy.patch
import numpy as np
import unittest
import matplotlib.pyplot as plt

class Test(unittest.TestCase):
  def test_patch1(self):
    strike = 0.0
    dip = 0.0
    length = 10.0
    width = 5.0
    pos = [0.0,0.0,0.0]
    pos_patch = [0.0,1.0,0.0] 
    P = slippy.patch.Patch(pos,length,width,strike,dip,pos_patch=pos_patch)
    corners_patch = np.array([[0.0,0.0,0.0],
                              [1.0,0.0,0.0],
                              [1.0,1.0,0.0],
                              [0.0,1.0,0.0]])
    corners_data = P.patch_to_user(corners_patch)
    true_corners_data = np.array([[5.0,0.0,0.0],
                                  [5.0,10.0,0.0],
                                  [0.0,10.0,0.0],
                                  [0.0,0.0,0.0]])
    self.assertTrue(np.all(np.isclose(corners_data,true_corners_data)))

