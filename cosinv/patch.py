#!/usr/bin/env python
import numpy as np
import transform
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class Patch:
  def __init__(self,pos,length,width,strike,dip,pos_patch=None):
    ''' 
    Parameters
    ----------
      pos : (3,) array
        by default, this is the top center of the fault patch. 
        This can be changed with the pos_patch argument.
      
      length : float
        patch length along strike
      
      width : float
        patch width along dip                
      
      strike : float
        patch strike in degrees
      
      dip : float
        patch dip in degrees
      
      pos_patch : (3,) array, optional
        Location of pos in the patch coordinate system.  
        Defaults to [0.5,1.0,0.0] so that pos refers to the top center 
        of the fault. Setting this to [0.0,1.0,0.0] will make pos 
        refer to the top left patch corner.

    Coordinate Systems
    ------------------
      data : This is the user coordinate system which is what pos, 
        length, width, strike, and dip are specified in. Note this is 
        a right-handed coordinate system where z is positive in the 
        vertical direction
        
      patch : The patch coordinate system has the first basis pointing 
        along the patch strike, the second basis along the patch dip, 
        and the third basis along the patch normal. The origin is the 
        bottom left corner of the patch when viewed from the side that 
        the patch is dipping towards
        
    '''  
    self.pos = np.array(pos,dtype=float)
    self.length = np.float64(length)
    self.width = np.float64(width)
    self.strike = np.float64(strike)
    self.dip = np.float64(dip)
    if pos_patch is None:
      self.pos_patch = np.array([0.5,1.0,0.0])
    else:  
      self.pos_patch = np.array(pos_patch,dtype=float)

    # build tranformation that transforms from patch coordinate system 
    # to data 
    trans  = transform.point_translation(-self.pos_patch)
    trans *= transform.point_stretch([self.length,self.width,1.0]) 
    trans *= transform.point_rotation_x(np.pi*self.dip/180.0)
    trans *= transform.point_rotation_z(np.pi/2.0 - np.pi*self.strike/180.0)
    trans *= transform.point_translation(self.pos)

    self._patch_to_data = trans
    self._data_to_patch = trans.inverse()
    self.check_breach()
    return
    
  def check_breach(self):
    ''' 
    Makes sure that the top of the fault does not breach the surface
    '''
    tol = 1e-10
    pnt_patch = np.array([[0.0,1.0,0.0],
                          [1.0,1.0,0.0],
                          [1.0,0.0,0.0],
                          [0.0,0.0,0.0]])
    pnt_data = self.patch_to_data(pnt_patch)
    if np.any(pnt_data[:,2] > tol):
      warnings.warn('patch has positive z coordinate')

  def patch_to_data(self,x):
    ''' 
    Parameters
    ----------
      x : (...,3) array in patch coordinates

    Returns 
    -------
      out : (...,3) array in data coordinates
    '''  
    return self._patch_to_data(x)

  def data_to_patch(self,x):
    ''' 
    Parameters
    ----------
      x : (...,3) array in data coordinates

    Returns 
    -------
      out : (...,3) array in patch coordinates
    '''  
    return self._data_to_patch(x)
    
  def discretize(self,Nl,Nw):    
    ''' 
    return divides the Patch into Nl*Nw Patch instances
    '''
    # create the top_corners of each subpatch  
    x_patch = np.linspace(0.0,1.0,Nl+1)[:-1]
    y_patch = np.linspace(1.0,0.0,Nw+1)[:-1]
    x_patch_grid,y_patch_grid = np.meshgrid(x_patch,y_patch)
    x_patch_flat,y_patch_flat = x_patch_grid.ravel(),y_patch_grid.ravel()
    pnt_patch = np.array([x_patch_flat,y_patch_flat,np.zeros(Nl*Nw)]).T
    print(pnt_patch.shape)
    pnt_data = self.patch_to_data(pnt_patch)
    length = self.length/Nl
    width = self.width/Nw
    sub_patches = []
    for p in pnt_data:
      sub_patches += [Patch(p,length,width,self.strike,self.dip,pos_patch=[0.0,1.0,0.0])]

    return sub_patches
  
  def get_patch_polygon(self,**kwargs):
    ''' 
    returns a matplotlib.patch.Polygon instance 
    '''
    vert = self.patch_to_data([[0.0,0.0,0.0],
                               [1.0,0.0,0.0],
                               [1.0,1.0,0.0],
                               [0.0,1.0,0.0]])
    poly = Polygon(vert[:,[0,1]],**kwargs)     
    return poly

  def draw_patch(self,ax=None,**kwargs):
    ''' 
    draws a Polygon
    '''
    if ax is None:
      ax = plt.gca()

    poly = self.get_patch_polygon(**kwargs)
    ax.add_artist(poly)
      
def draw_patches(patch_list,colors=None,ax=None,**kwargs):
  ''' 
  draws a list of Patch instances
  
  Parameters
  ----------
    patch_list : (N,) list of Patch instances
    
    colors : (N,) array of color values

  '''    
  if ax is None:
    ax = plt.gca()

  polys = []
  for p in patch_list:
    polys += [p.get_patch_polygon()]
  
  pc = PatchCollection(polys,**kwargs)
  if colors is not None:
    pc.set_array(np.array(colors))

  ax.add_collection(pc)
  return pc 

  
  
  
  
