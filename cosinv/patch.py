#!/usr/bin/env python
import numpy as np
import transform

class Patch:
  def __init__(pos,
               length,width,
               strike,dip):
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
      
      pos_patch : (3,) array
        Location of pos in the patch coordinate system.  
        Defaults to [0.5,1.0,0.0] so that pos refers to the top center 
        of the fault. Setting this to [0.0,1.0,0.0] will make pos 
        refer to the top left patch corner.

    Coordinate Systems
    ------------------
      data : This is the user coordinate system which is what pos, 
        length, width, strike, and dip are specified in.
        
      patch : The patch coordinate system has the first basis pointing 
        along the patch strike, the second basis along the patch dip, 
        and the third basis along the patch normal. The origin is the 
        bottom left corner of the patch when viewed from the side that 
        the patch is dipping towards
        
    '''  
    self.pos = np.array(pos,dtype=float)
    self.pos_patch = np.array(pos_patch,dtype=float)
    self.length = length
    self.width = width
    self.strike = strike
    self.dip = dip

    # build tranformation that transforms from patch coordinate system 
    # to data 
    trans = transform.point_translation(-self.pos_patch)
    trans *= transform.point_stretch([self.length,self.width,1.0]) 
    trans *= transform.point_rotation_x(np.pi*self.dip/180.0)
    trans *= transform.point_rotation_z(np.pi/2.0 - np.pi*self.strike/180.0)
    trans *= transform.point_translation(self.pos)

    self._patch_to_data = trans
    self._data_to_patch = trans.inverse()
    return
    
  def patch_to_data(self,x):
    return self._patch_to_data(x)

  def data_to_patch(self,x):
    return self._data_to_patch(x)
    
  def discretize(self,Nl,Nw):    
    ''' 
    return divides the Patch into Nl*Nw Patch instances
    '''
    return 
  
