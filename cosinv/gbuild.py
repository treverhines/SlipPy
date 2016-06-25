#!/usr/bin/env python
from cosinv.okada import dislocation
import numpy as np

def build_system_matrix(pos,patches,disp_directions,slip_directions):
  ''' 
  builds the system matrix 

  Parameters
  ----------
    pos : (N,3) array of surface observation points
    
    patches : (M,) list of Patch instances
    
    disp_direction : (N,3) array
      displacement direction
      
    slip_direction : (M,3) array
      slip directions
  
  Returns
  -------
    out : (N,M) array of dislocation greens functions

  '''
  pos = np.asarray(pos)
  slip_directions = np.asarray(slip_directions)
  disp_directions = np.asarray(disp_directions)
  
  G = np.zeros((len(pos),len(patches)))
  for i,p in enumerate(patches):
    top_center = p.patch_to_user([0.5,1.0,0.0])
    disp,derr = dislocation(pos,slip_directions[i],top_center,
                            p.length,p.width,p.strike,p.dip)
    disp = np.einsum('...j,...j',disp,disp_directions)
    G[:,i] = disp
                                
  return G
  
  
