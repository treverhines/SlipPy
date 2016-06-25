import numpy as np
import cosinv.dc3d
import warnings
import modest

def patch_dislocation(x,slip,patch,lamb=3.2e10,mu=3.2e10):
  ''' 
  Parameters
  ----------
    x : (N,3) array
      observation points
    
    slip : (3,) array
      slip vector
      
    patch : Patch instance
    
    lamb : float  

    mu : float
  
  Returns
  -------
    disp : (N,3) array
      displacements
    
    derr : (N,3,3) array
      displacement derivatives  
  '''
  return dislocation(x,slip,
                     patch.patch_to_user([0.5,1.0,0.0]),
                     patch.length,patch.width,
                     patch.strike,patch.dip,
                     lamb=lamb,mu=mu)
                      
@modest.funtime
def dislocation(x,
                slip,
                top_center,
                length,
                width,
                strike,
                dip,
                lamb=3.2e10,
                mu=3.2e10):
  ''' 
  wrapper for the Okada 1992 solution displacements and displacement 
  gradients resulting from a rectangular dislocation. This function 
  handles coordinate system rotations and translations needed to 
  describe dislocations which are anchored at the origin.

  Parameters
  ----------
   
    x : (N,3) array
      observation points. The z coordinate should be negative

    slip : (3,) array length
      left-lateral, thrust, and tensile motion on the fault

    top_center : (3,) array
      the top center of the dislocation

    length : float
      length of the dislocation along the strike direction

    width : float
      width of the dislocation along the dip direction
  
    strike : float
      strike of the fault patch in degrees 

    dip : float
      dip of the fault patch in degrees

  Returns
  -------
    disp,derr
      
    disp : (N,3) array
      displacements

    derr : (N,3,3) array 
      displacement gradient tensors. where the the second axis is the 
      displacement direction and the third axis is the derivative 
      direction

  '''
  tol = 1e-10

  x = np.array(x,copy=True)
  slip = np.asarray(slip)
  center = np.asarray(top_center)

  # elastic property
  alpha = (lamb + mu)/(lamb + 2*mu)
  
  # convert strike to the rotation angle w.r.t east and in radians
  argZ = np.pi/2.0 - strike*np.pi/180

  # convert dip to radians
  argX = dip*np.pi/180

  # depth to fault top
  c = -center[2]

  if np.any(x[:,2] > tol):
    raise ValueError('values for z coordinate must be negative')

  # length range
  length_range = np.array([-0.5*length,0.5*length])
  width_range = np.array([-width,0.0])

  # transform to the okada reference frame
  x[:,0] -= center[0]
  x[:,1] -= center[1]
  R = np.array([[   np.cos(argZ),   np.sin(argZ),       0.0],
                [  -np.sin(argZ),   np.cos(argZ),       0.0],
                [            0.0,            0.0,       1.0]])
  x = np.einsum('ij,...j',R,x)

  disp = np.zeros((len(x),3))
  derr = np.zeros((len(x),3,3))
  for i,xi in enumerate(x):
    out = cosinv.dc3d.dc3dwrapper(alpha,xi,c,dip,length_range,width_range,slip)
    status = out[0]
    if status != 0:
      warnings.warn('dc3d returned with error code %s' % status)   

    disp[i,:] = out[1]
    derr[i,:,:] = out[2].T

  # return solution to original coordinate system
  disp = np.einsum('ij,...j',R.T,disp)
  derr = np.einsum('ij,...jk,kl',R.T,derr,R)

  return disp,derr




