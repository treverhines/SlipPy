#!/usr/bin/env python
import numpy as np
import cosinv.patch
import cosinv.basis
import cosinv.io
from cosinv.gbuild import flatten_vector_array
from cosinv.gbuild import build_system_matrix
import matplotlib.pyplot as plt
import scipy.optimize 

def reg_nnls(G,L,d):
  dext = np.concatenate((d,np.zeros(L.shape[0])))
  Gext = np.vstack((G,L))
  return scipy.optimize.nnls(Gext,dext)[0]
  

def format_gps(pos,disp,sigma):
  ''' 
  converts GPS data into a flat format used for the inversion
  
  Parameters
  ----------
    pos : (Nx,2) array
    disp : (Nx,3) array
    sigma : (Nx,3) array

  Returns
  -------
    pos_f : (Nx*3,3) array
      station position for each component

    disp_f : (Nx*3,) array
      displacement component

    sigma_f : (Nx*3,) array
      displacement component uncertainty

    basis_f : (Nx*3,3) array
      direction for each displacement component
  '''  
  pos = np.asarray(pos)
  disp = np.asarray(disp)
  sigma = np.asarray(sigma)
  
  Nx = len(pos)  
  # convert pos from a 2D to 3D vector field
  pos = np.hstack((pos,np.zeros((Nx,1))))

  pos_f = pos[:,None,:].repeat(3,axis=1).reshape((Nx*3,3))
  disp_f = disp.reshape(Nx*3)  
  sigma_f = sigma.reshape(Nx*3)  

  basis = cosinv.basis.cardinal_basis((Nx,3))
  basis_f = basis.reshape((Nx*3,3))  
  return pos_f,disp_f,sigma_f,basis_f
  
def format_slip(patches,slip):
  ''' 
  Parameters
  ----------
    patches : (Ns,) array
    
    slip : (Ns,3) array
  '''  
  patches = np.asarray(patches)
  Ns = len(patches)     
  
  patches_f = patches[:,None].repeat(3,axis=1).reshape((Ns*3,))
  slip_f = slip.reshape((Ns*3,))
  basis = cosinv.basis.cardinal_basis((Ns,3))
  basis_f = basis.reshape((Ns*3,3))
  return patches_f,slip_f,basis_f
  

### patch specifications
#####################################################################
strike = 20.0 # degrees
dip = 80.0 # degrees
length = 5.0
width = 5.0
segment_top_center = [0.0,0.0,0.0] # top center of patch
Nl = 20
Nw = 20

## observation points
#####################################################################
Nx = 100
pos = np.random.uniform(-10,10,(Nx,2))
disp = np.zeros((Nx,3))
sigma = np.zeros((Nx,3))

pos_f,disp_f,sigma_f,disp_basis_f = format_gps(pos,disp,sigma)

#####################################################################
### create synthetic slip
#####################################################################
P = cosinv.patch.Patch(segment_top_center,length,width,strike,dip)
Ps = np.array(P.discretize(Nl,Nw))

patch_pos = [P.user_to_patch(i.patch_to_user([0.5,0.5,0.0])) for i in Ps]
patch_pos = np.asarray(patch_pos)
slip = np.zeros((len(Ps),3))
slip[:,0] = 1.0/(1.0 + 20*((patch_pos[:,0] - 0.5)**2 + (patch_pos[:,1] - 0.5)**2))

patches_f,slip_f,slip_basis_f = format_slip(Ps,slip)

#####################################################################
### create synthetic data
#####################################################################
G = cosinv.gbuild.build_system_matrix(pos_f,patches_f,disp_basis_f,slip_basis_f)

disp_f = G.dot(slip_f)
disp = disp_f.reshape((Nx,3))
sigma = np.ones((Nx,3))

#####################################################################
### write out synthetic data
#####################################################################
cosinv.io.write_gps_data(pos[:,[0,1]],disp,sigma,'synthetic_gps.txt')

#####################################################################
### view synthetic data
#####################################################################
fig,ax = plt.subplots()
pc = cosinv.patch.draw_patches(Ps,ax=ax,colors=slip[:,0],edgecolor='none')
ax.quiver(pos[:,0],pos[:,1],disp[:,0],disp[:,1])
ax.scatter(pos[:,0],pos[:,1],s=100,c=disp[:,2])
ax.set_xlim((-10,10))
ax.set_ylim((-10,10))
plt.show()


