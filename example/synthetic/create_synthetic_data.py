#!/usr/bin/env python
import numpy as np
import slippy.patch
import slippy.basis
import slippy.io
import slippy.gbuild
import slippy.bm
import slippy.quiver
import matplotlib.pyplot as plt

### patch specifications
#####################################################################
strike = 70.0 # degrees
dip = 45.0 # degrees
length = 200000.0 # meters
width = 60000.0 # meters
seg_pos_geo = [-84.2,43.3,0.0] # top center of patch
Nl = 10
Nw = 10

## observation points
#####################################################################
Nx = 100
#pos_geo = np.random.uniform(-3,3,(Nx,3))
pos_geo = np.random.normal(0.0,1.0,(Nx,3))
pos_geo[:,0] += -84.2
pos_geo[:,1] += 43.3
pos_geo[:,2] = 0.0
disp_basis = slippy.basis.cardinal_basis((Nx,3))

# flatten the observation points and basis vectors
pos_geo_f = pos_geo[:,None,:].repeat(3,axis=1).reshape((Nx*3,3))
disp_basis_f = disp_basis.reshape((Nx*3,3))

### convert from cartesian to geodetic
bm = slippy.bm.create_default_basemap(pos_geo[:,0],pos_geo[:,1],resolution='i')
bm.drawcoastlines()
bm.drawstates()

seg_pos_cart = slippy.bm.geodetic_to_cartesian(seg_pos_geo,bm)
pos_cart_f = slippy.bm.geodetic_to_cartesian(pos_geo_f,bm)
pos_cart = slippy.bm.geodetic_to_cartesian(pos_geo,bm)

### create synthetic slip
#####################################################################
P = slippy.patch.Patch(seg_pos_cart,length,width,strike,dip)
Ps = np.array(P.discretize(Nl,Nw))
Ns = len(Ps)
# find the centers of each patch in user coordinates
patch_pos = [P.user_to_patch(i.patch_to_user([0.5,0.5,0.0])) for i in Ps]
patch_pos = np.asarray(patch_pos)

# define slip as a function of patch position
slip = np.zeros((len(Ps),3))
slip[:,0] = 1.0/(1.0 + 20*((patch_pos[:,0] - 0.5)**2 + (patch_pos[:,1] - 0.5)**2))

slippy.patch.draw_patches(Ps,colors=slip[:,0],edgecolor='none')

slip_basis = slippy.basis.cardinal_basis((Ns,3))
slip_basis_f = slip_basis.reshape((Ns*3,3))

slip_f = slip.reshape((Ns*3,))

patches_f = Ps[:,None].repeat(3,axis=1).reshape((Ns*3,))

### create synthetic data
#####################################################################
G = slippy.gbuild.build_system_matrix(pos_cart_f,patches_f,disp_basis_f,slip_basis_f)

disp_f = G.dot(slip_f)
disp = disp_f.reshape((Nx,3))
sigma = 0.003*np.ones((Nx,3))
# add white noise
disp += np.random.normal(0.0,sigma)

slippy.quiver.quiver(pos_cart[:,0],pos_cart[:,1],disp[:,0],disp[:,1],
                     sigma=(sigma[:,0],sigma[:,1],0.0*sigma[:,0]),
                     scale=0.000001)

### write out synthetic gps data
#####################################################################
slippy.io.write_gps_data(pos_geo,disp,sigma,'synthetic_gps.txt')

### dot the synthetic gps data with an arbitrary look vector to get 
### synthetic insar
#####################################################################
look = np.array([1.0,1.0,1.0])
look = look[None,:].repeat(Nx,axis=0)
disp = np.einsum('...i,...i',disp,look)
sigma = 0.01*np.ones((Nx,))
slippy.io.write_insar_data(pos_geo,disp,sigma,look,'synthetic_insar.txt')

cm = plt.scatter(pos_cart[:,0],pos_cart[:,1],s=100,c=disp,cmap='viridis')
cbar = plt.colorbar(cm)
cbar.set_label('displacement along %s' % look[0])
plt.show()


