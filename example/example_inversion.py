#!/usr/bin/env python
import numpy as np
import cosinv.patch
import cosinv.basis
import cosinv.io
import cosinv.bm
import cosinv.inversion
from cosinv.gbuild import build_system_matrix
import matplotlib.pyplot as plt
import scipy.optimize
np.random.seed(1)

''' 
def reg_nnls(G,L,d):
  dext = np.concatenate((d,np.zeros(L.shape[0])))
  Gext = np.vstack((G,L))
  return scipy.optimize.nnls(Gext,dext)[0]


### patch specifications
#####################################################################
#####################################################################
#####################################################################
strike = 20.0 # degrees
dip = 80.0 # degrees
length = 1000000.0
width = 1000000.0
segment_pos_geo = np.array([0.0,0.0,0.0]) # top center of segment
Nl = 40
Nw = 20
slip_basis = np.array([[ 1.0,  1.0, 0.0],
                       [ 1.0, -1.0, 0.0]])

#####################################################################
#####################################################################
#####################################################################
### LOAD IN ALL DATA
pos_geo,disp,sigma = cosinv.io.read_gps_data('synthetic_gps.txt')



## observation points
#####################################################################
Nx  = len(pos_geo)

# create basemap to convert geodetic to cartesian
bm = cosinv.bm.create_default_basemap(pos_geo[:,0],pos_geo[:,1])
pos_cart = cosinv.bm.geodetic_to_cartesian(pos_geo,bm)

# flatten the positions, displacements, and uncertainties
pos_f = pos_cart[:,None,:].repeat(3,axis=1).reshape((Nx*3,3))
disp_f = disp.reshape(Nx*3)
sigma_f = sigma.reshape(Nx*3)

# get the basis vector for each displacement component
disp_basis = cosinv.basis.cardinal_basis((Nx,3))
disp_basis_f = disp_basis.reshape((Nx*3,3))


### build patches
#####################################################################
segment_pos_cart = cosinv.bm.geodetic_to_cartesian(segment_pos_geo,bm)

P = cosinv.patch.Patch(segment_pos_cart,length,width,strike,dip)
Ps = np.array(P.discretize(Nl,Nw))
Ns = len(Ps)
Ds = len(slip_basis)
slip_basis = np.array([slip_basis for i in Ps])
slip_basis_f = slip_basis.reshape((Ns*Ds,3))
Ps_f = Ps[:,None].repeat(Ds,axis=1).reshape((Ns*Ds,)) 


#####################################################################
### Build System matrix
#####################################################################
G = cosinv.gbuild.build_system_matrix(pos_f,Ps_f,disp_basis_f,slip_basis_f)
L = 0.001*np.eye(len(Ps_f))

#####################################################################
### Estimate slip
#####################################################################
slip_f = reg_nnls(G,L,disp_f)
pred_f = G.dot(slip_f)

slip = slip_f.reshape((Ns,Ds))
slip = cosinv.basis.cardinal_components(slip,slip_basis)
pred = pred_f.reshape((Nx,3))
pred_sigma = np.zeros((Nx,3))


#####################################################################
### write solution
#####################################################################
cosinv.io.write_gps_data(pos_geo,pred,pred_sigma,'predicted_gps.txt')

patch_pos_cart = np.array([i.patch_to_user([0.0,1.0,0.0]) for i in Ps])
patch_pos_geo = cosinv.bm.cartesian_to_geodetic(patch_pos_cart,bm)
patch_strike = [i.strike for i in Ps]
patch_dip = [i.dip for i in Ps]
patch_length = [i.length for i in Ps]
patch_width = [i.width for i in Ps]

cosinv.io.write_slip_data(patch_pos_geo,
                          patch_strike,patch_dip,
                          patch_length,patch_width,
                          slip,'predicted_slip.txt')

#####################################################################
### Plot Solution
#####################################################################
'''
strike = 20.0 # degrees
dip = 80.0 # degrees
length = 1000000.0
width = 1000000.0
segment_pos_geo = np.array([0.0,0.0,0.0]) # top center of segment
Nl = 40
Nw = 20
slip_basis = np.array([[ 1.0,  1.0, 0.0],
                       [ 1.0, -1.0, 0.0]])
params = {'strike':20.0,
          'dip':80.0,
          'length':1000000.0,
          'width':1000000.0,
          'position':[0.0,0.0,0.0],
          'Nlength':20,
          'Nwidth':20,
          'basis':[[1.0,1.0,0.0],[1.0,-1.0,0.0]],
          'penalty':0.001}

cosinv.inversion.main(params,
                      gps_input_file='synthetic_gps.txt',
                      gps_output_file='predicted_gps.txt',
                      slip_output_file='predicted_slip.txt')
                   

pred_pos_geo,pred_disp,pred_sigma = cosinv.io.read_gps_data('predicted_gps.txt')
bm = cosinv.bm.create_default_basemap(pred_pos_geo[:,0],pred_pos_geo[:,1])
pred_pos_cart = cosinv.bm.geodetic_to_cartesian(pred_pos_geo,bm)

obs_pos_geo,obs_disp,obs_sigma = cosinv.io.read_gps_data('synthetic_gps.txt')
obs_pos_cart = cosinv.bm.geodetic_to_cartesian(obs_pos_geo,bm)

input = cosinv.io.read_slip_data('predicted_slip.txt')
patch_pos_geo = input[0]
patch_pos_cart = cosinv.bm.geodetic_to_cartesian(patch_pos_geo,bm)
patch_strike = input[1]
patch_dip = input[2]
patch_length = input[3]
patch_width = input[4]
slip = input[5]
Ps = [cosinv.patch.Patch(p,l,w,s,d) for p,l,w,s,d in zip(patch_pos_cart,
                                                         patch_length,
                                                         patch_width,
                                                         patch_strike,
                                                         patch_dip)]
fig,ax = plt.subplots()
ax.set_title('left lateral')
bm.drawcoastlines(ax=ax)
q = ax.quiver(obs_pos_cart[:,0],obs_pos_cart[:,1],
              obs_disp[:,0],obs_disp[:,1],
              zorder=1,color='k',scale=1.0)
ax.quiver(pred_pos_cart[:,0],pred_pos_cart[:,1],
          pred_disp[:,0],pred_disp[:,1],
          zorder=1,color='m',scale=1.0)
          
ax.quiverkey(q,0.8,0.2,0.05,'0.05 m')
ps = cosinv.patch.draw_patches(Ps,colors=slip[:,0],ax=ax,edgecolor='none',zorder=0)
fig.colorbar(ps,ax=ax)
fig,ax = plt.subplots()

ax.set_title('thrust')
bm.drawcoastlines(ax=ax)
q = ax.quiver(obs_pos_cart[:,0],obs_pos_cart[:,1],
              obs_disp[:,0],obs_disp[:,1],
              zorder=1,color='k',scale=1.0)
ax.quiver(pred_pos_cart[:,0],pred_pos_cart[:,1],
          pred_disp[:,0],pred_disp[:,1],
          zorder=1,color='m',scale=1.0)
          
ax.quiverkey(q,0.8,0.2,0.05,'0.05 m')
ps = cosinv.patch.draw_patches(Ps,colors=slip[:,1],ax=ax,edgecolor='none',zorder=0)
fig.colorbar(ps,ax=ax)
plt.show()
quit()
