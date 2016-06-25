#!/usr/bin/env python
import cosinv.io
import cosinv.basis
import cosinv.bm
import cosinv.patch
import cosinv.gbuild
import numpy as np
import scipy.optimize
import sys
import modest

@modest.funtime
def reg_nnls(G,L,d):
  dext = np.concatenate((d,np.zeros(L.shape[0])))
  Gext = np.vstack((G,L))
  return scipy.optimize.nnls(Gext,dext)[0]

@modest.funtime
def main(command_line_kwargs,
         gps_input_file=None,
         insar_input_file=None,
         gps_output_file=None,
         insar_output_file=None,
         slip_output_file=None):

  ### load in all data
  ###################################################################
  seg_strike = command_line_kwargs['strike']
  seg_dip = command_line_kwargs['dip']
  seg_length = command_line_kwargs['length']
  seg_width = command_line_kwargs['width']
  seg_pos_geo = command_line_kwargs['position']
  seg_Nlength = command_line_kwargs['Nlength']
  seg_Nwidth = command_line_kwargs['Nwidth']
  slip_basis = command_line_kwargs['basis']
  penalty = command_line_kwargs['penalty']
  
  obs_disp_f = np.zeros((0,))
  obs_sigma_f = np.zeros((0,))
  obs_pos_geo_f = np.zeros((0,3))
  obs_basis_f = np.zeros((0,3))

  if gps_input_file is not None:
    gps_input = cosinv.io.read_gps_data(gps_input_file)    
    Ngps = len(gps_input[0])
    obs_gps_pos_geo = gps_input[0]
    obs_gps_disp = gps_input[1]
    obs_gps_sigma = gps_input[2]
    obs_gps_basis = cosinv.basis.cardinal_basis((Ngps,3))
    
    obs_disp_fi = obs_gps_disp.reshape((Ngps*3,))
    obs_sigma_fi = obs_gps_sigma.reshape((Ngps*3,))
    obs_basis_fi = obs_gps_basis.reshape((Ngps*3,3))
    obs_pos_geo_fi = obs_gps_pos_geo[:,None,:].repeat(3,axis=1).reshape((Ngps*3,3))
    
    obs_disp_f = np.concatenate((obs_disp_f,obs_disp_fi),axis=0)
    obs_sigma_f = np.concatenate((obs_sigma_f,obs_sigma_fi),axis=0)
    obs_basis_f = np.concatenate((obs_basis_f,obs_basis_fi),axis=0)    
    obs_pos_geo_f = np.concatenate((obs_pos_geo_f,obs_pos_geo_fi),axis=0)
    
  else:
    obs_gps_pos_geo = np.zeros((0,3))
    obs_gps_disp = np.zeros((0,3))
    obs_gps_sigma = np.zeros((0,3))
    obs_gps_basis = np.zeros((0,3,3))
    Ngps = 0

  if insar_input_file is not None:
    insar_input = cosinv.io.read_gps_data(gps_input_file)    
    Ninsar = len(insar_input[0])
    obs_insar_pos_geo = insar_input[0]
    obs_insar_disp = insar_input[1]
    obs_insar_sigma = insar_input[2]
    obs_insar_basis = insar_input[3]

    obs_disp_f = np.concatenate((obs_disp_f,obs_insar_disp),axis=0)
    obs_sigma_f = np.concatenate((obs_sigma_f,obs_insar_sigma),axis=0)
    obs_basis_f = np.concatenate((obs_basis_f,obs_insar_basis),axis=0)    
    obs_pos_geo_f = np.concatenate((obs_pos_geo_f,obs_insar_pos_geo),axis=0)
  
  else:
    obs_insar_pos_geo = np.zeros((0,3))
    obs_insar_disp = np.zeros((0,))
    obs_insar_sigma = np.zeros((0,))
    obs_insar_basis = np.zeros((0,3))
    Ninsar = 0

  if gps_output_file is None:
    gps_output_file = sys.stdout

  if insar_output_file is None:
    insar_output_file = sys.stdout

  if slip_output_file is None:
    slip_output_file = sys.stdout

  ### convert from geodetic to cartesian
  ###################################################################
  bm = cosinv.bm.create_default_basemap(obs_pos_geo_f[:,0],obs_pos_geo_f[:,1])
  
  obs_pos_cart_f = cosinv.bm.geodetic_to_cartesian(obs_pos_geo_f,bm)   
  seg_pos_cart = cosinv.bm.geodetic_to_cartesian(seg_pos_geo,bm)

  ### discretize the fault segment
  ###################################################################
  seg = cosinv.patch.Patch(seg_pos_cart,
                           seg_length,seg_width,
                           seg_strike,seg_dip)
  patches = np.array(seg.discretize(seg_Nlength,seg_Nwidth))
  Ns = len(patches)  

  ### create slip basis vectors for each patch
  ###################################################################
  Ds = len(slip_basis)
  slip_basis = np.array([slip_basis for i in range(Ns)])
  slip_basis_f = slip_basis.reshape((Ns*Ds,3))
  patches_f = patches[:,None].repeat(Ds,axis=1).reshape((Ns*Ds,))
    
  ### build system matrix
  ###################################################################
  G = cosinv.gbuild.build_system_matrix(obs_pos_cart_f,
                                        patches_f,
                                        obs_basis_f,
                                        slip_basis_f)
  
  ### build regularization matrix
  ###################################################################
  L = penalty*np.eye(Ns*Ds)
  
  ### estimate slip and compute predicted displacement
  #####################################################################
  slip_f = reg_nnls(G,L,obs_disp_f)
  pred_disp_f = G.dot(slip_f)

  slip = slip_f.reshape((Ns,Ds))
  cardinal_slip = cosinv.basis.cardinal_components(slip,slip_basis)

  # split predicted displacements into insar and GPS component
  pred_disp_f_gps = pred_disp_f[:3*Ngps]
  pred_disp_gps = pred_disp_f_gps.reshape((Ngps,3))
  pred_disp_insar = pred_disp_f[3*Ngps:]

  ### get slip patch data
  #####################################################################
  patches_pos_cart =[i.patch_to_user([0.5,1.0,0.0]) for i in patches]
  patches_pos_geo = cosinv.bm.cartesian_to_geodetic(patches_pos_cart,bm)
  patches_strike = [i.strike for i in patches]
  patches_dip = [i.dip for i in patches]
  patches_length = [i.length for i in patches]
  patches_width = [i.width for i in patches]

  ### write output
  #####################################################################
  cosinv.io.write_slip_data(patches_pos_geo,
                            patches_strike,patches_dip,
                            patches_length,patches_width,
                            cardinal_slip,slip_output_file)

  cosinv.io.write_gps_data(obs_gps_pos_geo,
                           pred_disp_gps,0.0*pred_disp_gps,
                           gps_output_file)

  cosinv.io.write_insar_data(obs_insar_pos_geo,
                             pred_disp_insar,0.0*pred_disp_insar,
                             obs_insar_basis, 
                             insar_output_file)
  
  return
