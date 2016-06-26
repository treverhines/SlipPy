#!/usr/bin/env python
import numpy as np

def read_gps_data(file_name):  
  ''' 
  FILE FORMAT
  -----------
    
    HEADER
    lon[degrees] lat[degrees] disp_e[m] disp_n[m] disp_v[m] sigma_e[m] sigma_n[m] sigma_v[m]
    
  '''
  data = np.loadtxt(file_name,skiprows=1)
  Nx = len(data) 
  lonlat = data[:,[0,1]]
  disp = data[:,[2,3,4]]
  sigma = data[:,[5,6,7]]
  pos_geodetic = np.hstack((lonlat,np.zeros((Nx,1))))
  return pos_geodetic,disp,sigma


def read_insar_data(file_name):  
  ''' 
  FILE FORMAT
  -----------
  
    HEADER
    lon[degrees] lat[degrees] component[m] sigma[m] basis_e basis_n basis_v
    
  '''  
  data = np.loadtxt(file_name,skiprows=1)
  Nx = len(data) 
  lonlat = data[:,[0,1]]
  disp = data[:,2]
  sigma = data[:,3]
  basis = data[:,[4,5,6]]
  pos_geodetic = np.hstack((lonlat,np.zeros((Nx,1))))
  return pos_geodetic,disp,sigma,basis


def read_slip_data(file_name):
  ''' 
  FILE FORMAT
  -----------

    HEADER
    lon[degrees] lat[degrees] height[m] strike[degrees] dip[degrees] length[m] width[m] left-lateral[m] thrust[m] tensile[m]
  '''
  data = np.loadtxt(file_name,skiprows=1)
  pos_geodetic = data[:,[0,1,2]]
  strike = data[:,3]
  dip = data[:,4]
  length = data[:,5]
  width = data[:,6]
  slip = data[:,[7,8,9]]
  return pos_geodetic,strike,dip,length,width,slip
  

def write_gps_data(pos_geodetic,disp,sigma,file_name):
  ''' 
  FILE FORMAT
  -----------

    HEADER
    lon[degrees] lat[degrees] disp_e[m] disp_n[m] disp_v[m] sigma_e[m] sigma_n[m] sigma_u[m]
  '''
  pos_geodetic = np.asarray(pos_geodetic)
  disp = np.asarray(disp)
  sigma = np.asarray(sigma)
  
  lonlat = pos_geodetic[:,[0,1]]
  data = np.hstack((lonlat,disp,sigma))
  header = "lon[degrees] lat[degrees] disp_e[m] disp_n[m] disp_v[m] sigma_e[m] sigma_n[m] sigma_u[m]"
  np.savetxt(file_name,data,header=header,fmt='%0.4f')
  return


def write_insar_data(pos_geodetic,disp,sigma,basis,file_name):  
  ''' 
  FILE FORMAT
  -----------
  
    HEADER
    lon[degrees] lat[degrees] disp_los[m] sigma_los[m] V_e V_n V_u
  '''  
  pos_geodetic = np.asarray(pos_geodetic)
  disp = np.asarray(disp)
  sigma = np.asarray(sigma)
  basis = np.asarray(basis)
  
  lonlat = pos_geodetic[:,[0,1]]
  data = np.hstack((lonlat,disp[:,None],sigma[:,None],basis))
  header = "lon[degrees] lat[degrees] disp_los[m] sigma_los[m] V_e V_n V_u"
  np.savetxt(file_name,data,header=header,fmt='%0.4f')
  return


def write_slip_data(pos_geodetic,strike,dip,length,width,slip,file_name):  
  ''' 
  FILE FORMAT
  -----------

    HEADER
    lon[degrees] lat[degrees] depth[m] strike[degrees] dip[degrees] length[m] width[m] left-lateral[m] thrust[m] tensile[m]
  '''
  pos_geodetic = np.asarray(pos_geodetic)
  strike = np.asarray(strike)
  dip = np.asarray(dip)
  length = np.asarray(length)
  width = np.asarray(width)
  slip = np.asarray(slip)

  data = np.hstack((pos_geodetic,strike[:,None],
                    dip[:,None],length[:,None],width[:,None],slip))
  header = "lon[degrees] lat[degrees] depth[m] strike[degrees] dip[degrees] length[m] width[m] left-lateral[m] thrust[m] tensile[m]"
  np.savetxt(file_name,data,header=header,fmt='%0.4f')
  return
