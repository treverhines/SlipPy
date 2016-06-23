#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import tikhonov
from okada_wrapper import dislocation
from mpl_toolkits.basemap import Basemap
from xsection import XSection
import mayavi.mlab
import transform as trans
import scipy.optimize

def create_basemap(lon_lst,lat_lst,resolution='i'):
  ''' 
  creates a basemap that bounds lat_lst and lon_lst
  '''
  lon_buff = (max(lon_lst) - min(lon_lst))/20.0
  lat_buff = (max(lat_lst) - min(lat_lst))/20.0
  if lon_buff < 0.2:
    lon_buff = 0.2

  if lat_buff < 0.2:
    lat_buff = 0.2

  llcrnrlon = min(lon_lst) - lon_buff
  llcrnrlat = min(lat_lst) - lat_buff
  urcrnrlon = max(lon_lst) + lon_buff
  urcrnrlat = max(lat_lst) + lat_buff
  lon_0 = (llcrnrlon + urcrnrlon)/2.0
  lat_0 = (llcrnrlat + urcrnrlat)/2.0
  return Basemap(projection='tmerc',
                 lon_0 = lon_0,
                 lat_0 = lat_0,
                 llcrnrlon = llcrnrlon,
                 llcrnrlat = llcrnrlat,
                 urcrnrlon = urcrnrlon,
                 urcrnrlat = urcrnrlat,
                 resolution = resolution)


def plot_segment(slip,top_center,length,width,strike,dip,clim=None,**kwargs):
  '''  
  plots a fault segment in Mayavi
  '''
  # set colorbar range here
  if clim is None:
    clim = (-2.0,2.0)

  def slip_fun(x,c): return c + 0.0*x[:,0]

  top_center = np.asarray(top_center)

  strike = np.pi*strike/180.0
  dip = np.pi*dip/180.0
  xc = top_center[0] - 0.5*length*np.cos(np.pi/2.0 - strike)
  yc = top_center[1] - 0.5*length*np.sin(np.pi/2.0 - strike)

  t = trans.point_stretch([length,width,1.0])
  t += trans.point_rotation_x(dip)
  t += trans.point_rotation_z(np.pi/2.0 - strike)
  t += trans.point_translation([xc,yc,top_center[2]])
  xs = XSection(slip_fun,f_args=(slip,),base_square_y=(-1,0),transforms=[t],clim=clim,Nl=2,Nw=2)
  xs.draw(**kwargs)


def discretize_segment(top_center,length,width,strike,dip,Nl,Nw):
  strike = np.pi*strike/180.0
  argz = np.pi/2.0 - strike
  dip = np.pi*dip/180.0
  xc = top_center[0] - 0.5*length*np.cos(np.pi/2.0 - strike)
  yc = top_center[1] - 0.5*length*np.sin(np.pi/2.0 - strike)
  zc = top_center[2]
  patches = []
  dl = length/(1.0*Nl)
  dw = width/(1.0*Nw)
  for i in range(Nl):
    for j in range(Nw):
      center_xi = xc + ((i+0.5)*dl)*np.cos(argz) + (j*dw)*np.cos(dip)*np.cos(argz-np.pi/2.0)
      center_yi = yc + ((i+0.5)*dl)*np.sin(argz) + (j*dw)*np.cos(dip)*np.sin(argz-np.pi/2.0)
      center_zi = zc - (j*dw)*np.sin(dip)
      #anchor_lon,anchor_lat = bm(anchor_xi,anchor_yi,inverse=True)
      centeri = np.array([center_xi,center_yi,center_zi])
      patches += [(centeri,dl,dw,180*strike/np.pi,180*dip/np.pi)]

  return patches


def build_system_matrix(obs_pnts,patches,direction='thrust'):
  if direction == 'thrust':
    slip = np.array([0.0,1.0,0.0])
  if direction == 'left-lateral':
    slip = np.array([1.0,0.0,0.0])

  M = len(patches)
  N = len(obs_pnts)
  G = np.zeros((N,3,M))
  for i,p in enumerate(patches):
    center = p[0]
    length = p[1]
    width = p[2]
    strike = p[3]
    dip = p[4]
    disp_pred,disp_derr = dislocation(obs_pnts,slip,center,
                                      length,width,strike,dip)
    disp_pred *= 1000.0
    G[:,:,i] = disp_pred
  
  return G

# FAULT GEOMETRY
#####################################################################
# center of fault segment
fault_lon = 120.4971
fault_lat = 22.9859
fault_strike = 169.6369
fault_dip = 80.8
fault_length = 50000.0 # meters
fault_width = 50000.0 # meters

# number of fault patches along strike direction
Nx = 10
# number of fault patches along dip direction
Ny = 10

# damping parameters
damping = 100.0

#####################################################################

# LOAD GPS DATA
#####################################################################
f1 = np.loadtxt('continel_gps_site.txt',skiprows=1,dtype=str) 
f2 = np.loadtxt('movement_gps_site.txt',skiprows=1,dtype=str) 
data = f1

# station IDs
names = data[:,0]

# position
lon = data[:,1].astype(float)
lat = data[:,2].astype(float)

# displacements
u = data[:,3].astype(float)
v = data[:,4].astype(float)
z = data[:,8].astype(float)
obs_disp = np.array([u,v,z]).T

# uncertainty
su = data[:,5].astype(float)
sv = data[:,6].astype(float)
# guess that uncertainty is about 2mm
sz = 2.0 + 0.0*data[:,5].astype(float)


# BUILD SYSTEM MATRIX AND REGULARIZATION MATRIX
#####################################################################

# make basemap for converting lon,lat to x,y,z coordinates
bm = create_basemap(lon,lat,resolution='i')
x,y = bm(lon,lat)

# array of observation points
obs_pnts = np.array([x,y,0.0*x]).T

# center of fault
xf,yf = bm(fault_lon,fault_lat)
top_center = np.array([xf,yf,0.0])

# discretize segment
patches = discretize_segment(top_center,fault_length,fault_width,fault_strike,fault_dip,Nx,Ny)

# evaluate the okada solution at each point for each patch
G = build_system_matrix(obs_pnts,patches,direction='thrust')

# second order tikhonov regularization matrix
L = tikhonov.tikhonov_matrix(np.arange(Nx*Ny).reshape(Nx,Ny),2)

# unravel G and displacement
Ns = G.shape[0] # number of stations
Nd = G.shape[1] # number of displacement dimensions
Nm = G.shape[2] # number of model parameters

G_flat = np.reshape(G,(Ns*Nd,Nm))
obs_disp_flat = np.reshape(obs_disp,(Ns*Nd))

# append regularization constraints to G and d
G_ext = np.vstack((G_flat,damping*L))
obs_disp_ext = np.concatenate((obs_disp_flat,np.zeros(L.shape[0])))

# COMPUTE OPTIMAL DAMPING PARAMETER

# ESTIMATE SLIP
###################################################################
# solve with non-negative least squares
slip = scipy.optimize.nnls(G_ext,obs_disp_ext)[0]

# PREDICTED DISPLACEMENT
###################################################################
disp_pred = np.einsum('ijk,k->ij',G,slip)
u_pred = disp_pred[:,0]
v_pred = disp_pred[:,1]
z_pred = disp_pred[:,2]

# PLOT THE OBSERVED AND PREDICTED DISPLACEMENTS
#####################################################################
fig,ax = plt.subplots()
bm.drawcoastlines(ax=ax,linewidth=2.0,zorder=1,color=(0.3,0.3,0.3,1.0))
bm.drawmeridians(np.arange(np.floor(bm.llcrnrlon),
                 np.ceil(bm.urcrnrlon),0.5),
                 labels=[0,0,0,1],dashes=[2,2],
                 ax=ax,zorder=1,color=(0.3,0.3,0.3,1.0))
bm.drawparallels(np.arange(np.floor(bm.llcrnrlat),
                 np.ceil(bm.urcrnrlat),0.5),
                 labels=[1,0,0,0],dashes=[2,2],
                 ax=ax,zorder=1,color=(0.3,0.3,0.3,1.0))

bm.drawmapscale(units='km',
                lat=bm.latmin+(bm.latmax-bm.latmin)/15.0,
                lon=bm.lonmin+(bm.lonmax-bm.lonmin)/4.0,
                fontsize=12,
                lon0=(bm.lonmin+bm.lonmax)/2.0,
                lat0=(bm.latmin+bm.latmax)/2.0,
                barstyle='fancy',ax=ax,
                length=50,zorder=4)

q1 = ax.quiver(x,y,u,v,scale=400.0,color='k')
q2 = ax.quiver(x,y,u_pred,v_pred,scale=400.0,color='r')
ax.quiverkey(q1,0.8,0.16,50.0,'50mm\nobserved')
ax.quiverkey(q2,0.8,0.06,50.0,'predicted')

# edges of fault trace
x1 = xf - 0.5*fault_length*np.cos((90-fault_strike)*np.pi/180.0)
y1 = yf - 0.5*fault_length*np.sin((90-fault_strike)*np.pi/180.0)
x2 = xf + 0.5*fault_length*np.cos((90-fault_strike)*np.pi/180.0)
y2 = yf + 0.5*fault_length*np.sin((90-fault_strike)*np.pi/180.0)

# surface projection of fault bottom corners
x3 = x1 + fault_width*np.cos(fault_dip*np.pi/180.0)*np.cos(fault_strike*np.pi/180.0)
y3 = y1 + fault_width*np.cos(fault_dip*np.pi/180.0)*np.sin(-fault_strike*np.pi/180.0)

# surface projection of fault bottom corners
x4 = x2 + fault_width*np.cos(fault_dip*np.pi/180.0)*np.cos(fault_strike*np.pi/180.0)
y4 = y2 + fault_width*np.cos(fault_dip*np.pi/180.0)*np.sin(-fault_strike*np.pi/180.0)

plt.plot([x1,x2],[y1,y2],'b-',lw=2,zorder=0)
plt.plot([x1,x3],[y1,y3],'b--',lw=2,zorder=0)
plt.plot([x3,x4],[y3,y4],'b--',lw=2,zorder=0)
plt.plot([x2,x4],[y2,y4],'b--',lw=2,zorder=0)
plt.show()
#####################################################################


# PLOT IN MAYAVI
#####################################################################

clim = np.min(slip),np.max(slip)
scale_factor = 100.0
mayavi.mlab.quiver3d(x,y,0*x,u,v,z,mode='arrow',color=(0,0,1),scale_factor=scale_factor)
mayavi.mlab.quiver3d(x,y,0*x,u_pred,v_pred,z_pred,mode='arrow',color=(0,1,0),scale_factor=scale_factor)
#plot_segment(1.0,[xf,yf,0.0],fault_length,fault_width,fault_strike,fault_dip)

# plot the surface
plot_segment(np.min(slip),[np.min(x),np.mean(y),0.0],np.max(y)-np.min(y),np.max(x)-np.min(x),0.0,0.0,opacity=0.5,clim=clim)
# plot slip on each fault patch
for i,p in enumerate(patches):
  plot_segment(slip[i],p[0],p[1],p[2],p[3],p[4],clim=clim)

mayavi.mlab.show()
