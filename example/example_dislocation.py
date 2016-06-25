#!/usr/bin/env python
''' 

this script demonstrates the usage of the functions in cosinv.okada 
which are dislocation and patch_dislocation. Both functions calculate 
the displacements resulting from a dislocation in an elastic halfspace

'''
import numpy as np
import cosinv.patch
import cosinv.okada
import matplotlib.pyplot as plt

### patch specifications
#####################################################################
strike = 20.0 # degrees
dip = 80.0 # degrees
length = 5.0
width = 5.0
pos = [1.0,-2.0,0.0] # top center of patch
slip = [1.0,0.0,0.0] # left-lateral slip

### observation points
#####################################################################
x = np.linspace(-10,10,50)
y = np.linspace(-10,10,50)
xg,yg = np.meshgrid(x,y)
xf,yf = xg.ravel(),yg.ravel()
zf = 0*xf
obs = np.array([xf,yf,zf]).T

### create patch instance
#####################################################################
p = cosinv.patch.Patch(pos,length,width,strike,dip)

### compute displacements
#####################################################################

# compute displacement by passing all patch geometry parameters  
disp1,derr1 = cosinv.okada.dislocation(obs,slip,pos,length,width,strike,dip)

# compute displacement by passing a patch instance
disp2,derr2 = cosinv.okada.patch_dislocation(obs,slip,p)

### plot solution
#####################################################################
fig,ax = plt.subplots()
ax.set_aspect('equal')
ax.quiver(obs[:,0],obs[:,1],disp1[:,0],disp1[:,1],scale=5.5,color='k')
ax.quiver(obs[:,0],obs[:,1],disp2[:,0],disp2[:,1],scale=5.5,color='b')

# draw patch polygon
ax.add_artist(p.get_polygon(zorder=0))

plt.show()
