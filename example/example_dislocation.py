#!/usr/bin/env python
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
pos = [0.0,0.0,0.0] # top center of patch
#####################################################################

### observation points
x = np.linspace(-10,10,50)
y = np.linspace(-10,10,50)
xg,yg = np.meshgrid(x,y)
xf,yf = xg.ravel(),yg.ravel()
zf = 0*xf
obs = np.array([xf,yf,zf]).T

# make patch
p = cosinv.patch.Patch(pos,length,width,strike,dip,pos_patch=[0.0,1.0,0.0])

# compute displacement for slip on the patch
disp,derr = cosinv.okada.dislocation(obs,
                                     [1.0,0.0,0.0], 
                                     p.patch_to_user([0.5,1.0,0.0]),
                                     p.length,p.width,
                                     p.strike,p.dip)
fig,ax = plt.subplots()
ax.set_aspect('equal')
ax.quiver(obs[:,0],obs[:,1],disp[:,0],disp[:,1])
ax.add_artist(p.get_polygon(zorder=0))
plt.show()
