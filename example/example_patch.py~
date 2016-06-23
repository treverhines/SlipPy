#!/usr/bin/env python
import cosinv.patch
import matplotlib.pyplot as plt

### patch specifications
#####################################################################
strike = 20.0 # degrees
dip = 45.0 # degrees
length = 5.0 
width = 5.0
pos = [1.0,2.0,0.0] # top center of patch
#####################################################################

# initiate patch
P = cosinv.patch.Patch(pos,length,width,strike,dip)
# plot patch
fig,ax = plt.subplots()
ax.set_ylim((-10,10))
ax.set_xlim((-10,10))
ax.set_aspect('equal')

P.draw_patch(ax=ax)

### discretize patch
#####################################################################
Nw = 10
Nl = 10
# create a list of patches
Ps = P.discretize(Nw,Nl) 

# plot patches and color by depth of center
depth = [i.patch_to_data([0.5,0.5,0.0])[2] for i in Ps]

fig,ax = plt.subplots()
ax.set_ylim((-10,10))
ax.set_xlim((-10,10))
ax.set_aspect('equal')

cosinv.patch.draw_patches(Ps,ax=ax,colors=depth)

### add another segment and specify pos with the top left corner
#####################################################################
strike = 120.0 # degrees
dip = 70.0 # degrees
length = 10.0 
width = 5.0
pos = [-2.0,-1.0,0.0] # top center of patch
P = cosinv.patch.Patch(pos,length,width,strike,dip,pos_patch=[0.0,1.0,0.0])

# append new discretized patches
Ps += P.discretize(Nw,Nl)

depth = [i.patch_to_data([0.5,0.5,0.0])[2] for i in Ps]

fig,ax = plt.subplots()
ax.set_ylim((-10,10))
ax.set_xlim((-10,10))
ax.set_aspect('equal')

cosinv.patch.draw_patches(Ps,ax=ax,colors=depth)
plt.show()
