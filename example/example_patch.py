#!/usr/bin/env python
''' 
this script demonstrates how to generate, discretize, and view Patch 
instances
'''
import cosinv.patch
import numpy as np
import matplotlib.pyplot as plt

# plot patch
fig,ax = plt.subplots()
ax.set_ylim((-10,10))
ax.set_xlim((-10,10))
ax.set_aspect('equal')

### patch specifications
#####################################################################
strike = 20.0 # degrees
dip = 45.0 # degrees
length = 5.0 
width = 5.0
pos = [-1.0,2.0,0.0] # top center of patch
#####################################################################

### initiate patch
#####################################################################
P = cosinv.patch.Patch(pos,length,width,strike,dip)

### discretize patch
#####################################################################
Nw = 10
Nl = 10

# create a list of discretized patches
Ps = P.discretize(Nw,Nl) 

# the patch_to_user method converts a point in the patch coordinate 
# system to the user coordinate system.  The center of each patch
# is [0.5,0.5,0.0] in the patch coordinate system and here I am finding
# where each of those points is in the user coordinate system
patch_centers = np.array([i.patch_to_user([0.5,0.5,0.0]) for i in Ps])
patch_depths = patch_centers[:,2]

# this function plots all patches in Ps and colors them by depth. pc 
# is a polygon collection which i use to generate a colorbar
pc = cosinv.patch.draw_patches(Ps,ax=ax,colors=patch_depths)
fig.colorbar(pc,ax=ax)

### initiate new patch
#####################################################################
strike = 120.0 # degrees
dip = 70.0 # degrees
length = 10.0 
width = 5.0
pos = [-2.0,-1.0,0.0] # top center of patch

# the pos_patch keyword argument indicates where pos is on the fault 
# patch with respect to the patch coordinate system. [0.0,1.0,0.0] is 
# the top left side of the fault when viewed from the side the patch 
# dips towards
P = cosinv.patch.Patch(pos,length,width,strike,dip,
                       pos_patch=[0.0,1.0,0.0])

# get a polygon artist for the patch and add it to the axes
ax.add_artist(P.get_polygon(facecolor='green'))
plt.show()
