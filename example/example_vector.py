#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cosinv.basis
import cosinv.transform

def plot_basis(B,ax,**kwargs):
  B = B[:2,:2]
  X,Y = 0*B
  U,V = B[:,0],B[:,1]
  ax.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1,**kwargs)
  ax.set_xlim([-1,2])
  ax.set_ylim([-1,2])
    
basis1 = np.array([[[ 1.0,  0.1],
                    [-0.1,  1.0]]])
basis2 = cosinv.basis.cardinal_basis(1,2)

point1 = np.array([[1.1,1.5]])
# change from basis1 to basis2 
point2 = cosinv.basis.change_basis(point1,basis1,basis2)
# change from basis2 to basis1
point3 = cosinv.basis.change_basis(point2,basis2,basis1)
# change from basis1 to cardinal
point4 = cosinv.basis.cardinal_components(point1,basis2)
print(point1)
print(point3)
print(point4)

print(cosinv.basis.fold_basis(cosinv.basis.flatten_basis(basis1)))
quit()
fig,ax = plt.subplots()
ax.set_title('cardinal bases space')
plot_basis(card_basis,ax,color='k')
plot_basis(my_basis,ax,color='m')

plt.show()
