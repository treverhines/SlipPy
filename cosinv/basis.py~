#!/usr/bin/env python
import numpy as np
import transform

def cardinal_basis(shape):
  ''' 
  Returns the cardinal basis vectors in the cardinal basis reference 
  frame.  This is just an array of N identity matrices
  '''
  D = shape[-1]
  basis = np.zeros(shape+(D,))
  for d in range(D):  
    basis[...,d,d] = 1.0

  return basis
  
def cardinal_components(components,basis):
  ''' 
  return the vector components in the cardinal coordinate system
  '''
  # transpose the second and third axes of basis so that each column 
  # is a basis vector rather than each row being a basis vector
  components = np.asarray(components)
  basis = np.asarray(basis)

  basis = np.einsum('...ij->...ji',basis)
  return np.einsum('...ij,...j->...i',basis,components)


def change_basis(components,basis,new_basis):
  ''' 
  Returns the components after transforming the basis from 'basis' to 
  'new_basis'
  '''
  # transpose the second and third axes of basis so that each column 
  # is a basis vector rather than each row being a basis vector
  components = np.asarray(components)
  basis = np.asarray(basis)
  new_basis = np.asarray(new_basis)
  
  basis = np.einsum('...ij->...ji',basis)
  new_basis = np.einsum('...ij->...ji',new_basis)
  # invert each new collection of basis vectors
  new_basis_inv = np.linalg.inv(new_basis)
  return np.einsum('...ij,...jk,...k->...i',new_basis_inv,basis,components)

def flatten_basis(basis):
  ''' 
  flattens the basis vectors from a (N,D,D) array to a (N*D,D) array.
  '''
  basis = np.asarray(basis)
  N = basis.shape[0]
  D = basis.shape[1]
  basis = basis.reshape((N*D,D))
  return basis

def flatten(components,basis):
  ''' 
  flattens the components from a (N,D) array to a (N*D,) array
  flattens the basis from a (N,D,D) array to a (N*D,D) array
  '''
  components = np.asarray(components)
  N,D = components.shape
  components = components.reshape((N*D,))
  basis = basis.reshape((N*D,D))
  return components,basis
  
def fold_basis(basis):
  ''' 
  folds the basis from a (N*3,3) array to a (N,3,3) array
  '''
  D = basis.shape[1]
  basis = basis.reshape((-1,D,D))
  return basis

def fold(components,basis):
  ''' 
  folds the components from a (N*3,) array to a (N,3) array
  '''
  D = basis.shape[1]
  basis = basis.reshape((-1,D,D))
  components = components.reshape((-1,D))
  return components,basis
  
class VectorField:
  def __init__(self,components,basis=None):
    ''' 
    Parmeters
    ---------
      components : (N,3) array 
      
      basis : (N,3,3) array, optional
    '''  
    components = np.asarray(components)
    N = components.shape[0]
    if basis is None:
      basis = cardinal_basis(N)
    else:
      basis = np.asarray(basis)  
      
    self.basis = basis
    self.components = components
    return
    
  def rotate_basis(self,angle,axis):
    if axis == 0:
      T = transform.basis_transform_x(angle)
    elif axis == 1:
      T = transform.basis_transform_y(angle)
    elif axis == 2:
      T = transform.basis_transform_z(angle)
      
    self.components = T(self.components)
    self.basis = T.inverse()(self.components)
    return 

  def stretch_basis(self,factor,axis):
    S = [0.0,0.0,0.0]
    S[axis] = factor
    T = transform.basis_stretch(S)
    self.components = T(self.components)
    self.basis = T.inverse()(self.components)
    return 
    
  def get_cardinal_components(self):  
    return cardinal_components(self.components,self.basis)
    
  def get_basis(self):
    return np.array(self.basis,copy=True)

  def get_components(self):
    return np.array(self.basis,copy=True)
    
    
