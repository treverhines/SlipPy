#!/usr/bin/env python
import numpy as np
import slippy.transform
import warnings

def ginv(A):
  ''' 
  Returns the generalized inverse of A. Unlike np.linalg.pinv, this 
  fuction supports broadcasting. 
  
  Parameters
  ----------
    A : (...,N,M) array
    
    damping : float, optional
    
  Returns
  -------
    out : (...,M,N) array
    
  ''' 
  AtA = np.einsum('...ji,...jk->...ik',A,A)
  # check condition number
  cond = np.linalg.cond(AtA)
  if np.any(cond > 1e10):
    warnings.warn('matrix is highly ill-conditioned')
  
  AtAinv = np.linalg.inv(AtA)
  out = np.einsum('...ij,...kj->...ik',AtAinv,A)
  return out
    
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
  return the vector components in the cardinal coordinate

  Parameters
  ----------
    components : (...,R) array
      components for each of the basis vector 
    
    basis : (...,R,D) array
      basis vectors defined with respect to the cardinal coordinate 
      system
      
  Returns
  -------
    components : (...,D) array
      components in the cardinal basis system
  
  '''
  # transpose the second and third axes of basis so that each column 
  # is a basis vector rather than each row being a basis vector
  components = np.asarray(components)
  basis = np.asarray(basis)
  new_basis_shape = basis.shape[:-2] + (basis.shape[-1],)
  new_basis = cardinal_basis(new_basis_shape)
  out = change_basis(components,basis,new_basis) 
  return out

def change_basis(components,basis,new_basis):
  ''' 
  returns the components with respect to a new basis system
  
  Parameters
  ----------
    components : (...,R) array
      basis components 
    
    basis : (...,R,D) array
      basis vectors define with respect to the cardinal coordinate 
      system
          
    new_basis : (...,P,D) array
      new basis vectors define with respect to the cardinal coordinate 
      system
    
  Notes
  -----
    if the new basis is not linearly independent then a singular 
    matrix error will be raised. If the new basis is almost not 
    linearly independent then a warning will be raised

  '''
  # transpose the second and third axes of basis so that each column 
  # is a basis vector rather than each row being a basis vector
  components = np.asarray(components)
  basis = np.asarray(basis)
  new_basis = np.asarray(new_basis)
  
  basis = np.einsum('...ij->...ji',basis)
  new_basis = np.einsum('...ij->...ji',new_basis)
  # invert each new collection of basis vectors
  new_basis_inv = ginv(new_basis)
  out = np.einsum('...ij,...jk,...k->...i',new_basis_inv,basis,components)
  return out
