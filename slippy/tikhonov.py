#!/usr/bin/env python
from __future__ import division
import numpy as np

def remove_zero_rows(M):
  ''' 
  used in tikhonov_matrix
  '''
  out = np.zeros((0,M.shape[1]))
  for m in M:
    if np.any(m):
      out = np.vstack((out,m))

  return out


def flat_to_array_index(idx,shape):
  ''' 
  converts an index for a flattened array to the subscript indices 
  which call the same element in the unflattened array

  Parameters
  ----------

    idx : int
      index for flat array

    shape : tuple
      shape of folded array
    
  '''
  indices = ()
  for dimsize in shape[::-1]:
    indices += idx%dimsize,
    idx = idx // dimsize
  return indices[::-1]


class Perturb:
  ''' 
  Iterator which takes an iterable, V, and returns Vk on iteration k, 
  where Vk is defined as:
  
    V[i] + delta     if i = k
    V[i]             else
  
  Parameters
  ----------
    v : iterable
    delta : optional

  '''
  def __init__(self,v,delta=1):
    self.v = v
    self.N = len(v)
    self.delta = delta
    self.k = 0 

  def __iter__(self):
    return self

  # this is for python 2 compatibility
  def next(self):
    return self.__next__()

  def __next__(self):
    if self.k == self.N:
      raise StopIteration

    else:
      out = [j if (i != self.k) else (j + self.delta) for i,j in enumerate(self.v)]   
      self.k += 1
      return out
      

class ArrayIndexEnumerate:
  def __init__(self,C):
    ''' 
    enumerates over the flattened elements of C and their array indices

    Example
    -------
      >> C = np.array([[1,2],[3,4]])
      >> for idx,val in ArrayIndexEnumerate(C):
      ...  print('idx: %s, val: %s' % (idx,val))   
      ...
      idx: (0, 0), val: 1
      idx: (0, 1), val: 2
      idx: (1, 0), val: 3
      idx: (1, 1), val: 4
    '''
    self.C = np.asarray(C)
    self.itr = 0

  def __iter__(self):
    return self

  # this is for python 2 compatibility
  def next(self):
    return self.__next__()

  def __next__(self):
    if self.itr == self.C.size:
      raise StopIteration

    else:
      idx = flat_to_array_index(self.itr,self.C.shape)
      self.itr += 1
      return (idx,self.C[idx])


class ForwardNeighbors:
  ''' 
  iterates over the flattened elements of C, returning the element and 
  its forward looking neighbors for each axis.
    
  Example
  -------
    >> C = np.array([[1,2,3],
                     [4,5,6]])  
    >> for i in ForwardNeighbors(C):
         print(i)
    [4,2],1
    [5,3],2
    [6],3
    [5],4
    [6],5
    [],6
  '''
  def __init__(self,C):
    self.C = np.asarray(C)
    self.itr = 0
    
  def next(self):
    return self.__next__()
    
  def __iter__(self):
    return self

  def __next__(self):  
    if self.itr == self.C.size:
      raise StopIteration
    
    else:
      out = []
      idx = flat_to_array_index(self.itr,self.C.shape)
      for idx_pert in Perturb(idx,1):
        # this checks if the perturbed indices are valid for C
        if any(i>=j for i,j in zip(idx_pert,self.C.shape)):
          continue
        out += [self.C[tuple(idx_pert)]] 

      self.itr += 1
      return out,self.C[idx]
      

class BackwardNeighbors:
  ''' 
  iterates over the flattened elements of C, returning the element and 
  its backward looking neighbors for each axis.
    
  Example
  -------
    >> C = np.array([[1,2,3],
                     [4,5,6]])  
    >> for i in ForwardNeighbors(C):
         print(i)
    [],1
    [1],2
    [2],3
    [1],4
    [2,4],5
    [3,5],6
  '''
  def __init__(self,C):
    self.C = np.asarray(C)
    self.itr = 0
    
  def next(self):
    return self.__next__()
    
  def __iter__(self):
    return self

  def __next__(self):  
    if self.itr == self.C.size:
      raise StopIteration
    
    else:
      out = []
      idx = flat_to_array_index(self.itr,self.C.shape)
      for idx_pert in Perturb(idx,-1):
        # this checks if the perturbed indices are valid for C
        if any(i<0 for i in idx_pert):
          continue
        out += [self.C[tuple(idx_pert)]] 

      self.itr += 1
      return out,self.C[idx]
      

class Neighbors:      
  ''' 
  iterates over the flattened elements of C, returning the element and 
  its forward and backward looking neighbors for each axis.
    
  Example
  -------
    >> C = np.array([[1,2,3],
                     [4,5,6]])  
    >> for i in ForwardNeighbors(C):
         print(i)
    [4,2],1
    [5,3,1],2
    [6,2],3
    [5,1],4
    [6,2,4],5
    [3,5],6
  '''
  def __init__(self,C):
    self.C = np.asarray(C)
    self.itr = 0
    
  def next(self):
    return self.__next__()
    
  def __iter__(self):
    return self

  def __next__(self):  
    if self.itr == self.C.size:
      raise StopIteration
    
    else:
      out = []
      idx = flat_to_array_index(self.itr,self.C.shape)
      for idx_pert in Perturb(idx,1):
        # this checks if the perturbed indices are valid for C
        if any(i>=j for i,j in zip(idx_pert,self.C.shape)):
          continue
        out += [self.C[tuple(idx_pert)]] 

      for idx_pert in Perturb(idx,-1):
        # this checks if the perturbed indices are valid for C
        if any(i<0 for i in idx_pert):
          continue
        out += [self.C[tuple(idx_pert)]] 

      self.itr += 1
      return out,self.C[idx]
      

def _tikhonov_zeroth_order(C,L):
  ''' 
  used in tikhonov_matrix
  '''
  for val in C.flat:
    if (val == -1):
      continue
    L[val,val] = 1

  return L


def _tikhonov_first_order(C,L):
  ''' 
  used in tikhonov_matrix
  '''
  Lrow = 0
  for neighbors,i in ForwardNeighbors(C):
    if i == -1:
      continue
    for k in neighbors:
      if k == -1:
        continue
      L[Lrow,i] += -1
      L[Lrow,k] += 1
      Lrow += 1

  return L 


def _tikhonov_second_order(C,L):
  ''' 
  used in tikhonov_matrix
  '''
  for nbrs,i in Neighbors(C):
    if i == -1:
      continue
    denom = len([n for n in nbrs if n != -1])
    for k in nbrs:
      if k == -1:
        continue
      L[i,i] += -1.0/denom 
      L[i,k]  +=  1.0/denom

  return L


def tikhonov_matrix(C,order,column_no=None):
  ''' 
  Parameters
  ----------
    C: int array
      connectivity matrix, this can contain '-1' elements which can be used
      to break connections. 
      
    order: int
      order of tikhonov regularization. can be either 0, 1, or 2

    column_no: int
      number of columns in the output matrix

  Returns
  -------
    L: tikhonov regularization matrix 

  Example
  -------
    >> Connectivity = [[0,1],[2,3]]
    >> L = tikhonov_matrix(Connectivity,1)

    array([[-1.,  0.,  1.,  0.],
           [-1.,  1.,  0.,  0.],
           [ 0., -1.,  0.,  1.],
           [ 0.,  0., -1.,  1.]])

  '''     
  C = np.asarray(C,dtype=int)
  # check to make sure all values (except -1) are unique
  idx = C != -1
  params = C[idx] 
  unique_params = np.unique(params)
  assert len(params) == len(unique_params), (
         'all values in C, except for -1, must be unique')

  Cdim = len(np.shape(C))

  if np.size(C) == 0:
    max_param = 0

  else: 
    max_param = np.max(C) + 1

  if column_no is None:
    column_no = max_param

  assert column_no >= max_param, (
         'column_no must be at least as large as max(C)')

  if order == 0:
    L = np.zeros((column_no,column_no))
    L =  _tikhonov_zeroth_order(C,L)

  if order == 1:
    L = np.zeros((Cdim*column_no,column_no))
    L = _tikhonov_first_order(C,L)

  if order == 2:
    L = np.zeros((column_no,column_no))
    L = _tikhonov_second_order(C,L)

  L = remove_zero_rows(L)     

  return L


