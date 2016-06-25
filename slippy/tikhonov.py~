#!/usr/bin/env python
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


def linear_to_array_index(val,shape,wrap=False):
  ''' 
  used in next method of IndexEnumerate
  '''
  if (not wrap) & (abs(val)+(val>0) > np.prod(shape)):
    raise IndexError(
      'positive indices must be less than the product of the shape '
      'and the absolute value of negative indices must be less than '
      'or equal to the product of the shape. received index %s for '
      'shape %s' % (val,shape))

  indices = ()
  for dimsize in shape[::-1]:
    indices += val%dimsize,
    val = val // dimsize
  return indices[::-1]


class Perturb:
  def __init__(self,v,delta=1,cast=np.asarray):
    self.v = v
    self.cast = cast
    self.N = len(v)
    self.d = delta
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
      out = [j if (i != self.k) else (j + self.d) 
             for i,j in enumerate(self.v)]   
      self.k += 1
      return self.cast(out)
      

class ArrayIndexEnumerate:
  def __init__(self,C):
    ''' 
    used in tikhonov matrix

    enumerates over the flattened elements of C and their index locations in C
  
    e.g.
  
    >> C = np.array([[1,2],[3,4]])
    >> for idx,val in IndexEnumerate(C):
    ...  print('idx: %s, val: %s' % (idx,val))   
    ...
    idx: [0, 0], val: 1
    idx: [0, 1], val: 2
    idx: [1, 0], val: 3
    idx: [1, 1], val: 4
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
      idx = linear_to_array_index(self.itr,self.C.shape)
      self.itr += 1
      return (idx,self.C[idx])


class Neighbors(ArrayIndexEnumerate):
  ''' 
  Iterator that Loops over elements in array C returning the element and its  
  neighbors
  '''
  def __init__(self,C,search='all'):
    ArrayIndexEnumerate.__init__(self,C)
    assert search in ['all','forward','backward']
    self.search = search

  def next(self):
    return self.__next__()

  def __next__(self):
    idx,val = ArrayIndexEnumerate.__next__(self)
    neighbors = np.zeros(0,dtype=self.C.dtype)
    if (self.search == 'all') | (self.search == 'forward'):
      for pert in Perturb(idx,1,cast=tuple):
        if any(i>=j for i,j in zip(pert,self.C.shape)):
          continue
        neighbors = np.append(neighbors,self.C[pert])

    if (self.search == 'all') | (self.search == 'backward'):
      for pert in Perturb(idx,-1,cast=tuple):
        if any(i<0 for i in pert):
          continue
        neighbors = np.append(neighbors,self.C[pert])
    
    return neighbors,val


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
  for neighbors,i in Neighbors(C,'forward'):
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
  for neighbors,i in Neighbors(C,'all'):
    if i == -1:
      continue
    order = sum(neighbors != -1)
    for k in neighbors:
      if k == -1:
        continue
      L[i,i] += -1.0/order  
      L[i,k]  +=  1.0/order

  return L


def tikhonov_matrix(C,n,column_no=None,dtype=None):
  ''' 
  Parameters
  ----------
    C: connectivity matrix, this can contain '-1' elements which can be used
       to break connections. 
    n: order of tikhonov regularization
    column_no: number of columns in the output matrix
    sparse_type: either 'bsr','coo','csc','csr','dia','dok','lil'

  Returns
  -------
    L: tikhonov regularization matrix saved as a csr sparse matrix

  Example
  -------
    first order regularization for 4 model parameters which are related in 2D 
    space
      >> Connectivity = [[0,1],[2,3]]
      >> L = tikhonov_matrix(Connectivity,1)
      
  '''     
  C = np.array(C)
  # check to make sure all values (except -1) are unique
  idx = C != -1
  params = C[idx] 
  unique_params = set(params)
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

  if n == 0:
    L = np.zeros((column_no,column_no),dtype=dtype)
    L =  _tikhonov_zeroth_order(C,L)

  if n == 1:
    L = np.zeros((Cdim*column_no,column_no),dtype=dtype)
    L = _tikhonov_first_order(C,L)

  if n == 2:
    L = np.zeros((column_no,column_no),dtype=dtype)
    L = _tikhonov_second_order(C,L)

  L = remove_zero_rows(L)     

  return L


