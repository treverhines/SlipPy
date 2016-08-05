import numpy as np
cimport numpy as np
from cython cimport boundscheck,wraparound,cdivision
from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport sqrt
from libc.math cimport log
from libc.math cimport fabs
from libc.math cimport atan
from libc.stdlib cimport malloc, free

DEF TOL = 1e-8

cdef struct vector:
  double x
  double y
  double z

# internal function for Okada92
@wraparound(False)
@boundscheck(False)
@cdivision(True)
cdef vector _f(double eps,
               double eta,
               double zeta,
               double delta,
               double alpha,
               double c,
               double y,
               unsigned int output,
               unsigned int term,
               unsigned int direction) nogil:
  cdef:
    vector out
    double sindel = sin(delta)
    double cosdel = cos(delta)
    double X=0,R=0,y_bar=0,c_bar=0,d_bar=0,X11=0,X32=0,\
           X53=0,Y11=0,Y32=0,Y53=0,h=0,theta=0,logReps=0,\
           logReta=0,I1=0,I2=0,I3=0,I4=0,K1=0,K2=0,K3=0,K4=0,\
           D11=0,J1=0,J2=0,J3=0,J4=0,J5=0,J6=0,E=0,F=0,G=0,H=0,P=0,\
           Q=0,Ep=0,Fp=0,Gp=0,Hp=0,Pp=0,Qp=0,d=0,p=0,q=0,f1=0,f2=0,\
           f3 = 0
    
  d = c - zeta
  p = y*cosdel + d*sindel
  q = y*sindel - d*cosdel
  R = sqrt(eps**2 + eta**2 + q**2)

  X = sqrt(eps**2 + q**2)    
  y_bar = eta*cosdel + q*sindel
  d_bar = eta*sindel - q*cosdel
  c_bar = d_bar + zeta
  h = q*cosdel - zeta

  if fabs(q) >= TOL:
    theta = atan(eps*eta/(q*R))

  if fabs(R+eps) < TOL:
    logReps  = -log(R - eps) 
  else:
    X11 = 1.0/(R*(R + eps))
    X32 = (2*R + eps)/(R**3*(R + eps)**2)
    X53 = (8*R**2 + 9*R*eps + 3*eps**2)/(R**5*(R + eps)**3)
    logReps = log(R + eps) 

  if fabs(R + eta) < TOL:
    logReta = -log(R - eta)
  else:
    Y11 = 1.0/(R*(R + eta))
    Y32 = (2*R + eta)/(R**3*(R + eta)**2)
    Y53 = (8*R**2 + 9*R*eta + 3*eta**2)/(R**5*((R + eta)**3))
    logReta = log(R + eta)

  if fabs(cosdel) >= TOL:
    I3 = (y_bar/(cosdel*(R + d_bar)) - 1/cosdel**2*
         (logReta - sindel*log(R + d_bar)))
    if fabs(eps) >= TOL:
      I4 = ((sindel*eps)/(cosdel*(R + d_bar)) + 
             2.0/cosdel**2*atan((eta*(X + q*cosdel) + 
             X*(R + X)*sindel)/(eps*(R + X)*cosdel)))
    else:
      I4 = 0.5*(eps*y_bar)/(R + d_bar)**2

  else:
    I3 = (0.5*(eta/(R + d_bar) + (y_bar*q)/
         (R + d_bar)**2 - logReta))
    I4 = 0.5*(eps*y_bar)/(R + d_bar)**2 

  I2 = log(R + d_bar) + I3*sindel
  I1 = -eps/(R + d_bar)*cosdel - I4*sindel
  Y0 = Y11 - eps**2*Y32
  Z32 = sindel/R**3 - h*Y32
  Z53 = 3*sindel/R**5 - h*Y53
  Z0 = Z32 - eps**2*Z53

  D11 = 1.0/(R*(R + d_bar))
  J2 = eps*y_bar/(R + d_bar)*D11
  J5 = -(d_bar + y_bar**2/(R + d_bar))*D11

  if fabs(cosdel) >= TOL:
    K1 = eps/cosdel*(D11 - Y11*sindel)
    K3 = 1.0/cosdel*(q*Y11 - y_bar*D11)
    J3 = 1.0/cosdel*(K1 - J2*sindel)
    J6 = 1.0/cosdel*(K3 - J5*sindel)
  else:
    K1 = eps*q/(R + d_bar)*D11
    K3 = (sindel/(R + d_bar)*
         (eps**2*D11 - 1))
    J3 = (-eps/(R + d_bar)**2*
         (q**2*D11 - 0.5))
    J6 = (-y_bar/(R + d_bar)**2*
         (eps**2*D11 - 0.5))

  K4 = eps*Y11*cosdel - K1*sindel
  K2 = 1.0/R + K3*sindel
  J4 = -eps*Y11 - J2*cosdel + J3*sindel
  J1 = J5*cosdel - J6*sindel
  E = sindel/R - y_bar*q/R**3
  F = d_bar/R**3 + eps**2*Y32*sindel
  G = 2.0*X11*sindel - y_bar*q*X32
  H = d_bar*q*X32 + eps*q*Y32*sindel
  P = cosdel/R**3 + q*Y32*sindel
  Q = 3*c_bar*d_bar/R**5 - (zeta*Y32 + Z32 + Z0)*sindel
  Ep = cosdel/R + d_bar*q/R**3
  Fp = y_bar/R**3 + eps**2*Y32*cosdel
  Gp = 2.0*X11*cosdel + d_bar*q*X32
  Hp = y_bar*q*X32 + eps*q*Y32*cosdel
  Pp = sindel/R**3 - q*Y32*cosdel
  Qp = (3*c_bar*y_bar/R**5 + 
       q*Y32 - (zeta*Y32 + Z32 + Z0)*cosdel)

  if output == 0:
    if direction == 0:
      if term == 0:
        f1 = theta/2.0 + alpha/2.0*eps*q*Y11  
        f2 = alpha/2.0*q/R
        f3 = (1-alpha)/2.0*logReta - alpha/2.0*q**2*Y11

      if term == 1:
        f1 = -eps*q*Y11 - theta - (1- alpha)/alpha*I1*sindel
        f2 = -q/R + (1-alpha)/alpha*y_bar/(R+d_bar)*sindel
        f3 = q**2*Y11 - (1 - alpha)/alpha*I2*sindel

      if term == 2:
        f1 = (1-alpha)*eps*Y11*cosdel - alpha*eps*q*Z32
        f2 = ((1-alpha)*(cosdel/R + 2*q*Y11*sindel) - 
              alpha*c_bar*q/R**3)
        f3 = ((1-alpha)*q*Y11*cosdel - alpha*(c_bar*eta/R**3 - 
              zeta*Y11 + eps**2*Z32))

    if direction == 1:
      if term == 0:
        f1 = alpha/2.0*q/R
        f2 = theta/2.0 + alpha/2.0*eta*q*X11
        f3 = (1-alpha)/2.0*logReps - alpha/2.0*q**2*X11

      if term == 1:
        f1 = -q/R + (1 - alpha)/alpha*I3*sindel*cosdel
        f2 = (-eta*q*X11 - theta - (1-alpha)/alpha*
              eps/(R + d_bar)*sindel*cosdel)
        f3 = q**2*X11 + (1-alpha)/alpha*I4*sindel*cosdel

      if term == 2:
        f1 = ((1-alpha)*cosdel/R - q*Y11*sindel - alpha*c_bar*q/R**3)
        f2 = (1 - alpha)*y_bar*X11 - alpha*c_bar*eta*q*X32
        f3 = (-d_bar*X11 - eps*Y11*sindel - 
              alpha*c_bar*(X11 - q**2*X32))

    if direction == 2:
      if term == 0:
        f1 = -(1 - alpha)/2.0*logReta - alpha/2.0*q**2*Y11
        f2 = -(1 - alpha)/2.0*logReps - alpha/2.0*q**2*X11
        f3 = theta/2.0 - alpha/2.0*q*(eta*X11 + eps*Y11)

      if term == 1:
        f1 = q**2*Y11 - (1 - alpha)/alpha*I3*sindel**2 
        f2 = (q**2*X11 + 
              (1 - alpha)/alpha*eps / (R + d_bar)*sindel**2)
        f3 = (q*(eta*X11 + eps*Y11) - theta - 
              (1 - alpha)/alpha*I4*sindel**2)

      if term == 2:
        f1 = (-(1 - alpha)*(sindel/R + q*Y11*cosdel) - 
              alpha*(zeta*Y11 - q**2*Z32))
        f2 = ((1 - alpha)*2.0*eps*Y11 *sindel + d_bar*X11 - 
              alpha*c_bar*(X11 - q**2*X32))
        f3 = ((1 - alpha)*(y_bar*X11 + eps*Y11*cosdel) + 
              alpha*q*(c_bar*eta*X32 + eps*Z32))

  if output == 1:
    if direction == 0:
      if term == 0:
        f1 = -(1-alpha)/2.0*q*Y11 - alpha/2.0*eps**2*q*Y32
        f2 = -alpha/2.0*eps*q/R**3
        f3 = (1 - alpha)/2.0*eps*Y11 + alpha/2.0*eps*q**2*Y32
      if term == 1:
        f1 =  eps**2*q*Y32 -(1 - alpha)/alpha*J1*sindel
        f2 =  eps*q/R**3 -(1 - alpha)/alpha*J2*sindel
        f3 = -eps*q**2*Y32 -(1 - alpha)/alpha*J3*sindel
      if term == 2:
        f1 = (1 - alpha)*Y0*cosdel - alpha*q*Z0
        f2 = (-(1 - alpha) *eps*(cosdel/R**3 + 
              2.0*q*Y32*sindel) + alpha* 3.0*c_bar*eps*q/R**5)
        f3 = (-(1 - alpha)*eps*q*Y32*cosdel + 
              alpha*eps*(3.0*c_bar*eta/R**5 - zeta*Y32 - Z32 - Z0))

    if direction == 1:
      if term == 0:
        f1 = -alpha/2.0*eps*q/R**3
        f2 = -q/2.0*Y11 - alpha/2.0*eta*q/R**3
        f3 = (1 - alpha)/2.0*1.0/R + alpha/2.0*q**2/R**3

      if term == 1:
        f1 = eps*q/R**3 + (1 - alpha)/alpha*J4*sindel*cosdel
        f2 = (eta*q/R**3 + q*Y11 + 
              (1 - alpha)/alpha*J5*sindel*cosdel)
        f3 = (-q**2/R**3 + (1- alpha)/alpha*J6*sindel*cosdel)

      if term == 2:
        f1 = (-(1 - alpha)* eps/R**3*cosdel + 
             eps*q*Y32*sindel + alpha*3*c_bar*eps*q/R**5)
        f2 = (-(1 - alpha)* y_bar/R**3 + alpha*3*c_bar*eta*q/R**5)
        f3 = (d_bar/R**3 - Y0*sindel + alpha*c_bar/R**3*(1 - 3*q**2/R**2))

    if direction == 2:
      if term == 0:
        f1 = -(1 - alpha)/2.0*eps*Y11 + alpha/2.0*eps*q**2*Y32
        f2 = (-(1 - alpha)/2.0*1.0/R + alpha/2.0*q**2/R**3)
        f3 = -(1 - alpha)/2.0*q*Y11 - alpha/2.0*q**3*Y32

      if term == 1:
        f1 = (-eps*q**2*Y32 - (1 - alpha)/alpha*J4*sindel**2)
        f2 = (-q**2/R**3 - (1 - alpha)/alpha*J5*sindel**2)
        f3 = q**3*Y32 - (1 - alpha)/alpha*J6*sindel**2

      if term == 2:
        f1 = ((1 - alpha)*eps/R**3*sindel + eps*q*Y32*cosdel + 
              alpha*eps*(3*c_bar*eta/R**5 - 2.0*Z32 - Z0))
        f2 = ((1 - alpha)*2.0*Y0*sindel - d_bar/R**3 + 
              alpha*c_bar/R**3*(1 - 3.0*q**2/R**2))
        f3 = (-(1- alpha)*(y_bar/R**3 - Y0*cosdel) - 
              alpha*(3*c_bar*eta*q/R**5 - q *Z0))

  if output == 2:
    if direction == 0:
      if term == 0:
        f1 = (1-alpha)/2.0*eps*Y11*sindel + d_bar/2.0*X11 + alpha/2.0*eps*F
        f2 = alpha/2.0*E
        f3 = (1 - alpha)/2.0*(cosdel/R + q*Y11*sindel) - alpha/2.0*q*F

      if term == 1:
        f1 = -eps*F - d_bar*X11 + (1- alpha)/alpha*(eps*Y11 + J4)*sindel
        f2 = -E + (1- alpha)/alpha*(1.0/R + J5)*sindel
        f3 = q*F - (1 - alpha)/alpha*(q*Y11 - J6)*sindel

      if term == 2:
        f1 = -(1.0 - alpha)*eps*P*cosdel - alpha*eps*Q
        f2 = (2*(1.0 - alpha)*(d_bar/R**3 - Y0*sindel)*sindel - 
              y_bar/R**3*cosdel - 
              alpha*((c_bar + d_bar)/R**3*sindel - 
              eta/R**3 - 3.0*c_bar*y_bar*q/R**5))
        f3 = (-(1-alpha)*q/R**3 + 
              (y_bar/R**3 - Y0*cosdel)*sindel + 
              alpha*((c_bar + d_bar)/R**3*cosdel + 
              3.0*c_bar*d_bar*q/R**5 - (Y0*cosdel + q*Z0)*sindel))

    if direction == 1:
      if term == 0:
        f1 = alpha/2.0*E
        f2 = (1 -alpha)/2.0*d_bar*X11 + eps/2.0*Y11*sindel + alpha/2.0*eta*G
        f3 = (1 -alpha)/2.0*y_bar*X11- alpha/2.0*q*G

      if term == 1:
        f1 = -E + (1- alpha)/alpha*J1*sindel*cosdel
        f2 = -eta*G - eps*Y11*sindel + (1- alpha)/alpha*J2*sindel*cosdel
        f3 = q*G + (1- alpha)/alpha*J3*sindel*cosdel

      if term == 2:
        f1 = (-(1 - alpha)*eta/R**3 + Y0*sindel**2 - 
              alpha*((c_bar+d_bar)/R**3*sindel - 
              3*c_bar*y_bar*q/R**5))
        f2 = ((1 - alpha)*(X11 - y_bar**2*X32) - 
              alpha*c_bar*((d_bar + 2.0*q*cosdel)*X32 - y_bar*eta*q*X53))
        f3 = (eps*P*sindel + y_bar*d_bar*X32 + 
              alpha*c_bar*((y_bar + 2*q*sindel)*X32 - 
              y_bar*q**2*X53))

    if direction == 2:
      if term == 0:
        f1 = -(1 - alpha)/2.0*(cosdel/R + q*Y11*sindel) - alpha/2.0*q*F
        f2 = -(1 - alpha)/2.0*y_bar*X11 - alpha/2.0*q*G
        f3 = (1 - alpha)/2.0*(d_bar*X11 + eps*Y11*sindel) + alpha/2.0*q*H

      if term == 1:
        f1 = q*F - (1- alpha)/alpha*J1*sindel**2
        f2 = q*G - (1- alpha)/alpha*J2*sindel**2
        f3 = -q*H - (1- alpha)/alpha*J3*sindel**2

      if term == 2:
        f1 = ((1 - alpha)*(q/R**3 + Y0*sindel*cosdel) + 
              alpha*(zeta/R**3*cosdel + 
              3.0*c_bar*d_bar*q/R**5 - q*Z0*sindel))
        f2 = (-(1-alpha)*2.0*eps*P*sindel - y_bar*d_bar*X32 + 
              alpha*c_bar*((y_bar + 2.0*q*sindel)*X32 - 
              y_bar*q**2*X53))
        f3 = (-(1 - alpha)*(eps*P*cosdel - X11 + y_bar**2*X32) + 
              alpha*c_bar*((d_bar + 2.0*q*cosdel)*X32 - y_bar*eta*q*X53) + 
              alpha*eps*Q)

  if output == 3:
    if direction == 0:
      if term == 0:
        f1 = ((1 - alpha)/2.0*eps *Y11*cosdel + y_bar/2.0*X11 + 
              alpha/2.0*eps*Fp)
        f2 = alpha/2.0*Ep
        f3 = -(1 - alpha)/2.0*(sindel/R - q*Y11*cosdel) - alpha/2.0*q*Fp

      if term == 1:
        f1 = -eps*Fp - y_bar*X11 + (1 - alpha)/alpha*K1*sindel
        f2 = -Ep + (1 - alpha)/alpha*y_bar*D11*sindel
        f3 = q*Fp + (1 - alpha)/alpha*K2*sindel

      if term == 2:
        f1 = (1 - alpha)*eps*Pp*cosdel - alpha*eps*Qp
        f2 = (2*(1-alpha)*(y_bar/R**3 - Y0*cosdel)*sindel + 
              d_bar/R**3*cosdel - 
              alpha*((c_bar + d_bar)/R**3*cosdel + 
              3*c_bar*d_bar*q/R**5))
        f3 = ((y_bar/R**3 - Y0*cosdel)*cosdel - 
              alpha*((c_bar + d_bar)/R**3*sindel - 
              3*c_bar*y_bar*q/R**5 - Y0*sindel**2 + 
              q*Z0*cosdel))

      if term == 3:
        f1   = (1-alpha)*eps*Y11*cosdel - alpha*eps*q*Z32
        f2   = ((1-alpha)*(cosdel/R + 2*q*Y11*sindel) - 
                alpha*c_bar*q/R**3)
        f3   = ((1-alpha)*q*Y11*cosdel - 
                alpha*(c_bar*eta/R**3 - 
                zeta*Y11 + eps**2*Z32))

    if direction == 1:
      if term == 0:
        f1 = alpha/2.0*Ep
        f2 = (1 - alpha)/2.0*y_bar*X11 + eps/2.0*Y11*cosdel + alpha/2.0*eta*Gp
        f3 = -(1 - alpha)/2.0*d_bar*X11 - alpha/2.0*q*Gp

      if term == 1:
        f1 = -Ep - (1 - alpha)/alpha*K3*sindel*cosdel
        f2 = (-eta*Gp - eps*Y11*cosdel - 
              (1 - alpha)/alpha*eps*D11*sindel*cosdel)
        f3 = q*Gp - (1 - alpha)/alpha*K4*sindel*cosdel

      if term == 2:
        f1 = (-q/R**3 + Y0*sindel*cosdel - 
              alpha*((c_bar + d_bar)/R**3*cosdel + 
              3*c_bar*d_bar*q/R**5))
        f2 = ((1 - alpha)*y_bar*d_bar*X32 - 
              alpha*c_bar*((y_bar - 2*q*sindel)*X32 + d_bar*eta*q*X53))
        f3 = (-eps*Pp*sindel + X11 - d_bar**2*X32 - 
              alpha*c_bar*((d_bar - 2*q*cosdel)*X32 - 
              d_bar*q**2*X53))

      if term == 3:
        f1  = ((1-alpha)*cosdel/R - q*Y11*sindel - 
               alpha*c_bar*q/R**3)
        f2  = (1 - alpha)*y_bar*X11 - alpha*c_bar*eta*q*X32
        f3  = (-d_bar*X11 - eps*Y11*sindel - 
               alpha*c_bar*(X11 - q**2*X32))

    if direction == 2:
      if term == 0:
        f1 = (1 - alpha)/2.0*(sindel/R - q*Y11*cosdel) - alpha/2.0*q*Fp
        f2 = (1 - alpha)/2.0*d_bar*X11 - alpha/2.0*q*Gp
        f3 = (1 - alpha)/2.0*(y_bar*X11 + eps*Y11*cosdel) + alpha/2.0*q*Hp

      if term == 1:
        f1 = q*Fp + (1 - alpha)/alpha*K3*sindel**2
        f2 = q*Gp + (1 - alpha)/alpha*eps*D11*sindel**2
        f3 = -q*Hp + (1 - alpha)/alpha*K4*sindel**2

      if term == 2:
        f1 = (-eta/R**3 + Y0*cosdel**2 - 
              alpha*(zeta/R**3*sindel- 
              3*c_bar*y_bar*q/R**5 - 
              Y0*sindel**2 + q*Z0*cosdel))
        f2 = ((1 - alpha)*2*eps*Pp*sindel - X11 + d_bar**2*X32 - 
              alpha*c_bar*((d_bar - 2*q*cosdel)*X32 - 
              d_bar*q**2*X53))
        f3 = ((1 - alpha)*(eps*Pp*cosdel + y_bar*d_bar*X32) + 
              alpha*c_bar*((y_bar - 2*q*sindel)*X32 + d_bar*eta*q*X53) + 
              alpha*eps*Qp)

      if term == 3:
        f1 = (-(1 - alpha)*(sindel/R + q*Y11*cosdel) - 
              alpha*(zeta*Y11 - q**2*Z32))
        f2 = ((1 - alpha)*2.0*eps*Y11 *sindel + d_bar*X11 - 
              alpha*c_bar*(X11 - q**2*X32))
        f3 = ((1 - alpha)*(y_bar*X11 + eps*Y11*cosdel) + 
              alpha*q*(c_bar*eta*X32 + eps*Z32))

  out.x = f1
  out.y = f2
  out.z = f3
  return out
  

@wraparound(False)
@boundscheck(False)
@cdivision(True)
cdef vector _dc3d_k(double alpha,
                    vector pos, 
                    double c,
                    double delta,
                    double[:] strike_width,
                    double[:] dip_width,
                    double[:] U,
                    unsigned int out_type):
  cdef:
    vector out
    vector fI,fII,fIII,fIV
    unsigned int itr
    double x = pos.x
    double y = pos.y
    double z = pos.z
    double L,W
    double ux=0,uy=0,uz=0
    double pi = 3.141592653589793
    double sindel,cosdel
    double p,p_,\
           uA1,uA2,uA3,\
           uhatA1,uhatA2,uhatA3,\
           uB1,uB2,uB3,\
           uC1,uC2,uC3,\
           uD1,uD2,uD3

  delta *= pi/180
  sindel = sin(delta)
  cosdel = cos(delta)
  
  # convert the strike_width, and dip_width to L and W
  L = strike_width[1] - strike_width[0]
  W = dip_width[1] - dip_width[0]
  
  x -= strike_width[0]
  y -= dip_width[0]*cosdel  
  # find the new depth of the bottom left corner
  c -= dip_width[0]*sindel

  # iterate over slip directions
  for itr in range(3):
    p  = y*cosdel + (c - z)*sindel
    p_ = y*cosdel + (c + z)*sindel 

    fI = _f(x,p,z,delta,alpha,c,y,out_type,0,itr)     
    fII = _f(x,p - W,z,delta,alpha,c,y,out_type,0,itr)
    fIII = _f(x-L,p,z,delta,alpha,c,y,out_type,0,itr)
    fIV = _f(x-L,p - W,z,delta,alpha,c,y,out_type,0,itr)
    uA1 = fI.x - fII.x - fIII.x + fIV.x  
    uA2 = fI.y - fII.y - fIII.y + fIV.y  
    uA3 = fI.z - fII.z - fIII.z + fIV.z  
  
    fI = _f(x,p_,-z,delta,alpha,c,y,out_type,0,itr)
    fII = _f(x,p_ - W,-z,delta,alpha,c,y,out_type,0,itr)
    fIII = _f(x-L,p_,-z,delta,alpha,c,y,out_type,0,itr)
    fIV = _f(x-L,p_ - W,-z,delta,alpha,c,y,out_type,0,itr)
    uhatA1 = fI.x - fII.x - fIII.x + fIV.x  
    uhatA2 = fI.y - fII.y - fIII.y + fIV.y  
    uhatA3 = fI.z - fII.z - fIII.z + fIV.z  
  
    fI = _f(x,p,z,delta,alpha,c,y,out_type,1,itr)
    fII = _f(x,p - W,z,delta,alpha,c,y,out_type,1,itr)
    fIII = _f(x-L,p,z,delta,alpha,c,y,out_type,1,itr)
    fIV = _f(x-L,p - W,z,delta,alpha,c,y,out_type,1,itr)
    uB1 = fI.x - fII.x - fIII.x + fIV.x  
    uB2 = fI.y - fII.y - fIII.y + fIV.y  
    uB3 = fI.z - fII.z - fIII.z + fIV.z  
  
    fI = _f(x,p,z,delta,alpha,c,y,out_type,2,itr)
    fII = _f(x,p - W,z,delta,alpha,c,y,out_type,2,itr)
    fIII = _f(x-L,p,z,delta,alpha,c,y,out_type,2,itr)
    fIV = _f(x-L,p - W,z,delta,alpha,c,y,out_type,2,itr)
    uC1 = fI.x - fII.x - fIII.x + fIV.x  
    uC2 = fI.y - fII.y - fIII.y + fIV.y  
    uC3 = fI.z - fII.z - fIII.z + fIV.z  
  
    if out_type == 3:
      fI = _f(x,p,z,delta,alpha,c,y,out_type,3,itr)
      fII = _f(x,p - W,z,delta,alpha,c,y,out_type,3,itr)
      fIII = _f(x-L,p,z,delta,alpha,c,y,out_type,3,itr)
      fIV = _f(x-L,p - W,z,delta,alpha,c,y,out_type,3,itr)
      uD1 = fI.x - fII.x - fIII.x + fIV.x  
      uD2 = fI.y - fII.y - fIII.y + fIV.y  
      uD3 = fI.z - fII.z - fIII.z + fIV.z  
      ux += U[itr]/(2*pi)*(uA1 + uhatA1 + uB1 + uD1 + z*uC1)
      uy += (U[itr]/(2*pi)*((uA2 + uhatA2 + uB2 + uD2 + z*uC2)*cosdel - 
             (uA3 + uhatA3 + uB3 + uD3 + z*uC3)*sindel))
      uz += (U[itr]/(2*pi)*((uA2 + uhatA2 + uB2 - uD2 - z*uC2)*sindel + 
             (uA3 + uhatA3 + uB3 - uD3 - z*uC3)*cosdel))
    else:
      ux += U[itr]/(2*pi)*(uA1 - uhatA1 + uB1 + z*uC1)
      uy += (U[itr]/(2*pi)*((uA2 - uhatA2 + uB2 + z*uC2)*cosdel - 
             (uA3 - uhatA3 + uB3 + z*uC3)*sindel))
      uz += (U[itr]/(2*pi)*((uA2 - uhatA2 + uB2 - z*uC2)*sindel + 
             (uA3 - uhatA3 + uB3 - z*uC3)*cosdel))

  out.x = ux
  out.y = uy
  out.z = uz
  return out


@wraparound(False)
@boundscheck(False)
@cdivision(True)
cpdef tuple dc3d(double alpha,
                 double[:] pos,
                 double c,
                 double delta,
                 double[:] strike_width,
                 double[:] dip_width,
                 double[:] U):
  cdef: 
    vector vec_in,vec_out
    double[:] out_disp = np.empty((3))
    double[:,:] out_derr = np.empty((3,3))
  
  vec_in.x = pos[0]    
  vec_in.y = pos[1]   
  vec_in.z = pos[2]    

  vec_out = _dc3d_k(alpha,vec_in,c,delta,strike_width,dip_width,U,0)
  out_disp[0] = vec_out.x
  out_disp[1] = vec_out.y
  out_disp[2] = vec_out.z

  vec_out = _dc3d_k(alpha,vec_in,c,delta,strike_width,dip_width,U,1)
  out_derr[0,0] = vec_out.x
  out_derr[0,1] = vec_out.y
  out_derr[0,2] = vec_out.z

  vec_out = _dc3d_k(alpha,vec_in,c,delta,strike_width,dip_width,U,2)
  out_derr[1,0] = vec_out.x
  out_derr[1,1] = vec_out.y
  out_derr[1,2] = vec_out.z

  vec_out = _dc3d_k(alpha,vec_in,c,delta,strike_width,dip_width,U,3)
  out_derr[2,0] = vec_out.x
  out_derr[2,1] = vec_out.y
  out_derr[2,2] = vec_out.z
  
  return 0,np.asarray(out_disp),np.asarray(out_derr)
  
