#!/usr/bin/env python
import numpy as np
import matplotlib.axes
import matplotlib.patches
from matplotlib.quiver import Quiver as _Quiver
from matplotlib.collections import EllipseCollection
from matplotlib.backends import pylab_setup
from matplotlib.pyplot import sci
from matplotlib.pyplot import gca
_backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()


def compute_abphi(sigma_x,sigma_y,rho):
  n = len(sigma_x)
  a = []
  b = []
  phi = []
  for i in range(n):
    if (np.ma.is_masked(sigma_x[i]) | 
        np.ma.is_masked(sigma_y[i]) | 
        np.ma.is_masked(rho[i])):
      a += [0.0]
      b += [0.0]
      phi += [0.0]
      continue

    sigma_xy = rho[i]*sigma_x[i]*sigma_y[i]
    cov_mat = np.array([[sigma_x[i]**2,sigma_xy],
                        [sigma_xy,sigma_y[i]**2]])
    val,vec = np.linalg.eig(cov_mat)
    maxidx = np.argmax(val)
    minidx = np.argmin(val)
    a += [np.sqrt(val[maxidx])]
    b += [np.sqrt(val[minidx])]
    phi += [np.arctan2(vec[:,maxidx][1],vec[:,maxidx][0])]
    
  a = np.array(a)
  b = np.array(b)
  phi = np.array(phi)*180/np.pi
  return a,b,phi


def quiver(*args, **kw):
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False               
    washold = ax.ishold()
    hold = kw.pop('hold', None)
    if hold is not None:
        ax.hold(hold)

    try:
      if not ax._hold:
        ax.cla()

      q = Quiver(ax, *args, **kw)
      ax.add_collection(q, autolim=True)
      ax.autoscale_view()
      draw_if_interactive()

    finally:
      ax.hold(washold)

    sci(q)
    return q


class Quiver(_Quiver):
  def __init__(self,ax,*args,**kwargs):
    if kwargs.has_key('sigma'):
      kwargs['scale_units'] = 'xy'
      kwargs['angles'] = 'xy'
      if not kwargs.has_key('scale'):
        kwargs['scale'] = 1.0

    su,sv,rho = kwargs.pop('sigma',(None,None,None))
        
    self.ellipse_kwargs = {}
    self.ellipse_kwargs['zorder'] = kwargs.get('zorder')
    self.ellipse_kwargs['edgecolors'] = kwargs.pop('ellipse_edgecolors','k')
    self.ellipse_kwargs['facecolors'] = kwargs.pop('ellipse_facecolors','none')
    self.ellipse_kwargs['linewidths'] = kwargs.pop('ellipse_linewidths',1.0)
    _Quiver.__init__(self,
                     ax,
                     *args,
                     **kwargs)

    if (su is not None) & (sv is not None):
      self._update_ellipsoids(su,sv,rho)


  def _update_ellipsoids(self,su,sv,rho):
    self.scale_units = 'xy'
    self.angles = 'xy'
    tips_x = self.X + self.U/self.scale
    tips_y = self.Y + self.V/self.scale
    tips = np.array([tips_x,tips_y]).transpose()

    a,b,angle = compute_abphi(su,sv,rho)

    width = 2.0*a/self.scale
    height = 2.0*b/self.scale
    if hasattr(self,'ellipsoids'):
      self.ellipsoids.remove()

    self.ellipsoids = EllipseCollection(width,
                                        height,
                                        angle,
                                        units=self.scale_units,
                                        offsets = tips,
                                        transOffset=self.ax.transData,
                                        **self.ellipse_kwargs)

    self.ax.add_collection(self.ellipsoids)

  def set_UVC(self,u,v,C=None,sigma=None):
    if C is None:
      _Quiver.set_UVC(self,u,v)
    else:
      _Quiver.set_UVC(self,u,v,C)

    if (sigma is not None):
      su = sigma[0]
      sv = sigma[1]
      rho = sigma[2]
      self._update_ellipsoids(su,sv,rho)

