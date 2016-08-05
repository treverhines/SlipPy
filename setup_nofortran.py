#!/usr/bin/env python
if __name__ == '__main__':
  from numpy.distutils.core import setup
  from numpy.distutils.extension import Extension
  from Cython.Build import cythonize
  ext = []
  ext += [Extension(name='slippy.cdc3d',sources=['slippy/cdc3d.pyx'])]
  setup(
     name='SlipPy',
     packages=['slippy'],
     scripts=['exec/slippy','exec/plot_slippy'],
     version='0.1.0',
     description='module for inverting coseismic slip from GPS and InSAR data',
     author='Trever Hines',
     author_email='hinest@umich.edu',
     url='https://github.com/treverhines/SlipPy',
     ext_modules=cythonize(ext))
