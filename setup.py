from numpy.distutils.core import setup, Extension
# -g compiles with debugging information.
# -O0 means compile with no optimization, try -O3 for blazing speed
compile_args = ['-O3']
ext = []
ext.append(Extension('DC3D',
                     sources = ['slippy/dc3d/DC3D.f',
                                'slippy/dc3d/DC3D.pyf'],
                     extra_compile_args=compile_args))
setup(
   name='SlipPy',
   packages=['slippy','slippy/dc3d'],
   scripts=['exec/SlipPy','exec/PlotSlipPy'],
   version='0.1.0',
   description='module for inverting coseismic slip from GPS and InSAR data',
   author='Trever Hines',
   author_email='hinest@umich.edu',
   url='https://github.com/treverhines/SlipPy',
   ext_modules=ext)
