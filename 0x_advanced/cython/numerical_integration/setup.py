# Setup file for compiling the cython code

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize(['integration.pyx',
                               'integration_v2.pyx'],
                             )
     )
