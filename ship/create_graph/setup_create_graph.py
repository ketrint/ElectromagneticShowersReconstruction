from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

import numpy as np
ext_modules = [
    Extension(
        "create_graph",
        ["create_graph.pyx"],
        libraries=["m"],
        extra_compile_args=['-fopenmp', '-O3', "-ffast-math", "-march=native"],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()],
       language="c++"
   )
]

setup(
    name='create_graph',
    ext_modules=cythonize(ext_modules, annotate=True),
    # gdb_debug=True
)
