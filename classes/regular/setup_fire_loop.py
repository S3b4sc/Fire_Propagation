from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import numpy

ext_modules = [
    Pybind11Extension(
        "fire_plus_loop",                # nombre del módulo
        ["fire_plus_loop.cpp"],          # archivos fuente
        include_dirs=[
            numpy.get_include()
            # pybind11 y xtensor-python ya vienen incluidos en conda
        ],
        #language="c++",
        define_macros=[("VERSION_INFO", "0.0.1")],
        cxx_std=17,
        #extra_compile_args=["-std=c++17"]
    ),
]

setup(
    name="fire_plus_loop",
    version="0.0.1",
    author="Sebastian",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
