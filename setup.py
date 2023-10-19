#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galva_simul (https://github.com/aruderman/galva_simul/).
# Copyright (c) 2021-2023, Maximiliano Gavilán and Andrés Ruderman
# License: MIT
#   Full Text: https://github.com/aruderman/galva_simul/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Installation of galva_simul."""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib
import sysconfig

from setuptools import find_packages

from distutils.core import Extension, setup  # noqa: I100

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

REQUIREMENTS = [
    "matplotlib",
    "numpy",
    "pandas",
    "ctypes",
    "csaps",
]

galva_simul/galva_diagrams/lib/

CFLAGS = sysconfig.get_config_var("CFLAGS").split()
CFLAGS += ["-O3", "-march=native", "-fopenmp", "-fPIC"]
C_MODS = [
    Extension(
        "galva_simul/galva_diagrams/lib/galva_paralelo2",
        sources=["galva_simul/galva_diagrams/lib/galva_paralelo2.cpp"],
        extra_compile_args=CFLAGS,
    )
]

with open(PATH / "galva_simul" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break

with open("README.md") as file_readme:
    LONG_DESCRIPTION = file_readme.read()

SHORT_DESCRIPTION = (
    "A Python library with C extensions..."
)


# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="galva_simul",
    version=VERSION,
    description=SHORT_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    author="Maximiliano Gavilán and Andrés Ruderman",
    author_email="andres.ruderman@unc.edu.ar",
    url="https://github.com/aruderman/galva_simul",
    license="MIT",
    install_requires=REQUIREMENTS,
    setup_requires=REQUIREMENTS,
    keywords=["galva_simul"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
    ],
    ext_modules=C_MODS,
)
