#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galva_simul (https://github.com/aruderman/galva_simul/).
# Copyright (c) 2022-2023, Maximiliano Gavilan and Andrés Ruderman
# License: MIT
#   Full Text: https://github.com/aruderman/galva_simul/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""A Python library with C++ extensions to make diagrams."""

# ============================================================================
# CONSTANTS
# ============================================================================

__author__ = """Maximiliano Gavilan and Andrés Ruderman"""
__email__ = "andres.ruderman@gmail.com"
__version__ = "0.1.1"


# ============================================================================
# IMPORTS
# ============================================================================

# spline
from .spline import SplineParams  # noqa

# diagram calculations
from .Simulation import GalvanostaticMap, GalvanostaticProfile
