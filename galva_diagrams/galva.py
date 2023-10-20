#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galva_simul (https://github.com/aruderman/galva_simul/).
# Copyright (c) 2021-2023, Maximiliano Gavilán y Andrés Ruderman
# License: MIT
#   Full Text: https://github.com/aruderman/galva_simul/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""galva_diagram class of galva_simul."""

# ============================================================================
# IMPORTS
# ============================================================================

import ctypes as ct
import os
import pathlib
import sysconfig

import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# CLASSES
# ============================================================================


class galva_diagram:
    def __init__(self, params, spline):
        self.params = params 
        self.spline = spline 

    @property
    def simulation_params(self):
        return self.params

    def galva_calc(self):

        #lib_galva = ct.CDLL(PATH / "lib" / "galva_PIBB" / ".so")
        lib_galva = ct.CDLL('../../paralelo/diagramas/galva_paralelo_chequeo.so')

        lib_galva.galva.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]

        N = int(self.params.NL * self.params.NXi)

        res1 = (ct.c_double * N)()
        res2 = (ct.c_double * N)()
        res3 = (ct.c_double * N)()

        lib_galva.galva(
            self.params.Npx,
            self.params.Npt,
            self.params.NPOINTS,
            self.params.Niso,
            self.params.Xif,
            self.params.Xi0,
            self.params.NXi,
            self.params.Lf,
            self.params.L0,
            self.params.NL,
            self.params.D,
            self.params.ks,
            self.params.T,
            self.params.Mr,
            self.params.m,
            self.params.rho,
            self.params.Rohm,
            self.params.Eoff,
            self.spline.Qmax,
            self.params.geo,
            self.spline.ai.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.spline.bi.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.spline.ci.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.spline.di.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.spline.capacity.ctypes.data_as(ct.POINTER(ct.c_double)),
            res1,
            res2,
            res3,
        )

        logL = np.asarray(np.frombuffer(res1, dtype=np.double, count=N))

        logXi = np.asarray(np.frombuffer(res2, dtype=np.double, count=N))

        SOC = np.asarray(np.frombuffer(res3, dtype=np.double, count=N))

        return logL, logXi, SOC
