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

import numpy as np
import ctypes as ct

# ============================================================================
# CLASSES
# ============================================================================


class galva_diagram:
    def __init__(self, params):
        self.params = params 

    @property
    def simulation_params(self):
        return self.params

    def galva_lineal(self):

        if self.params.method == 'CN':
            lib_galva = ct.CDLL('./lib/galva_LCN.so')
        elif self.params.method == "BI":
            lib_galva = ct.CDLL('./lib/galva_LBI.so')


        lib_galva.galva.argtypes = [
            ct.c_bool,
            ct.c_double,
            ct.c_int,
            ct.c_int,
            ct.c_int,
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
            ct.c_double,
            ct.c_double,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]

        N = int(self.params.Npt / (2 * self.params.NPOINTS))

        res1 = (ct.c_double * N)()
        res2 = (ct.c_double * N)()

        lib_galva.galva(
            self.params.frumkin,
            self.params.g,
            self.params.Npx,
            self.params.Npt,
            self.params.NPOINTS,
            self.params.isotherm.Niso,
            self.params.D,
            self.params.ks,
            self.params.T,
            self.params.Mr,
            self.params.m,
            self.params.rho,
            self.params.Rohm,
            self.params.isotherm.Eoff,
            self.params.isotherm.Qmax,
            self.params.geo,
            self.params.Xi,
            self.params.L,
            self.params.isotherm.ai.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.params.isotherm.bi.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.params.isotherm.ci.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.params.isotherm.di.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.params.isotherm.capacity.ctypes.data_as(ct.POINTER(ct.c_double)),
            res1,
            res2,
        )

        self.SOC = np.asarray(np.frombuffer(res1, dtype=np.double, count=N))

        self.E = np.asarray(np.frombuffer(res2, dtype=np.double, count=N))
