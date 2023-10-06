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
    def __init__(
        self,
        so_path,
        Npx,
        Npt,
        NPOINTS,
        Niso,
        Xif,
        Xi0,
        NXi,
        Lf,
        L0,
        NL,
        D,
        ks,
        T,
        Mr,
        m,
        rho,
        Rohm,
        Eoff,
        Qmax,
        ai,
        bi,
        ci,
        di,
        capacidad,
    ):
        self.so_path = so_path
        self.Npx = Npx
        self.Npt = Npt
        self.NPOINTS = NPOINTS
        self.Niso = Niso
        self.Xif = Xif
        self.Xi0 = Xi0
        self.NXi = NXi
        self.Lf = Lf
        self.L0 = L0
        self.NL = NL
        self.D = D
        self.ks = ks
        self.T = T
        self.Mr = Mr
        self.m = m
        self.rho = rho
        self.Rohm = Rohm
        self.Eoff = Eoff
        self.Qmax = Qmax
        self.ai = ai
        self.bi = bi
        self.ci = ci
        self.di = di
        self.capacidad = capacidad

    def simulation_params(self):
        return (
            self.Npx,
            self.Npt,
            self.NPOINTS,
            self.Niso,
            self.Xif,
            self.Xi0,
            self.NXi,
            self.Lf,
            self.L0,
            self.NL,
            self.D,
            self.ks,
            self.T,
            self.Mr,
            self.m,
            self.rho,
            self.Rohm,
            self.Eoff,
            self.Qmax,
        )

    def galva_calc(self):

        path = self.so_path

        lib_galva = ct.CDLL(path)

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
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]

        N = int(self.NL * self.NXi)

        res1 = (ct.c_double * N)()
        res2 = (ct.c_double * N)()
        res3 = (ct.c_double * N)()

        lib_galva.galva(
            self.Npx,
            self.Npt,
            self.NPOINTS,
            self.Niso,
            self.Xif,
            self.Xi0,
            self.NXi,
            self.Lf,
            self.L0,
            self.NL,
            self.D,
            self.ks,
            self.T,
            self.Mr,
            self.m,
            self.rho,
            self.Rohm,
            self.Eoff,
            self.Qmax,
            self.ai.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.bi.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.ci.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.di.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.capacidad.ctypes.data_as(ct.POINTER(ct.c_double)),
            res1,
            res2,
            res3,
        )

        logL = np.asarray(np.frombuffer(res1, dtype=np.double, count=N))

        logXi = np.asarray(np.frombuffer(res2, dtype=np.double, count=N))

        SOC = np.asarray(np.frombuffer(res3, dtype=np.double, count=N))

        return logL, logXi, SOC
