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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

        lib_galva = ct.CDLL('./lib/galva_PCN_frumkin.so')

        lib_galva.galva.argtypes = [
            ct.c_bool,
            ct.c_double,
            ct.c_int,
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
            self.params.model,
            self.params.g,
            self.params.N_THREADS,
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

        self.logL = np.asarray(np.frombuffer(res1, dtype=np.double, count=N))

        self.logXi = np.asarray(np.frombuffer(res2, dtype=np.double, count=N))

        self.SOC = np.asarray(np.frombuffer(res3, dtype=np.double, count=N))


    def to_DataFrame(self):

        SOCC = []

        for sss in self.SOC:
            if sss < 1.0:
                SOCC.append(sss)
            else:
                SOCC.append(.99999)

        Cr = [np.log(3600 / self.params.D *(self.params.ks / np.exp(xx))**2) for xx in self.logXi]


        d = [np.log(self.params.D * np.exp(xx) / self.params.ks * np.sqrt(np.exp(l) * self.params.geo)) for xx, l in zip(self.logXi, self.logL)]


        self.df = pd.DataFrame({'L': self.logL, 'Xi': self.logXi, 'Cr': Cr, 'd': d, 'SOC': SOCC})

        return self.df

    def plot(self, ax=None, plt_kws=None):

        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        x = self.df['L']
        y = self.df['Xi']
        z = self.df['SOC']

        contour_plot = ax.tricontourf(x, y, z, cmap="viridis", levels=20)  # Guardar el resultado de tricontourf

        # Etiquetas de ejes y título
        ax.set_xlabel('log(L)')
        ax.set_ylabel('log(Xi)')
        ax.set_title('DIAGRAMA ESFÉRICO')

        # Barra de colores
        colorbar = plt.colorbar(contour_plot, label='SOC')  # Usar contour_plot en plt.colorbar

        return ax
