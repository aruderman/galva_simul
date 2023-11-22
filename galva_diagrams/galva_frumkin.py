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
import scipy.interpolate

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# CLASSES
# ============================================================================


class galva_diagram:
    def __init__(self, params):
        self.params = params 
        self.logL = np.linspace(params.L0, params.Lf, params.NL)
        self.logXi = np.linspace(params.Xi0, params.Xif, params.NXi)

    @property
    def simulation_params(self):
        return self.params

    def galva_calc(self):

        if self.params.method == 'CN':
            lib_galva = ct.CDLL('./lib/galva_PCN.so')
        elif self.params.method == "BI":
            lib_galva = ct.CDLL('./lib/galva_PBI.so')

        lib_galva.galva.argtypes = [
            ct.c_bool,
            ct.c_double,
            ct.c_int,
            ct.c_int,
            ct.c_int,
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
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
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
            self.params.frumkin,
            self.params.g,
            self.params.N_THREADS,
            self.params.Npx,
            self.params.Npt,
            self.params.NPOINTS,
            self.params.isotherm.Niso,
            self.params.NL,
            self.params.NXi,
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
            self.logL.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.logXi.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.params.isotherm.ai.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.params.isotherm.bi.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.params.isotherm.ci.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.params.isotherm.di.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.params.isotherm.capacity.ctypes.data_as(ct.POINTER(ct.c_double)),
            res1,
            res2,
            res3,
        )

        self.logL = np.asarray(np.frombuffer(res1, dtype=np.double, count=N))

        self.logXi = np.asarray(np.frombuffer(res2, dtype=np.double, count=N))

        self.SOC = np.asarray(np.frombuffer(res3, dtype=np.double, count=N))


        SOCC = []

        for sss in self.SOC:
            if sss < 1.0:
                SOCC.append(sss)
            else:
                SOCC.append(.99999)

        Cr = np.log10([3600 / self.params.D *(self.params.ks / np.exp(xx))**2 for xx in self.logXi])


        d = np.log10([2 * self.params.D * np.exp(xx) / self.params.ks * np.sqrt(np.exp(l) * (1 + self.params.geo)) for xx, l in zip(self.logXi, self.logL)])


        self.df = pd.DataFrame({'L': self.logL, 'Xi': self.logXi, 'Cr': Cr, 'd': d, 'SOC': SOCC}).sort_values(by=['L', 'Xi'], ascending=[True, True])


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

        if self.params.geo == 0:
            ax.set_title('DIAGRAMA PLANO')
        elif self.params.geo == 1:
            ax.set_title('DIAGRAMA CILINDRICO')
        elif self.params.geo == 2:
            ax.set_title('DIAGRAMA ESFÉRICO')
        

        # Barra de colores
        colorbar = plt.colorbar(contour_plot, label='SOC')  # Usar contour_plot en plt.colorbar

        return ax

    def plot_s(self, ax=None, clb=True, clb_label="SOC"):

        ax = plt.gca() if ax is None else ax

        x = self.df.L
        y = self.df.Xi

        logells_ = np.unique(x)
        logxis_ = np.unique(y)
        socs = self.df.SOC.to_numpy().reshape(logells_.size, logxis_.size)

        spline_ = scipy.interpolate.RectBivariateSpline(logells_, logxis_, socs)

        xeval = np.linspace(x.min(), x.max(), 1000)
        yeval = np.linspace(y.min(), y.max(), 1000)

        z = spline_(xeval, yeval, grid=True)

        im = ax.imshow(z.T, 
            extent=[xeval.min(), 
            xeval.max(), 
            yeval.min(), 
            yeval.max(),],
            origin="lower",
        )


        if clb:
            clb = plt.colorbar(im)
            clb.ax.set_ylabel(clb_label)
            clb.ax.set_ylim((0, 1))

        ax.set_xlabel(r"log($\ell$)")
        ax.set_ylabel(r"log($\Xi$)")

        return ax


    def plot2(self, ax=None, plt_kws=None):

        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        x = self.df['d']
        y = self.df['Cr']
        z = self.df['SOC']

        contour_plot = ax.tricontourf(x, y, z, cmap="viridis", levels=20)  # Guardar el resultado de tricontourf

        # Etiquetas de ejes y título
        ax.set_xlabel('log(d)')
        ax.set_ylabel('log(Cr)')
        ax.set_title('DIAGRAMA d/Cr')
    

        # Barra de colores
        colorbar = plt.colorbar(contour_plot, label='SOC')  # Usar contour_plot en plt.colorbar

        return ax

    def plot2_s(self, ax=None, clb=True, clb_label="SOC"):

        ax = plt.gca() if ax is None else ax

        x = self.df.d
        y = self.df.Cr

        logds_ = np.unique(x)
        logcrs_ = np.unique(y)
        socs = self.df.SOC.to_numpy().reshape(logds_.size, logcrs_.size)

        spline_ = scipy.interpolate.RectBivariateSpline(logds_, logcrs_, socs)

        xeval = np.linspace(x.min(), x.max(), 1000)
        yeval = np.linspace(y.min(), y.max(), 1000)

        z = spline_(xeval, yeval, grid=True)

        im = ax.imshow(z.T, 
            extent=[xeval.min(), 
            xeval.max(), 
            yeval.min(), 
            yeval.max(),],
            origin="lower",
        )


        if clb:
            clb = plt.colorbar(im)
            clb.ax.set_ylabel(clb_label)
            clb.ax.set_ylim((0, 1))

        ax.set_xlabel(r"log(d)")
        ax.set_ylabel(r"log(Cr)")

        return ax
