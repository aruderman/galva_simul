#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galva_simul (https://github.com/aruderman/galva_simul/).
# Copyright (c) 2021-2023, Maximiliano Gavilán y Andrés Ruderman
# License: MIT
#   Full Text: https://github.com/aruderman/galva_simul/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Diagnosis class of Simulation."""

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

from spline import SplineParams

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# CLASSES
# ============================================================================


class Diagnosis:
    def __init__(
        self,
        Mr: float,
        m: float,
        rho: float,
        D=3.647e-13,
        ks=2.9502e-9,
        isotherm=False,
        g=0.0,
        N_THREADS=-1,
        Npx=1000,
        Npt=int(3e6),
        NPOINTS=100,
        T=298.0,
        Rohm=0,
        Xi0=2.0,
        Xif=-4.0,
        NXi=5,
        L0=2.0,
        Lf=-4.0,
        NL=5,
        geo=0,
        method="CN",
        parallel=True,
        Xi=0,
        L=0,
    ):
        self.isotherm = isotherm
        self.g = g
        self.N_THREADS = N_THREADS
        self.Npx = Npx
        self.Npt = Npt
        self.NPOINTS = NPOINTS
        self.D = D
        self.ks = ks
        self.T = T
        self.Mr = Mr
        self.m = m
        self.rho = rho
        self.Rohm = Rohm
        self.Xi0 = Xi0
        self.Xif = Xif
        self.NXi = NXi
        self.L0 = L0
        self.Lf = Lf
        self.NL = NL
        self.geo = geo
        self.method = method
        self.parallel = parallel
        self.Xi = Xi
        self.L = L

        if isotherm:
            self.frumkin = False
        else:
            Qm = 96484.5561 / (3.6 * self.Mr)
            self.isotherm = SplineParams(
                pd.DataFrame({"capacity": [Qm], "potential": [-0.15]})
            )
            self.isotherm.ai = np.array(0)
            self.isotherm.bi = np.array(0)
            self.isotherm.ci = np.array(0)
            self.isotherm.di = np.array(0)
            self.isotherm.capacity = np.array(0)
            self.frumkin = True

        self.logL = np.linspace(self.L0, self.Lf, self.NL)
        self.logXi = np.linspace(self.Xi0, self.Xif, self.NXi)

    @property
    def simulation_params(self):
        return self

    def calc(self):
        if self.parallel:
            if self.method == "CN":
                lib_galva = ct.CDLL("./lib/galva_PCN.so")
            elif self.method == "BI":
                lib_galva = ct.CDLL("./lib/galva_PBI.so")

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

            N = int(self.NL * self.NXi)

            res1 = (ct.c_double * N)()
            res2 = (ct.c_double * N)()
            res3 = (ct.c_double * N)()

            lib_galva.galva(
                self.frumkin,
                self.g,
                self.N_THREADS,
                self.Npx,
                self.Npt,
                self.NPOINTS,
                self.isotherm.Niso,
                self.NL,
                self.NXi,
                self.D,
                self.ks,
                self.T,
                self.Mr,
                self.m,
                self.rho,
                self.Rohm,
                self.isotherm.Eoff,
                self.isotherm.Qmax,
                self.geo,
                self.logL.ctypes.data_as(ct.POINTER(ct.c_double)),
                self.logXi.ctypes.data_as(ct.POINTER(ct.c_double)),
                self.isotherm.ai.ctypes.data_as(ct.POINTER(ct.c_double)),
                self.isotherm.bi.ctypes.data_as(ct.POINTER(ct.c_double)),
                self.isotherm.ci.ctypes.data_as(ct.POINTER(ct.c_double)),
                self.isotherm.di.ctypes.data_as(ct.POINTER(ct.c_double)),
                self.isotherm.capacity.ctypes.data_as(ct.POINTER(ct.c_double)),
                res1,
                res2,
                res3,
            )

            self.logL = np.asarray(
                np.frombuffer(res1, dtype=np.double, count=N)
            )

            self.logXi = np.asarray(
                np.frombuffer(res2, dtype=np.double, count=N)
            )

            self.SOC = np.asarray(
                np.frombuffer(res3, dtype=np.double, count=N)
            )

            SOCC = []

            for sss in self.SOC:
                if sss < 1.0:
                    SOCC.append(sss)
                else:
                    SOCC.append(0.99999)

            Cr = np.log10(
                [
                    3600 / self.D * (self.ks / np.exp(xx)) ** 2
                    for xx in self.logXi
                ]
            )

            d = np.log10(
                [
                    2
                    * self.D
                    * np.exp(xx)
                    / self.ks
                    * np.sqrt(np.exp(l) * (1 + self.geo))
                    for xx, l in zip(self.logXi, self.logL)
                ]
            )

            self.df = pd.DataFrame(
                {
                    "L": self.logL,
                    "Xi": self.logXi,
                    "Cr": Cr,
                    "d": d,
                    "SOC": SOCC,
                }
            ).sort_values(by=["L", "Xi"], ascending=[True, True])
        else:
            if self.method == "CN":
                lib_galva = ct.CDLL("./lib/galva_LCN.so")
            elif self.method == "BI":
                lib_galva = ct.CDLL("./lib/galva_LBI.so")

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

            N = int(self.Npt / (2 * self.NPOINTS))

            res1 = (ct.c_double * N)()
            res2 = (ct.c_double * N)()

            lib_galva.galva(
                self.frumkin,
                self.g,
                self.Npx,
                self.Npt,
                self.NPOINTS,
                self.isotherm.Niso,
                self.D,
                self.ks,
                self.T,
                self.Mr,
                self.m,
                self.rho,
                self.Rohm,
                self.isotherm.Eoff,
                self.isotherm.Qmax,
                self.geo,
                self.Xi,
                self.L,
                self.isotherm.ai.ctypes.data_as(ct.POINTER(ct.c_double)),
                self.isotherm.bi.ctypes.data_as(ct.POINTER(ct.c_double)),
                self.isotherm.ci.ctypes.data_as(ct.POINTER(ct.c_double)),
                self.isotherm.di.ctypes.data_as(ct.POINTER(ct.c_double)),
                self.isotherm.capacity.ctypes.data_as(ct.POINTER(ct.c_double)),
                res1,
                res2,
            )

            self.SOC = np.asarray(
                np.frombuffer(res1, dtype=np.double, count=N)
            )

            self.E = np.asarray(np.frombuffer(res2, dtype=np.double, count=N))

            self.df = pd.DataFrame(
                {
                    "SOC": [ss for ss in self.SOC if ss != 0.0],
                    "Potential": [ss for ss in self.E if ss != 0.0],
                }
            )

    def plot_old(self, ax=None, plt_kws=None):
        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        x = self.df["L"]
        y = self.df["Xi"]
        z = self.df["SOC"]

        contour_plot = ax.tricontourf(
            x, y, z, cmap="viridis", levels=20
        )  # Guardar el resultado de tricontourf

        # Etiquetas de ejes y título
        ax.set_xlabel("log(L)")
        ax.set_ylabel("log(Xi)")

        if self.geo == 0:
            ax.set_title("DIAGRAMA PLANO")
        elif self.geo == 1:
            ax.set_title("DIAGRAMA CILINDRICO")
        elif self.geo == 2:
            ax.set_title("DIAGRAMA ESFÉRICO")

        # Barra de colores
        plt.colorbar(contour_plot, label="SOC")

        return ax

    def plot(self, ax=None, plt_kws=None, clb=True, clb_label="SOC"):
        if self.parallel:
            ax = plt.gca() if ax is None else ax

            x = self.df.L
            y = self.df.Xi

            logells_ = np.unique(x)
            logxis_ = np.unique(y)
            socs = self.df.SOC.to_numpy().reshape(logells_.size, logxis_.size)

            spline_ = scipy.interpolate.RectBivariateSpline(
                logells_, logxis_, socs
            )

            xeval = np.linspace(x.min(), x.max(), 1000)
            yeval = np.linspace(y.min(), y.max(), 1000)

            z = spline_(xeval, yeval, grid=True)

            im = ax.imshow(
                z.T,
                extent=[
                    xeval.min(),
                    xeval.max(),
                    yeval.min(),
                    yeval.max(),
                ],
                origin="lower",
            )

            if clb:
                clb = plt.colorbar(im)
                clb.ax.set_ylabel(clb_label)
                clb.ax.set_ylim((0, 1))

            ax.set_xlabel(r"log($\ell$)")
            ax.set_ylabel(r"log($\Xi$)")

            return ax
        else:
            ax = plt.gca() if ax is None else ax
            plt_kws = {} if plt_kws is None else plt_kws

            x = self.df["SOC"]
            y = self.df["Potential"]

            ax.plot(x, y, **plt_kws)

            # Etiquetas de ejes y título
            ax.set_xlabel("SOC")
            ax.set_ylabel("Potential")
            ax.set_title("Isotherm")

            return ax

    """
    def plot2(self, ax=None, plt_kws=None):
        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        x = self.df["d"]
        y = self.df["Cr"]
        z = self.df["SOC"]

        contour_plot = ax.tricontourf(
            x, y, z, cmap="viridis", levels=20
        )  # Guardar el resultado de tricontourf

        # Etiquetas de ejes y título
        ax.set_xlabel("log(d)")
        ax.set_ylabel("log(Cr)")
        ax.set_title("DIAGRAMA d/Cr")

        # Barra de colores
        plt.colorbar(contour_plot, label="SOC")

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

        im = ax.imshow(
            z.T,
            extent=[
                xeval.min(),
                xeval.max(),
                yeval.min(),
                yeval.max(),
            ],
            origin="lower",
        )

        if clb:
            clb = plt.colorbar(im)
            clb.ax.set_ylabel(clb_label)
            clb.ax.set_ylim((0, 1))

        ax.set_xlabel(r"log(d)")
        ax.set_ylabel(r"log(Cr)")

        return ax
    """
