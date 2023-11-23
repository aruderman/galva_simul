#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galva_simul (https://github.com/aruderman/galva_simul/).
# Copyright (c) 2021-2023, Maximiliano Gavilán y Andrés Ruderman
# License: MIT
#   Full Text: https://github.com/aruderman/galva_simul/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""galva_isotherm class of galva_simul."""

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

from spline import SplineParams

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# CLASSES
# ============================================================================


class galva_isotherm:
    def __init__(
        self,
        Mr: float,
        m: float,
        rho: float,
        Xi: float,
        L: float,
        D=3.647e-13,
        ks=2.9502e-9,
        isotherm=False,
        g=0.0,
        Npx=1000,
        Npt=int(3e6),
        NPOINTS=100,
        T=298.0,
        Rohm=0,
        geo=0,
        method="CN",
    ):
        self.isotherm = isotherm
        self.g = g
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
        self.Xi = Xi
        self.L = L
        self.geo = geo
        self.method = method

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

    @property
    def simulation_params(self):
        return self

    def galva_lineal(self):
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

        self.SOC = np.asarray(np.frombuffer(res1, dtype=np.double, count=N))

        self.E = np.asarray(np.frombuffer(res2, dtype=np.double, count=N))

        self.df = pd.DataFrame(
            {
                "SOC": [ss for ss in self.SOC if ss != 0.0],
                "Potential": [ss for ss in self.E if ss != 0.0],
            }
        )

    def plot(self, ax=None, plt_kws=None):
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
