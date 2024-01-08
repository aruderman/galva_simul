#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galva_simul (https://github.com/aruderman/galva_simul/).
# Copyright (c) 2021-2023, Maximiliano Gavilán y Andrés Ruderman
# License: MIT
#   Full Text: https://github.com/aruderman/galva_simul/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""GalvanostaticMap and GalvanostaticProfile classes of Simulation."""

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

# from spline import SplineParams

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

_PROFILE_LIBS = {
    "CN": ct.CDLL(PATH / "lib" / "galva_LCN.so"),
    "BI": ct.CDLL(PATH / "lib" / "galva_LBI.so"),
}

_MAPS_LIBS = {
    "CN": ct.CDLL(PATH / "lib" / "galva_PCN.so"),
    "BI": ct.CDLL(PATH / "lib" / "galva_PBI.so"),
}

# ============================================================================
# CLASSES
# ============================================================================


class GalvanostaticMap:
    r"""Diagnostic map tool for galvanostatic intercalation material analysis.

    A tool to build a diagnostic map intercalation materials under galvanostatic conditions.


    The present software performs a series of galvanostatic simulations [1]_
    to systematically investigate the maximum state of charge (SOC) that a
    material is capable of accommodating at the single particle level under
    different experimental conditions. These simulations are used to produce
    a diagnostic map or zone diagram [2, 3]_.
    A similar concept is used here to construct level diagrams for
    galvanostatic simulations using two variables:

    :math:`\Xi = k^{0}\left (\frac{t_{h}}{C_{r}D}  \right )^{1/2}` and
    :math:`\ell=\frac{r^{2}C_{r}}{zt_{h}D}`
    at constant :math:`D` and :math:`k_0`.

    The SOC are calculated by means of the interpolation of the experimental
    or theoretical isotherm, the Fick's diffusion law and the Butler-Volmer
    charge transfer equation with a transfer coefficient of 0.5.



    Parameters
    ----------
    Mr : float
        Molecular mass of the electrode active material in :math:`g/mol`.

    m : float
        Total mass of the electrode active material in :math:`g`.

    rho : float
        Density of the electrode active material in :math:`g/cm^3`.

    D : float, default=3.647e-13 :math:`cm/s^2`
        The diffusion coefficient in :math:`cm/s^2`.

    ks : float, default=2.9502e-9
        Kinetic rate constant in :math:`cm/s`.

    isotherm : bool or pandas.DataFrame, default=False
        A dataset containing the experimental isotherm values in the
        format potential vs capacity. The isotherm is used to calculate
        the equilibrium potential for a given SOC. If False the
        equilibrium potential is calculated using the theroretical model
        ... [].

    g : float, default=0.0
        Interaction parameter of the theroretical model used to obtain
        the equilibrium potential if isotherm=False.

    N_THREADS : int, default=-1
        Number of threads in which the diagram calculation will be performed.
        -1 means use all available threads.

    Npx : int, default=1000
        Size of the spatial grid in wich the Fick's equation will be solved.

    Npt : int, default=3000000
        Size of the time grid in wich the Fick's equation will be solved.

    NPOINTS : int, default=100
        Npt/NPOINTS time points at which SOC and potential are printed.

    T : float, default=298.0
        Working temperature of the cell.

    Rohm : float, default=0.0
        Cell's resistance.

    xi0 : float, default=2.0
        Initial value of the :math:`\log(\Xi)`.

    xif : float, default=-4.0
        Final value of the :math:`\log(\Xi)`.

    Nxi : int, default=5
        Number of :math:`\log(\Xi)` values.

    L0 : float, default=2.0
        Initial value of the :math:`\log(\ell)`.

    Lf : float, default=-4.0
        Final value of the :math:`\log(\ell)`.

    NL : int, default=5
        Number of :math:`\log(\ell)` values.

    geo : int default=2
        Active material particle geometry. 0=planar, 1=cylindrical, 2=spherical.

    method : str, default="CN"
        Method to solve the Fick's equation. "CN"=Crank-Nicholson, "BI"=Backward
        Implicit.

    Efin : float, default=-0.15
        Cut potential if isotherm=False.

    Notes
    -----

    References
    ----------


    Attributes
    ----------

    """

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
        Npt=int(1e5),
        NPOINTS=100,
        T=298.0,
        Rohm=0,
        xi0=2.0,
        xif=-4.0,
        Nxi=5,
        L0=2.0,
        Lf=-4.0,
        NL=5,
        geo=2,
        method="CN",
        Efin=-0.15,
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
        self.xi0 = xi0
        self.xif = xif
        self.Nxi = Nxi
        self.L0 = L0
        self.Lf = Lf
        self.NL = NL
        self.geo = geo
        self.method = method
        self.Efin = Efin

        if isotherm:
            self.frumkin = False
            df = pd.read_csv(self.isotherm, names=["capacity", "potential"])
            self.isotherm = SplineParams(df)
            self.isotherm.iso_spline()
        else:
            Qm = 96484.5561 / (3.6 * self.Mr)
            self.isotherm = SplineParams(
                pd.DataFrame({"capacity": [Qm], "potential": [self.Efin]})
            )
            self.isotherm.ai = np.array(0)
            self.isotherm.bi = np.array(0)
            self.isotherm.ci = np.array(0)
            self.isotherm.di = np.array(0)
            self.isotherm.capacity = np.array(0)
            self.frumkin = True

        self.logL = np.linspace(self.L0, self.Lf, self.NL)
        self.logxi = np.linspace(self.xi0, self.xif, self.Nxi)

    def calc(self):
        lib_galva = _MAPS_LIBS[self.method]

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

        N = int(self.NL * self.Nxi)

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
            self.Nxi,
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
            self.logxi.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.ai.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.bi.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.ci.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.di.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.capacity.ctypes.data_as(ct.POINTER(ct.c_double)),
            res1,
            res2,
            res3,
        )

        self.logL = np.asarray(np.frombuffer(res1, dtype=np.double, count=N))

        self.logxi = np.asarray(np.frombuffer(res2, dtype=np.double, count=N))

        self.SOC = np.asarray(np.frombuffer(res3, dtype=np.double, count=N))

        SOCC = []

        for sss in self.SOC:
            if sss < 1.0:
                SOCC.append(sss)
            else:
                SOCC.append(0.99999)

        self.df = pd.DataFrame(
            {
                "L": self.logL,
                "xi": self.logxi,
                "SOC": SOCC,
            }
        ).sort_values(
            by=["L", "xi"], ascending=[True, True], ignore_index=True
        )

    def to_dataframe(self):
        """
        A function that returns the diagram dataset.
        """
        return self.df

    def plot(self, ax=None, plt_kws=None, clb=True, clb_label="SOC"):
        """
        A function that returns the axis of the two dimensional diagram
        for a given axis.

        Parameters
        -----
        ax : axis, default=None
            Axis of wich the diagram plot.

        plt_kws : dict, default=None
            A dictionary containig the parameters to be passed to the axis.

        clb : bool, default=True
            Parameter that determines if the color bar will be displayed.

        clb_label : str, default="SOC"
            Name of the color bar.
        """
        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        x = self.df.L
        y = self.df.xi

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
            **plt_kws,
        )

        if clb:
            clb = plt.colorbar(im)
            clb.ax.set_ylabel(clb_label)
            clb.ax.set_ylim((0, 1))

        ax.set_xlabel(r"log($\ell$)")
        ax.set_ylabel(r"log($\Xi$)")

        return ax


class GalvanostaticProfile:
    r"""A tool to extrapolate isotherms varing C-rate and particle size.


    This software simulates new isotherms individually, from an
    experimental or theoretical one, dependding on the value of
    :math:`\log(\ell)` and :math:`\log(\Xi)`. Given an isotherm for a
    particular particle size and C-rate this tool will predict the system
    behaviour varing :math:`\log(\ell)` and :math:`\log(\Xi)`. The
    resulting isotherms are calculated by means of the interpolation of
    the experimental or theoretical isotherm, the Fick's diffusion law and
    the Butler-Volmer charge transfer equation with a transfer coefficient
    of 0.5.


    Parameters
    ----------
    Mr : float
        Molecular mass of the electrode active material in :math:`g/mol`.

    m : float
        Total mass of the electrode active material in :math:`g`.

    rho : float
        Density of the electrode active material in :math:`g/cm^3`.

    D : float, default=3.647e-13 :math:`cm/s^2`
        The diffusion coefficient in :math:`cm/s^2`.

    ks : float, default=2.9502e-9
        Kinetic rate constant in :math:`cm/s`.

    isotherm : bool or pandas.DataFrame, default=False
        A dataset containing the experimental isotherm values in the
        format SOC vs potential. The isotherm is used to calculate the
        equilibrium potential for a given SOC. If False the equilibrium
        potential is calculated using the theroretical model ... [].

    g : float, default=0.0
        Interaction parameter of the theroretical model used to obtain
        the equilibrium potential if isotherm=False.

    N_THREADS : int, default=-1
        Number of threads in which the diagram calculation will be
        performed. N_THREADS=-1 means use all available threads.

    Npx : int, default=1000
        Size of the spatial grid in wich the Fick's equation will be
        solved.

    Npt : int, default=3000000
        Size of the time grid in wich the Fick's equation will be solved.

    NPOINTS : int, default=100
        Npt/NPOINTS time points at which SOC and potential are printed.

    T : float, default=298.0
        Working temperature of the cell.

    Rohm : float, default=0.0
        Cell's resistance.

    xi : float, default=2.0
        Value of the :math:`\log(\Xi)`.

    L : float, default=2.0
        Value of the :math:`\log(\ell)`.

    geo : int default=2
        Active material particle geometry. 0=planar, 1=cylindrical,
        2=spherical.

    method : str, default="CN"
        Method to solve the Fick's equation. "CN"=Crank-Nicholson,
        "BI"=Backward Implicit.

    Efin : float, default=-0.15
        Cut potential if isotherm=False.

    SOCperf : float default=0.5
        SOC value at wich the concentration profile will be calculated.

    Notes
    -----

    References
    ----------


    Attributes
    ----------
    isotherm_df : pandas.DataFrame
        A dataset containig the resulting isotherm in the potential vs SOC
        format.

    concentration_df : pandas.DataFrame
        A dataset containing the resulting concentration profile in the
        form :math:`\theta` (concentration) vs :math:`r` (distance to the
        center of the particle).

    """

    def __init__(
        self,
        Mr: float,
        m: float,
        rho: float,
        xi: float,
        L: float,
        D=3.647e-13,
        ks=2.9502e-9,
        isotherm=False,
        g=0.0,
        Npx=1000,
        Npt=int(1e5),
        NPOINTS=100,
        T=298.0,
        Rohm=0,
        geo=2,
        method="CN",
        Efin=-0.15,
        SOCperf=0.5,
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
        self.xi = xi
        self.L = L
        self.geo = geo
        self.method = method
        self.Efin = Efin
        self.SOCperf = SOCperf

        if isotherm:
            self.frumkin = False
            df = pd.read_csv(self.isotherm, names=["capacity", "potential"])
            self.isotherm = SplineParams(df)
            self.isotherm.iso_spline()
        else:
            Qm = 96484.5561 / (3.6 * self.Mr)
            self.isotherm = SplineParams(
                pd.DataFrame({"capacity": [Qm], "potential": [self.Efin]})
            )
            self.isotherm.ai = np.array(0)
            self.isotherm.bi = np.array(0)
            self.isotherm.ci = np.array(0)
            self.isotherm.di = np.array(0)
            self.isotherm.capacity = np.array(0)
            self.frumkin = True

    def calc(self):
        lib_galva = _PROFILE_LIBS[self.method]

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
        ]

        N = int(self.Npt / self.NPOINTS)

        res1 = (ct.c_double * N)()
        res2 = (ct.c_double * N)()
        res3 = (ct.c_double * self.Npx)()
        res4 = (ct.c_double * self.Npx)()

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
            self.xi,
            self.L,
            self.SOCperf,
            self.isotherm.ai.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.bi.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.ci.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.di.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.capacity.ctypes.data_as(ct.POINTER(ct.c_double)),
            res1,
            res2,
            res3,
            res4,
        )

        self.SOC = np.asarray(np.frombuffer(res1, dtype=np.double, count=N))

        self.E = np.asarray(np.frombuffer(res2, dtype=float, count=N))

        self.r_norm = np.asarray(
            np.frombuffer(res3, dtype=np.double, count=self.Npx)
        )

        self.tita1 = np.asarray(
            np.frombuffer(res4, dtype=np.double, count=self.Npx)
        )

        self.isotherm_df = pd.DataFrame(
            {
                "SOC": [ss for ss in self.SOC if ss != 0.0],
                "Potential": [ss for ss in self.E if ss != 0.0],
            }
        )

        self.concentration_df = pd.DataFrame(
            {"r_norm": self.r_norm, "theta": self.tita1}
        )

    def isoplot(self, ax=None, plt_kws=None):
        """
        A function that returns the axis of the simulated isotherm.

        Parameters
        -----
        ax : axis, default=None
            Axis of wich the diagram plot.

        plt_kws : dict, default=None
            A dictionary containig the parameters to be passed to the axis.
        """
        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        x = self.isotherm_df["SOC"]
        y = self.isotherm_df["Potential"]

        ax.plot(x, y, **plt_kws)

        return ax

    def consplot(self, ax=None, plt_kws=None):
        """
        A function that returns the axis of the simulated concentration
        profile for a given SOCperf.

        Parameters
        -----
        ax : axis, default=None
            Axis of wich the diagram plot.

        plt_kws : dict, default=None
            A dictionary containig the parameters to be passed to the axis.
        """
        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        x = self.concentration_df["r_norm"]
        y = self.concentration_df["theta"]

        ax.plot(x, y, **plt_kws)

        ax.set_xlabel("$r_{norm}$")
        ax.set_ylabel("$\theta$")
        ax.set_title(f"Concentration profile, SOC={self.SOCperf}")
        # ax.legend()

        return ax


class SplineParams:
    def __init__(self, df, col_names=["capacity", "potential"]):
        self.df = df
        self.capacity = self.df[col_names[0]].to_numpy()
        self.Qmax = np.max(self.capacity)
        self.capacity = self.df[col_names[0]].to_numpy() / self.Qmax
        self.potential = self.df[col_names[1]].to_numpy()
        self.Eoff = np.min(self.potential)
        self.Niso = self.df.shape[0]

    def iso_spline(self):
        """
        The function iso_spline takes the  normalized experimental
        capacity or the smooth isotherm.
        It returns the parameters ai, bi, ci, and di of the cubic
        spline of the isotherm. These parameters can be used to
        calculate the equilibrium potential.
        """

        pot = self.potential
        cap = self.capacity

        ai = pot.copy()
        hi = cap[1:] - cap[:-1]
        alfai = np.zeros(self.Niso - 1)

        li = np.zeros(self.Niso)
        ui = np.zeros(self.Niso)
        zi = np.zeros(self.Niso)

        ci = np.zeros(self.Niso)
        bi = np.zeros(self.Niso - 1)
        di = np.zeros(self.Niso - 1)

        for i in range(1, self.Niso - 1):
            alfai[i] = (
                3 * (pot[i + 1] - pot[i]) / hi[i]
                - 3 * (pot[i] - pot[i - 1]) / hi[i - 1]
            )

        li[0] = 1.0
        ui[0] = zi[0] = 0.0

        for i in range(1, self.Niso - 1):
            li[i] = 2 * (cap[i + 1] - cap[i - 1]) - (hi[i - 1] * ui[i - 1])
            ui[i] = hi[i] / li[i]
            zi[i] = (alfai[i] - hi[i - 1] * zi[i - 1]) / li[i]

        li[self.Niso - 1] = 1.0
        zi[self.Niso - 1] = ci[self.Niso - 1] = 0.0

        for j in range(self.Niso - 2, -1, -1):
            ci[j] = zi[j] - ui[j] * ci[j + 1]
            bi[j] = (pot[j + 1] - pot[j]) / hi[j] - hi[j] / 3 * (
                ci[j + 1] + 2 * ci[j]
            )
            di[j] = (ci[j + 1] - ci[j]) / (3 * hi[j])

        self.ai = ai
        self.bi = bi
        self.ci = ci
        self.di = di
