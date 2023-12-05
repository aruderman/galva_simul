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

from .spline import SplineParams

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


    The present software runs a set ofgalvanostatic simulations [1]_ to
    systematically examine the maximum capacity that a material is capable to
    accommodate at the single-particle level under different experimental
    conditions. These simulations are used to build a diagnostic maps or zone
    diagram [2, 3]_. A similar concept is used here to construct level diagrams
    for galvanostatic simulations.



    This physics-based heuristic model [1]_ uses the maps in
    :ref:`galpynostatic.datasets` to perform a grid search by taking different
    combinations of the diffusion coefficient, :math:`D`, and the
    kinetic-rate constant, :math:`k^0`, to fit experimental data of the
    State-of-Charge (SOC) of the electrode material as a function of the
    C-rates. This is done considering invariant all the other experimental
    values involved in the parameters :math:`\xi` and :math:`\ell` of the
    maps of the continuous galvanostatic model [1]_, such as the
    characteristic diffusion length, :math:`d`, and the geometrical factor,
    :math:`z` (see :ref:`galpynostatic.utils`).

    Each time a set of parameters :math:`D` and :math:`k^0` is taken, the
    SOC values are predicted and the mean square error (MSE) is calculated.
    Then, the set of parameters that minimizes the MSE is obtained, thus
    providing fundamental parameters of the system.

    Parameters
    ----------
    dataset : str or pandas.DataFrame, default="spherical"
        A str indicating the particle geometry (planar, cylindrical or
        spherical) to use the datasets distributed in this package which can
        also be loaded using the functions of the
        :ref:`galpynostatic.datasets` to give it as a ``pandas.DataFrame`` with
        the map of the maximum SOC values as function of the internal
        parameters :math:`\log(\ell)` and :math:`\log(\xi)`.

    d : float, default=1e-4
        Characteristic diffusion length (particle size) in cm.

    z : integer, default=3
        Geometric factor (1 for planar, 2 for cylinder and 3 for sphere).

    dcoeff_lle : integer, default=-15
        The lower limit exponent of the diffusion coefficient line to generate
        the grid.

    dcoeff_ule : integer, default=-6
        The upper limit exponent of the diffusion coefficient line to generate
        the grid.

    dcoeff_num : integer, default=100
        Number of samples of diffusion coefficients to generate between the
        lower and the upper limit exponent.

    k0_lle : integer, default=-14
        The lower limit exponent of the kinetic rate constant line to generate
        the grid.

    k0_ule : integer, default=-5
        The upper limit exponent of the kinetic rate constant line to generate
        the grid.

    k0_num : integer
        Number of samples of kinetic rate constants to generate between the
        lower and the upper limit exponent.

    Notes
    -----
    You can also give your own dataset to another potential cut-off in the
    same format as the distributed ones and as ``pandas.DataFrame``, i.e. in
    the column of :math:`\ell` the different values have to be grouped in
    ascending order and for each of these groups the :math:`\xi` have to be in
    decreasing order and respecting that for each group of :math:`\ell` the
    same values are simulated (this is a restriction to perform the
    ``scipy.interpolate.RectBivariateSpline``, since `x` and `y` have to be
    strictly in a special order, which is handled internally by the
    :ref:`galpynostatic.map`).

    References
    ----------
    .. [1] F. Fernandez, E. M. Gavilán-Arriazu, D. E. Barraco, A. Visintin,
       Y. Ein-Eli and E. P. M. Leiva. "Towards a fast-charging of LIBs
       electrode materials: a heuristic model based on galvanostatic
       simulations." `Electrochimica Acta 464` (2023): 142951.

    Attributes
    ----------
    dcoeff_ : float
        Predicted diffusion coefficient in :math:`cm^2/s`.

    dcoeff_err_ : float
        Uncertainty in the predicted diffusion coefficient.

    k0_ : float
        Predicted kinetic rate constant in :math:`cm/s`.

    k0_err_ : float
        Uncertainty in the predicted kinetic rate constant.

    mse_ : float
        Mean squared error of the best fitted model.
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
        Npt=int(3e6),
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

        self._df = pd.DataFrame(
            {
                "L": self.logL,
                "xi": self.logxi,
                "SOC": SOCC,
            }
        ).sort_values(by=["L", "xi"], ascending=[True, True])

    def to_dataframe(self):
        return self._df

    """
    def plot_old(self, ax=None, plt_kws=None):
        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        x = self.df["L"]
        y = self.df["xi"]
        z = self.df["SOC"]

        contour_plot = ax.tricontourf(
            x, y, z, cmap="viridis", levels=20
        )  # Guardar el resultado de tricontourf

        # Etiquetas de ejes y título
        ax.set_xlabel("log(L)")
        ax.set_ylabel("log(xi)")

        if self.geo == 0:
            ax.set_title("DIAGRAMA PLANO")
        elif self.geo == 1:
            ax.set_title("DIAGRAMA CILINDRICO")
        elif self.geo == 2:
            ax.set_title("DIAGRAMA ESFÉRICO")

        # Barra de colores
        plt.colorbar(contour_plot, label="SOC")

        return ax
    """

    def plot(self, ax=None, plt_kws=None, clb=True, clb_label="SOC"):
        ax = plt.gca() if ax is None else ax

        x = self._df.L
        y = self._df.xi

        logells_ = np.unique(x)
        logxis_ = np.unique(y)
        socs = self._df.SOC.to_numpy().reshape(logells_.size, logxis_.size)

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


class GalvanostaticProfile:
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
        Npt=int(3e6),
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

        N = int(self.Npt / (2 * self.NPOINTS))

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

        self.E = np.asarray(np.frombuffer(res2, dtype=np.double, count=N))

        self.r_norm = np.asarray(
            np.frombuffer(res3, dtype=np.double, count=self.Npx)
        )

        self.tita1 = np.asarray(
            np.frombuffer(res4, dtype=np.double, count=self.Npx)
        )

        self._df = pd.DataFrame(
            {
                "SOC": [ss for ss in self.SOC if ss != 0.0],
                "Potential": [ss for ss in self.E if ss != 0.0],
            }
        )

        self.condf = pd.DataFrame({"r_norm": self.r_norm, "tita": self.tita1})

    def plot(self, ax=None, plt_kws=None):
        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        x = self._df["SOC"]
        y = self._df["Potential"]

        ax.plot(x, y, **plt_kws)

        # Etiquetas de ejes y título
        ax.set_xlabel("SOC")
        ax.set_ylabel("Potential")
        ax.set_title("Isotherm")
        # ax.legend()

        return ax
