#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galva_simul (https://github.com/aruderman/galva_simul/).
# Copyright (c) 2021-2023, Maximiliano Gavilán y Andrés Ruderman
# License: MIT
#   Full Text: https://github.com/aruderman/galva_simul/blob/master/LICENSE

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DOCS
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""simulation_setup class of galva_simul."""

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# IMPORTS
import numpy as np
import pandas as pd
from spline import Spline_params
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# CLASSES
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


class SimulationSetup:
    def __init__(
        self,
        D: float,
        ks: float,
        Mr: float,
        m: float,
        d: float,
        rho: float,
        isotherm=False,
        g=0.5,
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
        self.d = d
        self.rho = rho
        self.Rohm = Rohm
        self.Xi0 = Xi0
        self.Xif = Xif
        self.NXi = NXi
        self.L0 = L0
        self.Lf = Lf
        self.NL = NL
        self.geo = geo

        if isotherm:
            self.frumkin = False
        else:
            self.isotherm = Spline_params(pd.DataFrame({"capacity":[134], "potential":[-0.15]}))
            self.isotherm.ai = np.array(0)
            self.isotherm.bi = np.array(0)
            self.isotherm.ci = np.array(0)
            self.isotherm.di = np.array(0)
            self.isotherm.capacity = np.array(0)
            self.frumkin = True