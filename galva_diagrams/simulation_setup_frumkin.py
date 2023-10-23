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
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# CLASSES
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


class SimulationSetup:
    def __init__(
        self,
        model: bool,
        g: float,
        N_THREADS: int,
        Npx: int,
        Npt: int,
        NPOINTS: int,
        Niso: int,
        D: float,
        ks: float,
        T: float,
        Mr: float,
        m: float,
        d: float,
        rho: float,
        Rohm: float,
        Xi0: float,
        Xif: float,
        NXi: int,
        L0: float,
        Lf: float,
        NL: int,
        Eoff: float,
        geo: int,
    ):
        self.model = model
        self.g = g
        self.N_THREADS = N_THREADS
        self.Npx = Npx
        self.Npt = Npt
        self.NPOINTS = NPOINTS
        self.Niso = Niso
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
        self.Eoff = Eoff
        self.geo = geo