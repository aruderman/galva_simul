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
	def __init__(self, 
		Npx, 
		Npt, 
		NPOINTS, 
		Niso, 
		Xif, 
		Xi0, 
		N
		Xi, 
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
		ai, 
		bi, 
		ci, 
		di, 
		theta_inter):

		self.Npx = NPx
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
		self.ai = ai
		self.bi = bi
		self.ci = ci 
		self.di = di 
		self.theta_inter = theta_inter
		

	def galva_calc(self):

	    lib_galva = ct.CDLL("./galva_parallel.so")
	    
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
	        ct.POINTER(ct.c_double),
	        ct.POINTER(ct.c_double),
	        ct.POINTER(ct.c_double),
	        ct.POINTER(ct.c_double),
	        ct.POINTER(ct.c_double),
	        ct.POINTER(ct.c_double),
	        ct.POINTER(ct.c_double),
	        ct.POINTER(ct.c_double),
	        ]

	    N = self.NL * self.NXi

	    logL = (ct.c_double * N)()
	    logXi = (ct.c_double * N)()
	    SOC = (ct.c_double * N)()

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
	        self.ai.ctypes.data_as(ct.POINTER(ct.c_double)),
	        self.bi.ctypes.data_as(ct.POINTER(ct.c_double)),
	        self.ci.ctypes.data_as(ct.POINTER(ct.c_double)),
	        self.di.ctypes.data_as(ct.POINTER(ct.c_double)),
	        self.theta_inter.ctypes.data_as(ct.POINTER(ct.c_double)),
	        logL,
	        logXi, 
	        SOC,   
	        )
	    
	    logL = np.asarray(
	        np.frombuffer(logL, dtype=np.double, count=N)
	    )
	    logXi = np.asarray(
	        np.frombuffer(logXi, dtype=np.double, count=N)
	    )

	    SOC = np.asarray(
	        np.frombuffer(SOC, dtype=np.double, count=N)
	    )
	    
	    return logL, logXi, SOC