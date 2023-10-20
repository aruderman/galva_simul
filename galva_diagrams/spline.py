#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galva_simul (https://github.com/aruderman/galva_simul/).
# Copyright (c) 2021-2023, Maximiliano Gavilán y Andrés Ruderman
# License: MIT
# Full Text: https://github.com/aruderman/galva_simul/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Spline class of galva_simul."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
from csaps import csaps
import matplotlib.pyplot as plt

# ============================================================================
# CLASSES
# ============================================================================


class Spline_params:
    def __init__(self, iso_path=None, Niso=49):
        self.iso_path = iso_path
        self.Niso = Niso
        self.capacity = None
        self.potential = None

        # self.capacity = df.iloc[0].to_numpy()
        # self.potential = df.iloc[1].to_numpy()
        # self.Qmax = np.max(self.capacity, dtype=float)

    def read_iso(self):
        '''
        Takes the path iso and reads the experimental isotherm. 
        The format of the input file must be capacity vs. potential, 
        separated by a space. Returns two arrays with numerical 
        values of capacity and potential.
        '''

        try:
            with open(self.iso_path, 'r') as iso_file:
                lines = iso_file.readlines()

            proc_iso = [line.strip().split() for line in lines]

            capacity = np.array([float(line[0]) for line in proc_iso])

            max_c = capacity.max()

            self.capacity = np.array([cap / max_c for cap in capacity])

            self.potential = np.array([float(line[1]) for line in proc_iso])

            self.Qmax = float(max_c)
            
            return self.capacity, self.potential, float(max_c)

        except FileNotFoundError:
            print(f"File '{self.iso_path}' not found.")
            return []

    def iso_csaps(self, smf=0.9999):
        '''
        Sorts capacity and the potential in increasing  order of capacity.
        Then builds the interpolation with the Cubic
        Spline Approximation (Smoothing). Returns the interpolated
        potential and the np_point grid (capacity). The smf parameter
        must have a value between 0.9999 and 0.999999.
        '''

        # Get the indices that would sort capacity


        if self.capacity is not None and self.potential is not None:
            idx = np.argsort(self.capacity)
            capacity_sort = self.capacity[idx]
            # Use the indices to sort the potential
            pot_sort = self.potential[idx]

            begin = capacity_sort[0]
            end = capacity_sort[-1]

            capacity_inter = np.linspace(begin, end, self.Niso)
            pot_csaps = csaps(capacity_sort, pot_sort, capacity_inter, smooth=smf)

            return capacity_inter, pot_csaps

        else:
            return "The SOC and potential has not been defined."

    def iso_spline(self):
        '''
        The function iso_spline takes the  normalized experimental
        capacity or the smooth isotherm.
        It returns the parameters ai, bi, ci, and di of the cubic 
        spline of the isotherm. These parameters can be used to 
        calculate the equilibrium potential.
        '''

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
            alfai[i] = 3 * (pot[i + 1] - pot[i]) / hi[i] -\
                    3 * (pot[i] - pot[i - 1]) / hi[i - 1]

        li[0] = 1.0
        ui[0] = zi[0] = 0.0

        for i in range(1, self.Niso-1):
            li[i] = 2 * (cap[i + 1] - cap[i - 1]) -\
                (hi[i - 1] * ui[i - 1])
            ui[i] = hi[i]/li[i]
            zi[i] = (alfai[i] - hi[i - 1] * zi[i - 1])/li[i]

        li[self.Niso - 1] = 1.0
        zi[self.Niso - 1] = ci[self.Niso - 1] = 0.0

        for j in range(self.Niso - 2, -1, -1):
            ci[j] = zi[j] - ui[j] * ci[j + 1]
            bi[j] = (pot[j + 1]-pot[j]) / hi[j] -\
                hi[j] / 3 * (ci[j + 1] + 2 * ci[j])
            di[j] = (ci[j+1] - ci[j])/(3*hi[j])

        self.ai = ai
        self.bi = bi
        self.ci = ci
        self.di = di


    def plot(self, ax=None, plt_kws=None):

        ax = plt.gca() if ax is None else ax

        plt_kws = {} if plt_kws is None else plt_kws

        ax.plot(self.capacity, self.potential)

        return ax
