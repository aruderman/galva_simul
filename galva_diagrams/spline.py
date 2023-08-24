#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galva_simul (https://github.com/aruderman/galva_simul/).
# Copyright (c) 2021-2023, Maximiliano Gavilán y Andrés Ruderman
# License: MIT
#   Full Text: https://github.com/aruderman/galva_simul/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Spline class of galva_simul."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
from csaps import csaps

# ============================================================================
# CLASSES
# ============================================================================


class Spline_params:
    def __init__(self, iso_path=None):
        self.iso_path = iso_path

    def read_iso(self):
        '''
        Takes the path iso and reads the experimental isotherm. The format
        of the input file must be theta vs. potential, separated by a
        space. Returns two arrays with numerical values of theta and potential.
        '''

        try:
            with open(self.iso_path, 'r') as iso_file:
                lines = iso_file.readlines()

            proc_iso = [ff.strip().split('\t') for ff in lines]

            theta = np.array([float(p[1]) for p in proc_iso])

            potential = np.array([float(p[0]) for p in proc_iso])
            return theta, potential

        except FileNotFoundError:
            print(f"File '{self.iso_path}' not found.")
            return []

    def iso_csaps(self, theta, potential, smf=0.9999, Niso=499):
        '''
        Sorts theta and the potential in increasing  order of theta.
        Then builds the interpolation with the Cubic
        Spline Approximation (Smoothing). Returns the interpolated
        potential and the np_point grid (theta). The smf parameter
        must have a value between 0.9999 and 0.999999.
        '''

        # Get the indices that would sort theta
        idx = np.argsort(theta)
        theta_sort = theta[idx]
        # Use the indices to sort the potential
        pot_sort = potential[idx]

        begin = theta_sort[0]
        end = theta_sort[-1]

        theta_inter = np.linspace(begin, end, Niso)
        pot_csaps = csaps(theta_sort, pot_sort, theta_inter, smooth=smf)

        return theta_inter, pot_csaps, Niso

    def iso_spline(self, theta_inter, pot_csaps, Niso):
        '''
        The function iso_spline takes the theta_inter of the smooth isotherm.
        It returns the parameters ai, bi, ci, and di of the cubic spline of
        the isotherm. These parameters can be used to calculate the
        equilibrium potential.
        '''
        ai = pot_csaps.copy()
        hi = theta_inter[1:] - theta_inter[:-1]
        alfai = np.zeros(Niso - 1)

        li = np.zeros(Niso)
        ui = np.zeros(Niso)
        zi = np.zeros(Niso)

        ci = np.zeros(Niso)
        bi = np.zeros(Niso - 1)
        di = np.zeros(Niso - 1)

        for i in range(1, Niso - 1):
            alfai[i] = 3 * (pot_csaps[i + 1] - pot_csaps[i]) / hi[i] -\
                    3 * (pot_csaps[i] - pot_csaps[i - 1]) / hi[i - 1]

        li[0] = 1.0
        ui[0] = zi[0] = 0.0

        for i in range(1, Niso-1):
            li[i] = 2 * (theta_inter[i + 1] - theta_inter[i - 1]) -\
                (hi[i - 1] * ui[i - 1])
            ui[i] = hi[i]/li[i]
            zi[i] = (alfai[i] - hi[i - 1] * zi[i - 1])/li[i]

        li[Niso - 1] = 1.0
        zi[Niso - 1] = ci[Niso - 1] = 0.0

        for j in range(Niso - 2, -1, -1):
            ci[j] = zi[j] - ui[j] * ci[j + 1]
            bi[j] = (pot_csaps[j + 1]-pot_csaps[j]) / hi[j] -\
                hi[j] / 3 * (ci[j + 1] + 2 * ci[j])
            di[j] = (ci[j+1] - ci[j])/(3*hi[j])

        return ai, bi, ci, di
