#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022-2023, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

# import galpynostatic.simulation.spline
from .spline import SplineParams

from .Simulation import GalvanostaticProfile, GalvanostaticMap

import numpy as np

import pandas as pd

import pytest

import scipy

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("capacity", "potential", "refs"),
    [
        (
            [0, 1, 2],
            [0, 1, 4],
            [[0, 1, 4], [1.0, 4.0], [0.0, 6.0, 0.0], [4.0, -4.0]],
        ),
        (
            [0, 1, 2],
            [0, 1, 4],
            [[0, 1, 4], [1.0, 4.0], [0.0, 6.0, 0.0], [4.0, -4.0]],
        ),
    ],
)
def test_spline(capacity, potential, refs):
    """Test the spline of simulation module."""

    df = pd.DataFrame({"capacity": capacity, "potential": potential})

    spl = SplineParams(df)

    spl.iso_spline()

    np.testing.assert_array_almost_equal(spl.ai, refs[0], 6)
    np.testing.assert_array_almost_equal(spl.bi, refs[1], 6)
    np.testing.assert_array_almost_equal(spl.ci, refs[2], 6)
    np.testing.assert_array_almost_equal(spl.di, refs[3], 6)


@pytest.mark.parametrize(
    ("method", "isotherm", "refs"),
    [
        (
            "CN",
            False,
            [
                [0.428358, 0.0, 0.921084],
                [-0.006383, -0.120818, 0.157191],
                PATH / "test_data" / "profileCN.csv",
            ],
        ),
        (
            "BI",
            False,
            [
                [0.428358, 0.0, 0.921084],
                [-0.006383, -0.120818, 0.157191],
                PATH / "test_data" / "profileBI.csv",
            ],
        ),
    ],
)
class TestGalvanostaticProfile:
    def test_soc(self, method, isotherm, refs):
        profile = GalvanostaticProfile(
            180.815,
            2.26e-3,
            4.58,
            method=method,
            L=-1,
            xi=1,
            Npt=20000,
            isotherm=isotherm,
        )
        profile.calc()

        np.testing.assert_almost_equal(np.mean(profile.SOC), refs[0][0], 6)
        np.testing.assert_almost_equal(np.min(profile.SOC), refs[0][1], 6)
        np.testing.assert_almost_equal(np.max(profile.SOC), refs[0][2], 6)

    def test_potential(self, method, isotherm, refs):
        profile = GalvanostaticProfile(
            180.815,
            2.26e-3,
            4.58,
            L=-1,
            xi=1,
            Npt=20000,
            method=method,
        )
        profile.calc()

        np.testing.assert_almost_equal(np.mean(profile.E), refs[1][0], 6)
        np.testing.assert_almost_equal(np.min(profile.E), refs[1][1], 6)
        np.testing.assert_almost_equal(np.max(profile.E), refs[1][2], 6)

    def test_dataframe(self, method, isotherm, refs):
        df = pd.read_csv(refs[2])

        profile = GalvanostaticProfile(
            180.815,
            2.26e-3,
            4.58,
            L=-1,
            xi=1,
            Npt=20000,
            method=method,
        )
        profile.calc()

        pd.testing.assert_frame_equal(profile.df, df)


@pytest.mark.parametrize(
    ("method",),
    [
        ("CN",),
        ("BI",),
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_plot_prifle(fig_test, fig_ref, method):
    profile = GalvanostaticProfile(
        180.815,
        2.26e-3,
        4.58,
        L=-1,
        xi=1,
        Npt=20000,
        method=method,
    )
    profile.calc()

    test_ax = fig_test.subplots()
    profile.plot(ax=test_ax)

    ref_ax = fig_ref.subplots()
    ref_ax.plot(profile.df["SOC"], profile.df["Potential"])

    ref_ax.set_xlabel("SOC")
    ref_ax.set_ylabel("Potential")
    ref_ax.set_title("Isotherm")


@pytest.mark.parametrize(
    ("method", "isotherm", "refs"),
    [
        (
            "CN",
            False,
            [0.417645, 0.000100, 0.998098],
        ),
        (
            "CN",
            PATH / "LMO-1C.csv",
            [0.589062, 7.44792e-7, 1.000949],
        ),
        (
            "BI",
            False,
            [0.417645, 0.000100, 0.9980978],
        ),
        (
            "BI",
            PATH / "LMO-1C.csv",
            [0.589062, 7.44792e-7, 1.000949],
        ),
    ],
)
class TestGalvanostaticMap:
    def test_soc(self, method, isotherm, refs):
        galvamap = GalvanostaticMap(
            180.815,
            2.26e-3,
            4.58,
            Npt=20000,
            NL=3,
            Nxi=3,
            method=method,
            isotherm=isotherm,
        )
        galvamap.calc()

        np.testing.assert_almost_equal(np.mean(galvamap.SOC), refs[0], 4)
        np.testing.assert_almost_equal(np.min(galvamap.SOC), refs[1], 4)
        np.testing.assert_almost_equal(np.max(galvamap.SOC), refs[2], 6)


@pytest.mark.parametrize(
    ("method", "isotherm"),
    [
        ("CN", False),
        ("CN", PATH / "LMO-1C.csv"),
        ("BI", False),
        ("BI", PATH / "LMO-1C.csv"),
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_map_plot(fig_test, fig_ref, method, isotherm):
    galvamap = GalvanostaticMap(
        180.815,
        2.26e-3,
        4.58,
        NL=4,
        Nxi=4,
        Npt=20000,
        method=method,
        isotherm=isotherm,
    )
    galvamap.calc()

    test_ax = fig_test.subplots()
    galvamap.plot(ax=test_ax)

    plt.clf()
    ref_ax = fig_ref.subplots()

    x = galvamap.df_.L
    y = galvamap.df_.xi

    logells_ = np.unique(x)
    logxis_ = np.unique(y)
    socs = galvamap.df_.SOC.to_numpy().reshape(logells_.size, logxis_.size)

    spline_ = scipy.interpolate.RectBivariateSpline(logells_, logxis_, socs)

    xeval = np.linspace(x.min(), x.max(), 1000)
    yeval = np.linspace(y.min(), y.max(), 1000)

    z = spline_(xeval, yeval, grid=True)

    im = ref_ax.imshow(
        z.T,
        extent=[
            xeval.min(),
            xeval.max(),
            yeval.min(),
            yeval.max(),
        ],
        origin="lower",
    )

    clb = plt.colorbar(im, ax=ref_ax)
    clb.ax.set_ylabel("SOC")
    clb.ax.set_ylim((0, 1))

    ref_ax.set_xlabel(r"log($\ell$)")
    ref_ax.set_ylabel(r"log($\Xi$)")

    plt.close(fig_ref)
