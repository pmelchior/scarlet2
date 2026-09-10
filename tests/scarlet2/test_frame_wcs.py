# ruff: noqa: D103

import astropy.units as u
import astropy.wcs
import numpy as np
from numpy.testing import assert_allclose

from scarlet2.frame import _wcs_default, get_affine, get_pixel_size


def _tan_wcs():
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.crpix = [50, 50]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs


SCALE = 0.000278  # deg / pixel


def test_get_affine_cd_matrix():
    wcs = _tan_wcs()
    wcs.wcs.cd = [[-SCALE, 0.0], [0.0, SCALE]]
    assert_allclose(get_affine(wcs), [[-SCALE, 0.0], [0.0, SCALE]])


def test_get_affine_pc_plus_cdelt():
    # PC identity + CDELT: this is what _wcs_default and many real pipelines produce
    wcs = _tan_wcs()
    wcs.wcs.pc = [[1.0, 0.0], [0.0, 1.0]]
    wcs.wcs.cdelt = [-SCALE, SCALE]
    assert_allclose(get_affine(wcs), [[-SCALE, 0.0], [0.0, SCALE]])


def test_get_affine_pc_only():
    wcs = _tan_wcs()
    wcs.wcs.pc = [[-SCALE, 0.0], [0.0, SCALE]]
    assert_allclose(get_affine(wcs), [[-SCALE, 0.0], [0.0, SCALE]])


def test_get_pixel_size_matches_astropy():
    # the CDELT-based default WCS used to report 1 deg/pixel
    wcs = _wcs_default((100, 100))
    pixel_size = get_pixel_size(wcs)
    assert isinstance(pixel_size, u.Quantity)
    assert_allclose(pixel_size.to_value(u.deg), SCALE, rtol=1e-6)
    scales = [s.to_value(u.deg) for s in wcs.proj_plane_pixel_scales()]
    assert_allclose(pixel_size.to_value(u.deg), np.sqrt(np.prod(scales)), rtol=1e-6)
