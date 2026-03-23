# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103

import numpy as np
import pytest

from scarlet2.bbox import Box
from scarlet2.detect import (
    Footprint,
    Peak,
    HierarchicalFootprint,
    box_intersect,
    hierarchical_footprints,
    footprint_intersect,
    get_connected_pixels,
    footprints,
)
from scarlet2.wavelets import get_multiresolution_support, starlet_transform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blob(shape, center, radius, peak_value, background=0.0):
    """Return an image with a filled square blob."""
    img = np.full(shape, background, dtype=np.float32)
    y0, y1 = max(0, center[0] - radius), min(shape[0], center[0] + radius + 1)
    x0, x1 = max(0, center[1] - radius), min(shape[1], center[1] + radius + 1)
    img[y0:y1, x0:x1] = peak_value / 2
    img[center] = peak_value
    return img


def _detect_coeffs(img, scales=3):
    """Return masked starlet coefficients for a 2-D image."""
    coeffs = np.asarray(starlet_transform(img, scales=scales))
    sigma = float(np.std(img[img > 0])) * 0.1
    M = get_multiresolution_support(img, coeffs, sigma)
    return M * coeffs


# ---------------------------------------------------------------------------
# get_connected_pixels
# ---------------------------------------------------------------------------


def test_get_connected_pixels_basic():
    img = np.zeros((10, 10))
    img[3:6, 3:6] = 1.0
    unchecked = np.ones((10, 10), dtype=bool)
    footprint = np.zeros((10, 10), dtype=bool)
    bounds = [[4, 5], [4, 5]]

    get_connected_pixels(4, 4, img, unchecked, footprint, bounds, thresh=0)

    assert footprint[3:6, 3:6].all()
    assert not footprint[0, 0]
    assert bounds[0] == [3, 6]
    assert bounds[1] == [3, 6]


def test_get_connected_pixels_thresh():
    img = np.zeros((5, 5))
    img[2, 2] = 0.5  # below thresh=1
    unchecked = np.ones((5, 5), dtype=bool)
    footprint = np.zeros((5, 5), dtype=bool)
    bounds = [[2, 3], [2, 3]]

    get_connected_pixels(2, 2, img, unchecked, footprint, bounds, thresh=1.0)

    assert not footprint.any()


# ---------------------------------------------------------------------------
# get_footprints
# ---------------------------------------------------------------------------


def test_get_footprints_two_blobs():
    img = np.zeros((30, 30), dtype=np.float32)
    img[4:7, 4:7] = 5.0
    img[5, 5] = 10.0
    img[20:23, 20:23] = 3.0
    img[21, 21] = 6.0

    fps = footprints(img, min_separation=2, min_area=4, thresh=0)

    assert len(fps) == 2
    centers = {(fp.peaks[0].y, fp.peaks[0].x) for fp in fps}
    assert (5, 5) in centers
    assert (21, 21) in centers


def test_get_footprints_returns_footprint_type():
    img = np.zeros((10, 10), dtype=np.float32)
    img[3:7, 3:7] = 2.0
    img[5, 5] = 5.0

    fps = footprints(img, min_separation=1, min_area=4, thresh=0)

    assert len(fps) == 1
    fp = fps[0]
    assert isinstance(fp, Footprint)
    assert isinstance(fp.peaks[0], Peak)
    assert fp.peaks[0].y == 5 and fp.peaks[0].x == 5


def test_get_footprints_below_min_area():
    img = np.zeros((10, 10), dtype=np.float32)
    img[5, 5] = 1.0  # single pixel — area = 1

    fps = footprints(img, min_separation=1, min_area=4, thresh=0)

    assert len(fps) == 0


# ---------------------------------------------------------------------------
# Footprint bounds / Box
# ---------------------------------------------------------------------------


def test_footprint_bounds():
    """Footprint bounds use exclusive end coordinates, matching Box convention."""
    img = np.zeros((10, 10), dtype=np.float32)
    img[2:8, 3:9] = 1.0
    img[5, 6] = 2.0
    fps = footprints(img, min_separation=1, min_area=4, thresh=0)
    assert len(fps) == 1
    bounds = fps[0].bounds
    bbox = Box.from_bounds(*bounds)
    assert bbox.origin == (bounds[0][0], bounds[1][0])
    assert bbox.shape == (bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0])


# ---------------------------------------------------------------------------
# box_intersect / footprint_intersect
# ---------------------------------------------------------------------------


def test_box_intersect_overlapping():
    b1 = Box((10, 10), origin=(0, 0))
    b2 = Box((10, 10), origin=(5, 5))
    assert box_intersect(b1, b2)


def test_box_intersect_non_overlapping():
    b1 = Box((5, 5), origin=(0, 0))
    b2 = Box((5, 5), origin=(10, 10))
    assert not box_intersect(b1, b2)


def test_footprint_intersect():
    fp1 = np.ones((5, 5), dtype=bool)
    fp2 = np.ones((5, 5), dtype=bool)
    b1 = Box((5, 5), origin=(0, 0))
    b2 = Box((5, 5), origin=(3, 3))
    assert footprint_intersect(fp1, b1, fp2, b2)


def test_footprint_no_intersect():
    fp1 = np.ones((5, 5), dtype=bool)
    fp2 = np.ones((5, 5), dtype=bool)
    b1 = Box((5, 5), origin=(0, 0))
    b2 = Box((5, 5), origin=(10, 10))
    assert not footprint_intersect(fp1, b1, fp2, b2)


# ---------------------------------------------------------------------------
# build_source_list
# ---------------------------------------------------------------------------


def test_build_source_list_single_source():
    # Use a Gaussian-like point source so the wavelet peak is unambiguous
    yy, xx = np.mgrid[:40, :40]
    img = 10.0 * np.exp(-((yy - 20) ** 2 + (xx - 20) ** 2) / (2 * 2.0**2)).astype(np.float32)
    detect = _detect_coeffs(img)
    sources = hierarchical_footprints(detect)

    assert len(sources) >= 1
    centers = [(s.center[0], s.center[1]) for s in sources]
    # the dominant peak should be close to (20, 20)
    assert any(abs(cy - 20) <= 3 and abs(cx - 20) <= 3 for cy, cx in centers)


def test_build_source_list_two_blobs():
    """Two well-separated blobs should produce at least two detected sources."""
    img = np.zeros((50, 50), dtype=np.float32)
    img += _make_blob((50, 50), center=(12, 12), radius=4, peak_value=10.0)
    img += _make_blob((50, 50), center=(37, 37), radius=4, peak_value=6.0)
    detect = _detect_coeffs(img)
    sources = hierarchical_footprints(detect, flatten=True)

    assert len(sources) >= 2


def test_build_source_list_returns_scene_source():
    img = _make_blob((30, 30), center=(15, 15), radius=4, peak_value=8.0)
    detect = _detect_coeffs(img)
    sources = hierarchical_footprints(detect)

    assert all(isinstance(s, HierarchicalFootprint) for s in sources)
    for s in sources:
        assert isinstance(s.center, tuple) and len(s.center) == 2
        assert isinstance(s.scale, int)
        assert isinstance(s.children, list)


def test_build_source_list_orphan_promoted():
    """A compact source should be detected even if it only appears at finer scales."""
    img = np.zeros((50, 50), dtype=np.float32)
    # large blob visible at all scales
    img += _make_blob((50, 50), center=(15, 15), radius=6, peak_value=10.0)
    # compact source: bright single pixel, tiny footprint
    img[38, 38] = 8.0
    img[37:40, 37:40] = 2.0
    detect = _detect_coeffs(img)
    all_sources = hierarchical_footprints(detect, flatten=True)

    all_centers = [(s.center[0], s.center[1]) for s in all_sources]
    assert any(abs(cy - 38) <= 3 and abs(cx - 38) <= 3 for cy, cx in all_centers)


def test_build_source_list_flat_list():
    """flat_list=True returns all nodes with empty children lists."""
    img = np.zeros((50, 50), dtype=np.float32)
    img += _make_blob((50, 50), center=(12, 12), radius=4, peak_value=10.0)
    img += _make_blob((50, 50), center=(37, 37), radius=4, peak_value=6.0)
    detect = _detect_coeffs(img)
    sources = hierarchical_footprints(detect, flatten=True)

    assert all(isinstance(s, HierarchicalFootprint) for s in sources)
    assert all(s.children == [] for s in sources)


def test_build_source_list_scale_coarsening():
    """Passing detect[k:] restricts detection to scales k and above."""
    img = np.zeros((50, 50), dtype=np.float32)
    img += _make_blob((50, 50), center=(25, 25), radius=6, peak_value=10.0)
    detect = _detect_coeffs(img, scales=4)

    sources_all = hierarchical_footprints(detect)
    sources_coarse = hierarchical_footprints(detect[2:])

    assert all(s.scale >= 0 for s in sources_coarse)
    assert len(sources_all) >= 1
    assert len(sources_coarse) >= 1
