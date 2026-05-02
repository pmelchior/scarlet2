# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103

import types

import numpy as np

from scarlet2.bbox import Box
from scarlet2.detect import (
    Footprint,
    HierarchicalFootprint,
    Peak,
    box_intersect,
    footprint_intersect,
    footprints,
    hierarchical_footprints,
)

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


def _make_obs(img, noise=0.1):
    """Wrap a 2-D image in a minimal observation-like object."""
    data = img[np.newaxis].astype(np.float32)  # (1, H, W)
    weights = np.full_like(data, 1.0 / noise**2)
    return types.SimpleNamespace(data=data, weights=weights)


# ---------------------------------------------------------------------------
# get_connected_pixels
# ---------------------------------------------------------------------------


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
    obs = _make_obs(img)
    sources = hierarchical_footprints(obs)

    assert len(sources) >= 1
    peaks = [(s.peak.y, s.peak.x) for s in sources]
    # the dominant peak should be close to (20, 20)
    assert any(abs(py - 20) <= 3 and abs(px - 20) <= 3 for py, px in peaks)


def test_build_source_list_two_blobs():
    """Two well-separated blobs should produce at least two detected sources."""
    img = np.zeros((50, 50), dtype=np.float32)
    img += _make_blob((50, 50), center=(12, 12), radius=4, peak_value=10.0)
    img += _make_blob((50, 50), center=(37, 37), radius=4, peak_value=6.0)
    obs = _make_obs(img)
    sources = hierarchical_footprints(obs, flatten=True)

    assert len(sources) >= 2


def test_build_source_list_returns_scene_source():
    img = _make_blob((30, 30), center=(15, 15), radius=4, peak_value=8.0)
    obs = _make_obs(img)
    sources = hierarchical_footprints(obs)

    assert all(isinstance(s, HierarchicalFootprint) for s in sources)
    for s in sources:
        assert isinstance(s.peak, Peak)
        assert isinstance(s.footprint, np.ndarray)
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
    obs = _make_obs(img)
    all_sources = hierarchical_footprints(obs, flatten=True)

    peaks = [(s.peak.y, s.peak.x) for s in all_sources]
    assert any(abs(py - 38) <= 3 and abs(px - 38) <= 3 for py, px in peaks)


def test_build_source_list_flat_list():
    """flat_list=True returns all nodes with empty children lists."""
    img = np.zeros((50, 50), dtype=np.float32)
    img += _make_blob((50, 50), center=(12, 12), radius=4, peak_value=10.0)
    img += _make_blob((50, 50), center=(37, 37), radius=4, peak_value=6.0)
    obs = _make_obs(img)
    sources = hierarchical_footprints(obs, flatten=True)

    assert all(isinstance(s, HierarchicalFootprint) for s in sources)
    assert all(s.children == [] for s in sources)


def test_build_source_list_scale_coarsening():
    """Passing scales=[2, 3] restricts detection to coarser scales only."""
    img = np.zeros((50, 50), dtype=np.float32)
    img += _make_blob((50, 50), center=(25, 25), radius=6, peak_value=10.0)
    obs = _make_obs(img)

    sources_all = hierarchical_footprints(obs)
    sources_coarse = hierarchical_footprints(obs, scales=[2, 3])

    assert all(s.scale >= 2 for s in sources_coarse)
    assert len(sources_all) >= 1
    assert len(sources_coarse) >= 1
