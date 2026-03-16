"""Detection utilities: connected-pixel footprint extraction and peak finding.

Translated from scarlet v1's ``detect_pybind11.cc``. Uses NumPy rather than
JAX because detection involves dynamic data structures (variable-length peak
lists, irregularly shaped footprints) that are not compatible with JAX's JIT.
Both NumPy and JAX arrays are accepted as inputs.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Peak:
    """A peak (local maximum) in a :class:`Footprint`.

    Attributes
    ----------
    y : int
        Row index in the full image.
    x : int
        Column index in the full image.
    flux : float
        Pixel value at the peak location.
    """

    y: int
    x: int
    flux: float


@dataclass
class Footprint:
    """A detected footprint (connected region above threshold) in an image.

    Attributes
    ----------
    footprint : ndarray of bool, shape (height, width)
        Boolean mask of the footprint pixels, sized to the bounding box.
    peaks : list of Peak
        Peaks found within this footprint, sorted brightest-first.
    bounds : ndarray of int, shape (4,)
        Bounding box ``[y_min, y_max, x_min, x_max]`` in the full image.
    """

    footprint: np.ndarray
    peaks: List[Peak]
    bounds: np.ndarray


def get_connected_pixels(i, j, image, unchecked, footprint, bounds, thresh=0):
    """Find all pixels 4-connected to ``(i, j)`` that exceed ``thresh``.

    Uses an iterative flood-fill to avoid Python's recursion limit on large
    images. Modifies ``unchecked``, ``footprint``, and ``bounds`` in-place.

    Parameters
    ----------
    i, j : int
        Seed pixel coordinates (row, column).
    image : 2D ndarray
        The image to search.
    unchecked : 2D boolean ndarray
        Tracks unvisited pixels; updated in-place.
    footprint : 2D boolean ndarray
        Accumulates the footprint mask; updated in-place.
    bounds : ndarray of int, shape (4,)
        Bounding box ``[y_min, y_max, x_min, x_max]``; updated in-place.
    thresh : float, optional
        Pixels must strictly exceed this value to join the footprint.
    """
    height, width = image.shape
    stack = [(i, j)]
    while stack:
        ci, cj = stack.pop()
        if not unchecked[ci, cj]:
            continue
        unchecked[ci, cj] = False
        if image[ci, cj] > thresh:
            footprint[ci, cj] = True
            if ci < bounds[0]:
                bounds[0] = ci
            elif ci > bounds[1]:
                bounds[1] = ci
            if cj < bounds[2]:
                bounds[2] = cj
            elif cj > bounds[3]:
                bounds[3] = cj
            if ci > 0:
                stack.append((ci - 1, cj))
            if ci < height - 1:
                stack.append((ci + 1, cj))
            if cj > 0:
                stack.append((ci, cj - 1))
            if cj < width - 1:
                stack.append((ci, cj + 1))


def get_peaks(image, min_separation, y0=0, x0=0):
    """Find local maxima in an image patch.

    A pixel is a local maximum if its value strictly exceeds all 8-connected
    neighbours that lie within the image boundary.

    Parameters
    ----------
    image : 2D ndarray
        The (possibly masked) image patch to search for peaks.
    min_separation : float
        Minimum pixel distance between peaks. When two peaks are closer than
        this, the dimmer one is removed.
    y0, x0 : int, optional
        Row and column offsets added to peak coordinates so they refer to the
        full image rather than the patch.

    Returns
    -------
    peaks : list of Peak
        Local maxima sorted brightest-first, with coordinates in the full image.
    """
    height, width = image.shape
    peaks = []

    for i in range(height):
        for j in range(width):
            val = image[i, j]
            # 4-connected neighbours
            if i > 0 and val <= image[i - 1, j]:
                continue
            if i < height - 1 and val <= image[i + 1, j]:
                continue
            if j > 0 and val <= image[i, j - 1]:
                continue
            if j < width - 1 and val <= image[i, j + 1]:
                continue
            # diagonal neighbours
            if i > 0 and j > 0 and val <= image[i - 1, j - 1]:
                continue
            if i < height - 1 and j < width - 1 and val <= image[i + 1, j + 1]:
                continue
            if i < height - 1 and j > 0 and val <= image[i + 1, j - 1]:
                continue
            if i > 0 and j < width - 1 and val <= image[i - 1, j + 1]:
                continue
            peaks.append(Peak(i + y0, j + x0, float(val)))

    # Sort brightest first
    peaks.sort(key=lambda p: p.flux, reverse=True)

    # Remove peaks within min_separation of a brighter peak
    min_separation2 = min_separation**2
    i = 0
    while i < len(peaks):
        j = i + 1
        while j < len(peaks):
            dy = peaks[i].y - peaks[j].y
            dx = peaks[i].x - peaks[j].x
            if dy * dy + dx * dx < min_separation2:
                peaks.pop(j)
            else:
                j += 1
        i += 1

    return peaks


def get_footprints(image, min_separation, min_area, thresh=0):
    """Detect footprints and their peaks in an image.

    Iterates over every pixel, flood-fills each connected region of pixels
    above ``thresh``, filters by minimum area, and locates peaks within each
    footprint.

    Parameters
    ----------
    image : 2D array-like
        The image to detect sources in. Accepts NumPy or JAX arrays.
    min_separation : float
        Minimum pixel separation between peaks within a footprint.
    min_area : int
        Minimum number of pixels a footprint must contain to be kept.
    thresh : float, optional
        Detection threshold; pixels must strictly exceed this value.

    Returns
    -------
    footprints : list of Footprint
        Detected footprints, each containing the boolean mask (sized to the
        bounding box), peak list, and bounding box in the full image.
    """
    image = np.asarray(image)
    height, width = image.shape
    footprints = []
    unchecked = np.ones((height, width), dtype=bool)
    footprint = np.zeros((height, width), dtype=bool)

    for i in range(height):
        for j in range(width):
            bounds = np.array([i, i, j, j], dtype=int)
            get_connected_pixels(i, j, image, unchecked, footprint, bounds, thresh)
            sub_h = bounds[1] - bounds[0] + 1
            sub_w = bounds[3] - bounds[2] + 1
            if sub_h * sub_w > min_area:
                sub_fp = footprint[bounds[0] : bounds[1] + 1, bounds[2] : bounds[3] + 1]
                if sub_fp.sum() >= min_area:
                    patch = image[bounds[0] : bounds[1] + 1, bounds[2] : bounds[3] + 1].copy()
                    patch[~sub_fp] = 0
                    peaks = get_peaks(patch, min_separation, y0=int(bounds[0]), x0=int(bounds[2]))
                    if peaks:
                        footprints.append(Footprint(sub_fp.copy(), peaks, bounds.copy()))
            footprint[bounds[0] : bounds[1] + 1, bounds[2] : bounds[3] + 1] = False

    return footprints
