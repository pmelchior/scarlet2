"""Detection utilities: connected-pixel footprint extraction and peak finding.

Low-level routines (``get_connected_pixels``, ``get_footprints``) are
translated from scarlet v1's ``detect_pybind11.cc``.  Higher-level helpers
(``get_wavelets``, ``QuadTreeRegion``, ``get_peaks``, …) are adapted from
scarlet v1's ``detect.py``.

Uses NumPy rather than JAX because detection involves dynamic data structures
(variable-length peak lists, irregularly shaped footprints) that are not
compatible with JAX's JIT.  Both NumPy and JAX arrays are accepted as inputs.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import heapq

import numpy as np
from scipy.ndimage import binary_fill_holes


from scipy.optimize import linear_sum_assignment

from .bbox import Box, overlap_slices
from .wavelets import get_multiresolution_support, starlet_transform

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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
    bounds : tuple of two (min, max) pairs
        Bounding box ``((y_min, y_max), (x_min, x_max))`` in the full image,
        with exclusive end coordinates (consistent with :class:`~scarlet2.bbox.Box`).
    """

    footprint: np.ndarray
    peaks: List[Peak]
    bounds: Tuple[Tuple[int, int], Tuple[int, int]]


# ---------------------------------------------------------------------------
# Low-level detection (translated from detect_pybind11.cc)
# ---------------------------------------------------------------------------


def get_connected_pixels(i, j, image, unchecked, footprint, bounds, thresh=0):
    """Find all pixels 4-connected to ``(i, j)`` that exceed ``thresh``.

    Uses an iterative flood-fill to avoid Python's recursion limit on large
    images.  Modifies ``unchecked``, ``footprint``, and ``bounds`` in-place.

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
    bounds : list of two [min, max] pairs
        Bounding box ``[[y_min, y_max], [x_min, x_max]]`` with exclusive end
        coordinates; updated in-place.
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
            if ci < bounds[0][0]:
                bounds[0][0] = ci
            if ci + 1 > bounds[0][1]:
                bounds[0][1] = ci + 1
            if cj < bounds[1][0]:
                bounds[1][0] = cj
            if cj + 1 > bounds[1][1]:
                bounds[1][1] = cj + 1
            if ci > 0:
                stack.append((ci - 1, cj))
            if ci < height - 1:
                stack.append((ci + 1, cj))
            if cj > 0:
                stack.append((ci, cj - 1))
            if cj < width - 1:
                stack.append((ci, cj + 1))


def _get_patch_peaks(image, min_separation, y0=0, x0=0):
    """Find local maxima in an image patch.

    A pixel is a local maximum if its value strictly exceeds all 8-connected
    neighbours that lie within the image boundary.

    Parameters
    ----------
    image : 2D ndarray
        The (possibly masked) image patch to search for peaks.
    min_separation : float
        Minimum pixel distance between peaks.  When two peaks are closer than
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


def footprints(image, min_separation=0, min_area=9, thresh=0):
    """Detect footprints and their peaks in an image.

    Iterates over every pixel, flood-fills each connected region of pixels
    above ``thresh``, filters by minimum area, and locates peaks within each
    footprint.

    Parameters
    ----------
    image : 2D array-like
        The image to detect sources in.  Accepts NumPy or JAX arrays.
    min_separation : float, optional
        Minimum pixel separation between peaks within a footprint.
    min_area : int, optional
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
    _footprints = []
    unchecked = np.ones((height, width), dtype=bool)
    footprint = np.zeros((height, width), dtype=bool)

    for i in range(height):
        for j in range(width):
            bounds = [[i, i + 1], [j, j + 1]]
            get_connected_pixels(i, j, image, unchecked, footprint, bounds, thresh)
            (y0, y1), (x0, x1) = bounds
            sub_h = y1 - y0
            sub_w = x1 - x0
            if sub_h * sub_w > min_area:
                sub_fp = footprint[y0:y1, x0:x1]
                if sub_fp.sum() >= min_area:
                    patch = image[y0:y1, x0:x1].copy()
                    patch[~sub_fp] = 0
                    peaks = _get_patch_peaks(patch, min_separation, y0=y0, x0=x0)
                    if peaks:
                        _footprints.append(
                            Footprint(sub_fp.copy(), peaks, ((y0, y1), (x0, x1)))
                        )
            footprint[y0:y1, x0:x1] = False

    return _footprints


# ---------------------------------------------------------------------------
# Box / footprint utilities (adapted from scarlet v1 detect.py)
# ---------------------------------------------------------------------------


def box_intersect(box1, box2):
    """Check whether two :class:`~scarlet2.bbox.Box` instances overlap.

    Parameters
    ----------
    box1, box2 : Box

    Returns
    -------
    overlap : bool
    """
    overlap = box1 & box2
    return overlap.shape[0] != 0 and overlap.shape[1] != 0


def footprint_intersect(footprint1, box1, footprint2, box2):
    """Check whether two footprint masks overlap.

    Parameters
    ----------
    footprint1, footprint2 : ndarray of bool
        The boolean masks for the two footprints, each sized to its own
        bounding box.
    box1, box2 : Box
        The corresponding bounding boxes.

    Returns
    -------
    overlap : bool
    """
    if not box_intersect(box1, box2):
        return False
    slices1, slices2 = overlap_slices(box1, box2)
    return np.sum(footprint1[slices1] * footprint2[slices2]) > 0


def footprint_iou(source1, source2):
    """Compute intersection over union (IoU) between two source footprints.

    Parameters
    ----------
    source1, source2 : :class:`HierarchicalFootprint`
        Sources with a `.bbox` (:class:`~scarlet2.bbox.Box`) and `.footprint`
        (boolean ndarray sized to that bbox).

    Returns
    -------
    iou : float
        IoU in ``[0, 1]``.  Returns ``0`` if the bounding boxes do not overlap.
    """
    def _mask(source):
        fp = source.footprint
        return fp.footprint if isinstance(fp, Footprint) else fp

    if not box_intersect(source1.bbox, source2.bbox):
        return 0.0
    slices1, slices2 = overlap_slices(source1.bbox, source2.bbox)
    mask1, mask2 = _mask(source1), _mask(source2)
    intersection = int(np.sum(mask1[slices1] & mask2[slices2]))
    union = int(np.sum(mask1)) + int(np.sum(mask2)) - intersection
    return float(intersection / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# QuadTree
# ---------------------------------------------------------------------------


class QuadTreeRegion:
    """A QuadTree that stores bounding boxes (rather than points).

    Boxes that span sub-region boundaries are stored in *all* overlapping
    sub-regions so that :meth:`query` always returns the full set of
    overlapping boxes.
    """

    def __init__(self, bbox, capacity=5, sub_regions=None, boxes=None, depth=0, detect=None):
        """
        Parameters
        ----------
        bbox : Box
            The box that encloses this region.
        capacity : int
            Maximum number of boxes before the region is split.
        sub_regions : list of QuadTreeRegion, optional
            Pre-existing sub-regions (normally left as ``None``).
        boxes : list of Box, optional
            Pre-existing boxes (normally left as ``None``).
        depth : int
            Depth of this node in the full tree (used for debugging).
        detect : array-like, optional
            Detection image; when provided enables debug visualisations.
        """
        self.bbox = bbox
        self.sub_regions = sub_regions
        self.boxes = boxes if boxes is not None else []
        self.capacity = capacity
        self.depth = depth
        self.detect = detect
        self.debug = detect is not None

    def footprint_image(self, bbox=None):
        """Return a 2-D image of all footprint masks in the tree.

        Parameters
        ----------
        bbox : Box, optional
            Output image bounding box.  If ``None``, the union of all
            footprint bounding boxes is used.

        Returns
        -------
        image : ndarray
        """
        boxes = self.query(self.bbox)

        if bbox is None:
            bbox = Box((0, 0))
            for box in boxes:
                bbox = bbox | box

        footprint = np.zeros(bbox.shape)
        for box in boxes:
            full, local = overlap_slices(bbox, box)
            footprint[full] += box.footprint.footprint[local]
        return footprint

    @property
    def peaks(self):
        """Yield all :class:`Peak` objects contained in the tree."""
        for box in self.query(self.bbox):
            yield from box.footprint.peaks

    def add(self, other_box):
        """Add a box to the region.

        Parameters
        ----------
        other_box : Box
            The box to insert.
        """
        if not box_intersect(self.bbox, other_box):
            return
        if self.sub_regions is not None:
            self._add_to_sub_regions(other_box)
            return
        if len(self.boxes) < self.capacity - 1:
            self.boxes.append(other_box)
        else:
            self.split()
            self.boxes = None
            self._add_to_sub_regions(other_box)

    def add_footprints(self, footprints):
        """Insert bounding boxes for a list of :class:`Footprint` objects.

        Each box gets a ``.footprint`` attribute pointing back to the
        originating :class:`Footprint` so it can be retrieved from a query.

        Parameters
        ----------
        footprints : list of Footprint

        Returns
        -------
        self : QuadTreeRegion
        """
        for fp in footprints:
            box = Box.from_bounds(*fp.bounds)
            box.footprint = fp
            self.add(box)
        return self

    def split(self):
        """Sub-divide this region into four quadrants."""
        import matplotlib.pyplot as plt

        height, width = self.bbox.shape
        h2 = height // 2
        w2 = width // 2
        h3 = height - h2
        w3 = width - w2

        if self.debug:
            fig, ax = plt.subplots()
            ax.imshow(self.detect[2], cmap="Greys")
            ax.set_title(self.depth)
            draw_box(self.bbox, ax, "r")
            for box in self.boxes:
                draw_box(box, ax, "b")

        origin = self.bbox.origin
        self.sub_regions = [
            QuadTreeRegion(Box((h2, w2), origin), capacity=self.capacity, depth=self.depth + 1),
            QuadTreeRegion(
                Box((h3, w2), (origin[0] + h2, origin[1])),
                capacity=self.capacity,
                depth=self.depth + 1,
            ),
            QuadTreeRegion(
                Box((h2, w3), (origin[0], origin[1] + w2)),
                capacity=self.capacity,
                depth=self.depth + 1,
            ),
            QuadTreeRegion(
                Box((h3, w3), (origin[0] + h2, origin[1] + w2)),
                capacity=self.capacity,
                depth=self.depth + 1,
            ),
        ]
        for box in self.boxes:
            self._add_to_sub_regions(box)

    def _add_to_sub_regions(self, other_box):
        for region in self.sub_regions:
            region.add(other_box)

    def query(self, other_box=None):
        """Return all boxes that overlap with ``other_box``.

        Parameters
        ----------
        other_box : Box, optional
            Query box.  Defaults to the full region bbox.

        Returns
        -------
        results : set of Box
            Boxes that overlap with ``other_box``.  A ``set`` is used so that
            boxes stored in multiple sub-regions are only returned once.
        """
        if other_box is None:
            other_box = self.bbox
        if self.boxes is not None:
            return {box for box in self.boxes if box_intersect(box, other_box)}
        if self.sub_regions is not None:
            results = set()
            for region in self.sub_regions:
                if box_intersect(region.bbox, other_box):
                    results |= region.query(other_box)
            return results
        return set()


# ---------------------------------------------------------------------------
# Multi-scale structure
# ---------------------------------------------------------------------------


class SingleScaleStructure:
    """A connected set of pixels with common peaks at a single wavelet scale.

    Using the terminology of Starck et al. 2011, a *structure* is a connected
    set of significant wavelet coefficients at a given scale, together with
    any peaks contributed by overlapping structures at other scales.

    Attributes
    ----------
    scale : int
        The wavelet scale of this structure.
    footprint : Footprint
        The footprint at the primary scale.
    bbox : Box
        Bounding box of the primary footprint.
    peaks : dict
        ``{scale: [Peak, …]}`` — peaks contributed from each scale.
    """

    def __init__(self, scale, footprint):
        """
        Parameters
        ----------
        scale : int
            Wavelet scale of the primary footprint.
        footprint : Footprint
        """
        self.scale = scale
        self.footprint = footprint
        self.bbox = Box.from_bounds(*footprint.bounds)
        self.peaks = {scale: footprint.peaks}
        self._all_peaks = None

    def add_footprint(self, scale, footprint):
        """Add peaks from a footprint at another scale.

        Parameters
        ----------
        scale : int
        footprint : Footprint
        """
        if scale not in self.peaks:
            self.peaks[scale] = []
        self.peaks[scale] += footprint.peaks
        self._all_peaks = None

    def add_scale_tree(self, scale, tree):
        """Add all footprints from a :class:`QuadTreeRegion` at another scale
        that overlap with this structure.

        Parameters
        ----------
        scale : int
        tree : QuadTreeRegion

        Returns
        -------
        self : SingleScaleStructure
        """
        for box in tree.query(self.bbox):
            self.add_footprint(scale, box.footprint)
        return self

    @property
    def all_peaks(self):
        """Set of ``(x, y)`` tuples for every peak across all scales."""
        if self._all_peaks is not None:
            return self._all_peaks
        all_peaks = set()
        for peaks in self.peaks.values():
            all_peaks |= {(peak.x, peak.y) for peak in peaks}
        self._all_peaks = all_peaks
        return self._all_peaks


# ---------------------------------------------------------------------------
# Wavelet-based detection helpers
# ---------------------------------------------------------------------------


def get_wavelets(images, variance, max_scale=3):
    """Compute significant starlet coefficients for a multi-band image cube.

    Parameters
    ----------
    images : array-like, shape (bands, height, width)
        Observed images.
    variance : array-like, shape (bands, height, width)
        Per-pixel variances matching ``images``.
    max_scale : int
        Number of wavelet scales.

    Returns
    -------
    coeffs : ndarray, shape (bands, max_scale+1, height, width)
        Starlet coefficients masked to the multi-resolution support.
    """
    images = np.asarray(images)
    variance = np.asarray(variance)
    sigma = np.median(np.sqrt(variance), axis=(-2, -1))
    coeffs = []
    for b, image in enumerate(images):
        _coeffs = np.asarray(starlet_transform(image, scales=max_scale))
        M, _ = get_multiresolution_support(image, _coeffs, sigma[b], K=3, epsilon=1e-1, max_iter=20)
        coeffs.append(M * _coeffs)
    return np.array(coeffs)


def get_detect_wavelets(images, variance, max_scale=3, K=3):
    """Get starlet coefficients of a detection image for source finding.

    The detection image is inverse varianced weighted sum of `images` across all bands.

    Parameters
    ----------
    images : array-like, shape (bands, height, width)
    variance : array-like, shape (bands, height, width)
    max_scale : int
        Number of wavelet scales.
    K: float
        The multiple of the coefficient scatter to calculate significance.
        Coefficients `w` with `|w| > K*sigma_j`, where `sigma_j` is
        the standard deviation at the jth scale, are considered significant.


    Returns
    -------
    coeffs : ndarray, shape (max_scale+1, height, width)
        Masked starlet coefficients of the summed detection image.
    sigma_j : ndarray, shape (max_scale+1,)
        Per-scale noise estimate used for thresholding, as returned by
        :func:`~scarlet2.wavelets.get_multiresolution_support`.
    """
    images = np.asarray(images)
    variance = np.asarray(variance)
    sigma = np.median(np.sqrt(variance), axis=(-2,-1))
    weights = 1/sigma**2 # inverse variance weighting, per band
    detect = np.sum(images * weights[:,None,None], axis=0) / np.sum(weights)
    sigma = np.sqrt(1/weights.sum())
    _coeffs = np.asarray(starlet_transform(detect, scales=max_scale))
    M, sigma_j = get_multiresolution_support(detect, _coeffs, sigma, K=K, epsilon=1e-1, max_iter=20)
    return M * _coeffs, np.asarray(sigma_j)


def get_blend_trees(detect, scales=None, min_separation=0, min_area=9, thresh=0):
    """Build a :class:`QuadTreeRegion` for each wavelet scale in ``detect``.

    Parameters
    ----------
    detect : ndarray, shape (scales+1, height, width)
        Masked starlet coefficients (e.g. from :func:`get_detect_wavelets`).
    scales : list of int, optional
        Indices into ``detect`` specifying which scales to use.  If ``None``
        (default) all scales are used.
    min_separation : float, optional
        Minimum pixel separation between peaks within a footprint.
    min_area : int, optional
        Minimum number of pixels a footprint must contain to be kept.
    thresh : float, optional
        Detection threshold; pixels must strictly exceed this value.

    Returns
    -------
    trees : list of QuadTreeRegion
        One tree per selected scale.
    all_footprints : list of list of Footprint
        Raw footprints at each selected scale (same ordering as ``trees``).
    """
    if scales is None:
        scales = list(range(len(detect)))
    else:
        scales = sorted(scales)

    all_footprints = []
    for s in scales:
        _footprints = footprints(
            np.asarray(detect[s]),
            min_separation=min_separation,
            min_area=min_area,
            thresh=thresh,
        )
        all_footprints.append(_footprints)

    trees = [
        QuadTreeRegion(Box(detect.shape[-2:]), capacity=10).add_footprints(fps)
        for fps in all_footprints
    ]
    return trees, all_footprints


def get_blend_structures(detect, scales=None, min_separation=0, min_area=9, thresh=0):
    """Build :class:`SingleScaleStructure` objects for the third wavelet scale.

    Each structure at the largest scale is linked to all overlapping footprints at
    finer scales, creating a hierarchy that connects fine-scale peaks to coarser detections.

    Parameters
    ----------
    detect : ndarray, shape (scales+1, height, width)
        Masked starlet coefficients (e.g. from :func:`get_detect_wavelets`).
    scales : list of int, optional
        Indices into ``detect`` specifying which scales to use.  If ``None``
        (default) all scales are used.
    min_separation : float, optional
        Minimum pixel separation between peaks within a footprint.
    min_area : int, optional
        Minimum number of pixels a footprint must contain to be kept.
    thresh : float, optional
        Detection threshold; pixels must strictly exceed this value.

    Returns
    -------
    structures : list of SingleScaleStructure
        Structures at largest scale with peaks from smaller scales attached.
    """
    if scales is None:
        scales = list(range(len(detect)))
    else:
        scales = sorted(scales)

    all_footprints = []
    for s in scales:
        _footprints = footprints(
            np.asarray(detect[s]),
            min_separation=min_separation,
            min_area=min_area,
            thresh=thresh,
        )
        all_footprints.append(_footprints)

    # start with the footprints at the largest selected scale
    structures = [ SingleScaleStructure(scales[-1], fp) for fp in all_footprints[-1] ]
    # add trees connecting to smaller selected scales
    box = Box(detect.shape[-2:])
    scale_trees = {
        scale: QuadTreeRegion(box, capacity=10).add_footprints(fps)
        for scale, fps in zip(scales[:-1], all_footprints[:-1])
    }
    for i in range(len(structures)):
        for scale, tree in scale_trees.items():
            structures[i].add_scale_tree(scale, tree)
    return structures


# ---------------------------------------------------------------------------
# Footprint splitting
# ---------------------------------------------------------------------------


def split_footprint(fp, image, min_area=0):
    """Split a multi-peak :class:`Footprint` into single-peak sub-footprints.

    Segments the footprint area by finding the saddle points between peaks using
    a priority-queue flooding watershed.  The wavelet coefficient image at the
    relevant scale is inverted so that peaks become low-cost basins; the watershed
    floods outward from each peak seed simultaneously in order of increasing cost,
    and region boundaries follow the intensity saddles between peaks.

    Parameters
    ----------
    fp : Footprint
        The footprint to split.  Returned unchanged (as a one-element list) if
        it contains at most one peak.
    image : 2D ndarray
        Wavelet coefficient image at the scale of ``fp``.
    min_area : int, optional
        Minimum number of pixels a sub-footprint must contain to be kept.
        Peaks whose watershed region is smaller than this are dropped.
        Default is ``0`` (keep all).

    Returns
    -------
    list of Footprint
        One single-peak :class:`Footprint` per peak in ``fp`` that meets the
        minimum area requirement.
    """
    if len(fp.peaks) <= 1:
        return [fp]

    (y0, y1), (x0, x1) = fp.bounds
    mask = fp.footprint
    sub_image = np.asarray(image[y0:y1, x0:x1], dtype=float)

    # Cost: invert intensity so peaks are cheap; normalize to [0, 1].
    vals = sub_image[mask]
    vmin, vmax = vals.min(), vals.max()
    if vmax > vmin:
        cost = np.where(mask, (vmax - sub_image) / (vmax - vmin), 1.0)
    else:
        cost = np.where(mask, 0.0, 1.0)

    # Priority-queue flooding watershed: expand from all seeds simultaneously
    # in order of increasing pixel cost (decreasing intensity).  A flood can
    # only reach a pixel through adjacent labeled pixels, so it cannot arc
    # around another seed's region.
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    heap = []
    _nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for k, peak in enumerate(fp.peaks):
        py, px = peak.y - y0, peak.x - x0
        labels[py, px] = k + 1
        for di, dj in _nbrs:
            ni, nj = py + di, px + dj
            if 0 <= ni < h and 0 <= nj < w and mask[ni, nj] and labels[ni, nj] == 0:
                heapq.heappush(heap, (cost[ni, nj], ni, nj, k + 1))

    while heap:
        c, i, j, label = heapq.heappop(heap)
        if labels[i, j] != 0:
            continue
        labels[i, j] = label
        for di, dj in _nbrs:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and mask[ni, nj] and labels[ni, nj] == 0:
                heapq.heappush(heap, (cost[ni, nj], ni, nj, label))

    sub_footprints = []
    for k, peak in enumerate(fp.peaks):
        raw_region = (labels == k + 1) & mask
        if raw_region.sum() < min_area:
            continue
        region = binary_fill_holes(raw_region)
        if not region.any():
            continue
        rows = np.where(region.any(axis=1))[0]
        cols = np.where(region.any(axis=0))[0]
        ry0, ry1 = int(rows[0]), int(rows[-1]) + 1
        rx0, rx1 = int(cols[0]), int(cols[-1]) + 1
        bounds = ((y0 + ry0, y0 + ry1), (x0 + rx0, x0 + rx1))
        sub_footprints.append(Footprint(region[ry0:ry1, rx0:rx1], [peak], bounds))

    return sub_footprints


# ---------------------------------------------------------------------------
# Hierarchical source list
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalFootprint:
    """A source detected in the starlet hierarchy.

    Attributes
    ----------
    center : tuple of int
        ``(y, x)`` peak position, refined to the finest scale reached.
    bbox : Box
        Bounding box at the scale this source was first detected.
    footprint : np.ndarray
        Binary mask for detected source in bbox.
    scale : int
        Largest wavelet scale at which this source was first identified.
    children : list of HierarchicalFootprint
        Sources whose peaks lie inside this source's footprint and are
        spatially inconsistent with this source's primary peak.
    """
    center: Tuple[int, int]
    bbox: Box
    footprint: np.ndarray
    scale: int
    children: List["HierarchicalFootprint"] = field(default_factory=list)


def hierarchical_footprints(detect, flatten=True, scales=None, catalog=None, sigma_scales=None, K=3, split_peaks=True, min_separation=0, min_area=9, thresh=0):
    """Decompose a detection image into a hierarchy of :class:`HierarchicalFootprint` objects.

    Iterates from the largest starlet scale to the smallest.  At each scale,
    every detected footprint is matched to the best-overlapping source already
    registered from larger scales (measured by IoU).  If the registered source's
    center lies inside the new footprint, it is a *primary* match: the source
    center is refined and its footprint is grown to the union.  Otherwise the new
    footprint becomes a *child* of the best-matching source.  Footprints with no
    overlap with any registered source are promoted to new top-level sources.

    When ``split_peaks`` is ``True`` (default), footprints that contain more than
    one peak are split into separate sources by using a watershed algorithm.
    Otherwise, additional peaks become children of the originating footprint.

    If ``catalog`` is provided, the detected sources are matched to the catalog
    positions via a global bipartite assignment (Hungarian algorithm).  The cost
    of assigning catalog entry ``i`` to source ``j`` is the squared distance from
    the catalog position to the nearest peak of source ``j``, restricted to cases
    where the catalog position falls inside source ``j``'s footprint mask.
    Catalog entries with no containing footprint are returned as ``None``.

    If ``sigma_scales`` is provided, each footprint's bounding box is grown beyond
    the detection threshold to the noise level.  Assuming an exponential profile
    I(r) = I0*exp(-r/h), the scale length h is estimated from the footprint size
    as h = r_foot / ln(S), where r_foot is the mean distance from the peak to the
    edge of the footprint bounding box and S = I0 / (K*sigma_j).  The box is grown
    to the radius where the profile reaches the noise level (1*sigma_j):
    half_size = r_foot * ln(S*K) / ln(S).

    Parameters
    ----------
    detect : ndarray, shape (scales+1, H, W)
        Masked starlet coefficients from :func:`get_detect_wavelets`.
    flatten : bool, optional
        Whether to flatten the source list so that children appear as independent
        entries.  Default ``True``.
    scales : list of int, optional
        Indices into ``detect`` specifying which planes to use.  If ``None``
        (default) all planes are used.
    catalog : list of (y, x) tuples, optional
        If given, the output is catalog-indexed: one entry per catalog position,
        matched to the best overlapping detected source, or ``None`` if undetected.
        Matching is a global optimal assignment — each source is assigned to at
        most one catalog entry.
    sigma_scales : array-like, shape (max_scale+1,), optional
        Per-scale noise estimate from :func:`~scarlet2.detect.get_detect_wavelets`.
        When provided, each footprint's bounding box is grown to the noise level
        using the SNR-based exponential-profile formula described above.
    K : float, optional
        Detection threshold multiplier used when ``sigma_scales`` is given.
        Must match the value passed to :func:`~scarlet2.detect.get_detect_wavelets`.
        Default ``3``.
    split_peaks : bool, optional
        If ``True`` (default), footprints with multiple peaks are split into separate sources
        using a watershed algorithm. Otherwise, additional peaks become children of the
        originating footprint,  which retains the full footprint area, i.e. the children overlap.
        Splitting peaks allows to reduce the overlap of mostly independent sources.
    min_separation : float, optional
        Minimum pixel separation between peaks within a footprint.
    min_area : int, optional
        Minimum number of pixels a footprint must contain to be kept.  Also used
        as the minimum area for watershed sub-footprints when ``split_peaks`` is
        ``True``.
    thresh : float, optional
        Detection threshold; pixels must strictly exceed this value.

    Returns
    -------
    sources : list of HierarchicalFootprint or None
        When ``catalog`` is ``None``: top-level sources, each potentially carrying
        a tree of children.  Sources detected only at finer scales appear as
        additional top-level entries with their ``scale`` set accordingly.
        When ``catalog`` is given: catalog-length list where each entry is the
        matched :class:`HierarchicalFootprint`, or ``None`` if no source was
        detected at that catalog position.
    """
    def snr_bbox(fp, scale):
        """Bounding box grown to the SNR-predicted extent of an exponential profile.

        For a profile I(r) = I0*exp(-r/h), the footprint boundary lies at the
        detection threshold K*sigma_j, so h = r_foot / ln(S) where
        S = I0/(K*sigma_j) and r_foot is the mean distance from the peak to the
        edge of the footprint bbox.  The box is grown to the radius
        where the profile drops to the noise level (1*sigma_j):
        ``half_size = r_foot * ln(S*K) / ln(S)``.
        The result is unioned with the tight footprint bbox so the box never shrinks.
        """
        bbox = Box.from_bounds(*fp.bounds)
        outer_box = Box(detect.shape[1:])
        if sigma_scales is None:
            return bbox
        sigma = float(sigma_scales[scale])
        if sigma <= 0:
            return bbox
        w_peak = fp.peaks[0].flux
        S = w_peak / (K * sigma)
        if S <= 1:
            return bbox
        peak = fp.peaks[0]
        (y0, y1), (x0, x1) = fp.bounds
        r_foot = np.mean([peak.y - y0, y1 - peak.y, peak.x - x0, x1 - peak.x])
        half_size = int(np.ceil(r_foot * np.log(S * K) / np.log(S)))
        snr_box = Box(
            (2 * half_size + 1, 2 * half_size + 1),
            origin=(peak.y - half_size, peak.x - half_size),
        )
        return (bbox | snr_box) & outer_box

    def peak_in_footprint(y, x, fp):
        """True if pixel (y, x) lies inside the boolean mask of Footprint fp."""
        ly = y - fp.bounds[0][0]
        lx = x - fp.bounds[1][0]
        h, w = fp.footprint.shape
        return 0 <= ly < h and 0 <= lx < w and bool(fp.footprint[ly, lx])

    def all_nodes(node_list):
        """Yield every SceneSource in the tree rooted at each node in node_list."""
        for node in node_list:
            yield node
            if node is not None:
                yield from all_nodes(node.children)

    def peaks2children(fp, scale):
        """Return children for additional peaks in ``fp`` using watershed sub-footprints.
        """
        children = []
        for sub_fp in split_footprint(fp, detect[scale], min_area=min_area)[1:]:
            child = HierarchicalFootprint(
                center=(sub_fp.peaks[0].y, sub_fp.peaks[0].x),
                bbox=Box.from_bounds(*sub_fp.bounds),
                footprint=sub_fp,
                scale=scale,
            )
            children.append(child)
        return children


    if scales is None:
        scales = scale_indices = list(range(len(detect)))
    else:
        scales = sorted(scales)
        scale_indices = list(range(len(scales)))

    all_footprints = [
        footprints(
            detect[s],
            min_separation=min_separation,
            min_area=min_area,
            thresh=thresh,
        )
        for s in scales
    ]

    if split_peaks:
        # Pre-split multi-peak footprints so that all footprints at each scale
        # are non-overlapping and carry exactly one peak.
        all_footprints = [
            [sub for fp in fps for sub in split_footprint(fp, detect[s], min_area=min_area)]
            for s, fps in zip(scales, all_footprints)
        ]

    # --- initial hierarchy at largest scale: additional peaks become children -
    sources = []
    idx = -1
    for fp in all_footprints[idx]:
        node = HierarchicalFootprint(
            center=(fp.peaks[0].y, fp.peaks[0].x),
            bbox=Box.from_bounds(*fp.bounds),
            footprint=fp,
            scale=scales[idx],
            children=peaks2children(fp, scales[idx]),
        )
        sources.append(node)

    # --- link smaller scale footprints to larger scale footprints -
    for idx in scale_indices[:-1][::-1]:  # exclude largest scale, 2nd largest to smallest
        registered_nodes = list(all_nodes(sources))

        for fp in all_footprints[idx]:
            peak = fp.peaks[0]
            node = HierarchicalFootprint(
                center=(peak.y, peak.x),
                bbox=Box.from_bounds(*fp.bounds),
                footprint=fp,
                scale=scales[idx],
                children=peaks2children(fp, scales[idx]),
            )

            # new fps are either matches to an existing source, to one of their children, or an orphan
            overlapping = [
                i for i, rfp in enumerate(registered_nodes)
                if peak_in_footprint(peak.y, peak.x, rfp.footprint)
            ]
            if len(overlapping) == 0: # orphan: add to sources
                sources.append(node)
            else:
                # determine best match: intersection over union
                overlap = {i: footprint_iou(node, registered_nodes[i]) for i in overlapping}
                max_i = max(overlap, key=overlap.get)
                parent = registered_nodes[max_i]

                # if peak of parent is in footprint of new fp: primary match
                # use fp center to refine parent center and update footprint with union
                cy, cx = parent.center
                if peak_in_footprint(cy, cx, fp):
                    parent.center = node.center
                    # union bbox and footprint mask with the primary fp at this finer scale
                    primary_bbox = Box.from_bounds(*fp.bounds)
                    parent_bbox = Box.from_bounds(*parent.footprint.bounds)
                    union_bbox = parent_bbox | primary_bbox
                    union_mask = np.zeros(union_bbox.shape, dtype=bool)
                    p_slices, pp_slices = overlap_slices(union_bbox, parent_bbox)
                    union_mask[p_slices] |= parent.footprint.footprint[pp_slices]
                    q_slices, qp_slices = overlap_slices(union_bbox, primary_bbox)
                    union_mask[q_slices] |= fp.footprint[qp_slices]
                    parent.footprint = Footprint(union_mask, parent.footprint.peaks, union_bbox.bounds)
                    parent.bbox = union_bbox
                # if not: new child
                else:
                    parent.children.append(node)

    # --- flatten list: children are listed separately -
    if flatten:
        sources = list(all_nodes(sources))

    # --- catalog filter: keep only footprints containing a given sky position -
    if catalog is not None:
        # Build cost matrix: rows = catalog entries, cols = detected sources.
        # Cost is squared distance from catalog position to nearest source peak,
        # restricted to cases where the catalog position falls inside the footprint.
        # np.inf marks invalid (position outside footprint) pairs.
        _no_match = 1e18
        cost = np.full((len(catalog), len(sources)), _no_match)
        for i, (py, px) in enumerate(catalog):
            for j, s in enumerate(sources):
                if peak_in_footprint(int(py), int(px), s.footprint):
                    cost[i, j] = min(
                        (py - p.y) ** 2 + (px - p.x) ** 2
                        for p in s.footprint.peaks
                    )

        # Solve the global assignment problem; pairs with sentinel cost are unmatched.
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = [None] * len(catalog)
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < _no_match:
                matches[i] = j

        sources = [sources[j] if j is not None else None for j in matches]

    # clean up list: pad footprint mask to match the (possibly enlarged) bbox,
    # then replace the Footprint object with the plain boolean array.
    for i in range(len(sources)):
        # if limit_to is used, we can get None for non-detections, and we can have
        # the several limit_to centers point to the same detection.
        # in either case: don't postprocess them (again)
        if sources[i] is None or not isinstance(sources[i].footprint, Footprint):
            continue
        fp_obj = sources[i].footprint # still a Footprint at this point
        enlarged_bbox = snr_bbox(fp_obj, sources[i].scale)
        tight_bbox = Box.from_bounds(*fp_obj.bounds)
        padded = np.zeros(enlarged_bbox.shape, dtype=bool)
        enlarged_slices, tight_slices = overlap_slices(enlarged_bbox, tight_bbox)
        padded[enlarged_slices] = fp_obj.footprint[tight_slices]
        sources[i].bbox = enlarged_bbox # now only numpy array of footprint
        sources[i].footprint = padded
        if flatten:
            sources[i].children = []

    return sources