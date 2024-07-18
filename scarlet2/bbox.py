import equinox as eqx
import jax.numpy as jnp

class Box(eqx.Module):
    """Bounding Box for an object

    A Bounding box describes the location of a data unit in the global/model coordinate
    system. It is used to identify spatial and channel overlap and to map from model
    to observed frames and back.

    The `BBox` code is agnostic about the meaning of the dimensions.
    We generally use this convention:

    - 2D shapes denote (Height, Width)
    - 3D shapes denote (Channels, Height, Width)
    """

    shape: tuple
    origin: tuple

    def __init__(self, shape, origin=None):
        self.shape = tuple(shape)
        if origin is None:
            origin = (0,) * len(shape)
        self.origin = tuple(origin)

    @staticmethod
    def from_bounds(*bounds):
        """Initialize a box from its bounds

        Parameters
        ----------
        bounds: tuple of (min,max) pairs
            Min/Max coordinate for every dimension

        Returns
        -------
        bbox: :class:`scarlet2.bbox.Box`
            A new box bounded by the input bounds.
        """
        shape = tuple(max(0, cmax - cmin) for cmin, cmax in bounds)
        origin = (cmin for cmin, cmax in bounds)
        return Box(shape, origin=origin)

    @staticmethod
    def from_data(X, min_value=0):
        """Define box where `X` is above `min_value`

        Parameters
        ----------
        X: jnp.ndarray
            Data to threshold
        min_value: float
            Minimum value of the result.

        Returns
        -------
        bbox: :class:`scarlet2.bbox.Box`
            Bounding box for the thresholded `X`
        """
        sel = X > min_value
        if sel.any():
            nonzero = jnp.where(sel)
            bounds = []
            for dim in range(len(X.shape)):
                bounds.append((nonzero[dim].min(), nonzero[dim].max() + 1))
        else:
            bounds = [[0, 0]] * len(X.shape)
        return Box.from_bounds(*bounds)

    def contains(self, p):
        """Whether the box contains a given coordinate `p`
        """
        if len(p) != self.D:
            raise ValueError(f"Dimension mismatch in {p} and {self.D}")

        for d in range(self.D):
            if p[d] < self.origin[d] or p[d] >= self.origin[d] + self.shape[d]:
                return False
        return True

    def insert_into(self, image, sub):
        """Insert `sub` into `image` according to this bbox

        Inverse operation to :func:`~scarlet.bbox.Box.extract_from`.

        Parameters
        ----------
        image: array
            Full image
        sub: array
            Extracted sub-image

        Returns
        -------
        image: array
        """
        imbox = Box(image.shape)

        im_slices, sub_slices = overlap_slices(imbox, self)
        image[im_slices] = sub[sub_slices]
        return image


    @property
    def D(self):
        """Dimensionality of this BBox
        """
        return len(self.shape)

    @property
    def start(self):
        """Tuple of start coordinates
        """
        return self.origin

    @property
    def stop(self):
        """Tuple of stop coordinates
        """
        return tuple(o + s for o, s in zip(self.origin, self.shape))

    @property
    def center(self):
        """Tuple of center coordinates
        """
        return tuple(o + s // 2 for o, s in zip(self.origin, self.shape))

    @property
    def bounds(self):
        """Bounds of the box
        """
        return tuple((o, o + s) for o, s in zip(self.origin, self.shape))

    @property
    def slices(self):
        """Bounds of the box as slices
        """
        return tuple([slice(o, o + s) for o, s in zip(self.origin, self.shape)])

    @property
    def spatial(self):
        """Spatial component of higher-dimensional box"""
        assert self.D >= 2
        return self[-2:]

    def set_center(self, pos):
        """Center box at given position
        """
        pos_ = tuple(_.item() for _ in pos)
        origin = tuple(o + p - c for o, p, c in zip(self.origin, pos_, self.center))
        object.__setattr__(self, 'origin', origin)

    def grow(self, delta):
        """Grow the Box by the given delta in each direction
        """
        if not hasattr(delta, "__iter__"):
            delta = [delta] * self.D
        origin = tuple([self.origin[d] - delta[d] for d in range(self.D)])
        shape = tuple([self.shape[d] + 2 * delta[d] for d in range(self.D)])
        return Box(shape, origin=origin)

    def shrink(self, delta):
        """Shrink the Box by the given delta in each direction
        """
        if not hasattr(delta, "__iter__"):
            delta = [delta] * self.D
        origin = tuple([self.origin[d] + delta[d] for d in range(self.D)])
        shape = tuple([self.shape[d] - 2 * delta[d] for d in range(self.D)])
        return Box(shape, origin=origin)

    def __or__(self, other):
        """Union of two bounding boxes

        Parameters
        ----------
        other: `Box`
            The other bounding box in the union

        Returns
        -------
        result: `Box`
            The smallest rectangular box that contains *both* boxes.
        """
        if other.D != self.D:
            raise ValueError(f"Dimension mismatch in the boxes {other} and {self}")
        bounds = []
        for d in range(self.D):
            bounds.append(
                (min(self.start[d], other.start[d]), max(self.stop[d], other.stop[d]))
            )
        return Box.from_bounds(*bounds)

    def __and__(self, other):
        """Intersection of two bounding boxes

        If there is no intersection between the two bounding
        boxes then an empty bounding box is returned.

        Parameters
        ----------
        other: `Box`
            The other bounding box in the intersection

        Returns
        -------
        result: `Box`
            The rectangular box that is in the overlap region
            of both boxes.
        """
        if other.D != self.D:
            raise ValueError(f"Dimension mismatch in the boxes {other} and {self}")
        assert other.D == self.D
        bounds = []
        for d in range(self.D):
            bounds.append(
                (max(self.start[d], other.start[d]), min(self.stop[d], other.stop[d]))
            )
        return Box.from_bounds(*bounds)

    def __getitem__(self, i):
        s_ = self.shape[i]
        o_ = self.origin[i]
        if not hasattr(s_, "__iter__"):
            s_ = (s_,)
            o_ = (o_,)
        return Box(s_, origin=o_)

    def __add__(self, offset):
        if not hasattr(offset, "__iter__"):
            offset = (offset,) * self.D
        origin = tuple([a + o for a, o in zip(self.origin, offset)])
        return Box(self.shape, origin=origin)

    def __sub__(self, offset):
        if not hasattr(offset, "__iter__"):
            offset = (offset,) * self.D
        origin = tuple([a - o for a, o in zip(self.origin, offset)])
        return Box(self.shape, origin=origin)

    def __matmul__(self, bbox):
        bounds = self.bounds + bbox.bounds
        return Box.from_bounds(*bounds)

    def __copy__(self):
        return Box(self.shape, origin=self.origin)

    def __eq__(self, other):
        return self.shape == other.shape and self.origin == other.origin

    def __hash__(self):
        return hash((self.shape, self.origin))


def overlap_slices(bbox1, bbox2, return_boxes=False):
    """Slices of bbox1 and bbox2 that overlap

    Parameters
    ----------
    bbox1: `~scarlet.bbox.Box`
    bbox2: `~scarlet.bbox.Box`

    Returns
    -------
    slices: tuple of slices
        The slice of an array bounded by `bbox1` and
        the slice of an array bounded by `bbox2` in the
        overlapping region.
    """
    overlap = bbox1 & bbox2
    _bbox1 = overlap - bbox1.origin
    _bbox2 = overlap - bbox2.origin
    if return_boxes:
        return _bbox1, _bbox2
    slices = (
        _bbox1.slices,
        _bbox2.slices,
    )
    return slices
