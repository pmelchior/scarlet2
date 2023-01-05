from dataclasses import dataclass
import jax
import jax.numpy as jnp

@dataclass
class Box:
    """Bounding Box for an object

    A Bounding box describes the location of a data unit in the global/model coordinate
    system. It is used to identify spatial and channel overlap and to map from model
    to observed frames and back.

    The `BBox` code is agnostic about the meaning of the dimensions.
    We generally use this convention:

    - 2D shapes denote (Height, Width)
    - 3D shapes denote (Channels, Height, Width)
    """

    shape: jax.numpy.ndarray
    origin: jax.numpy.ndarray

    def __init__(self, shape, origin=None):
        self.shape = jnp.array(shape).astype(int)
        if origin is None:
            origin = jnp.zeros(len(self.shape), dtype=int)
        self.origin = jnp.array(origin)

    @staticmethod
    def from_bounds(*bounds):
        """Initialize a box from its bounds

        Parameters
        ----------
        bounds: tuple of (min,max) pairs
            Min/Max coordinate for every dimension

        Returns
        -------
        bbox: :class:`scarlet.bbox.Box`
            A new box bounded by the input bounds.
        """
        shape = jnp.array([max(0, cmax - cmin) for cmin, cmax in bounds])
        origin = jnp.array([cmin for cmin, cmax in bounds])
        return Box(shape, origin=origin)

    def contains(self, p):
        """Whether the box contains a given coordinate `p`
        """
        if len(p) != self.D:
            raise ValueError(f"Dimension mismatch in {p} and {self.D}")

        for d in range(self.D):
            if p[d] < self.origin[d] or p[d] >= self.origin[d] + self.shape[d]:
                return False
        return True

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
        return self.origin + self.shape

    @property
    def center(self):
        """Tuple of center coordinates
        """
        return self.origin + self.shape / 2

    @property
    def bounds(self):
        """Bounds of the box
        """
        return jnp.stack((self.start, self.stop), axis=1)

    def set_center(self, pos):
        """Center box at given position
        """
        self.origin = pos - self.center

    def grow(self, delta):
        """Grow the Box by the given delta in each direction
        """
        if not hasattr(delta, "__iter__"):
            radius = [delta] * self.D
        delta = jnp.array(delta)
        origin = self.origin - delta
        shape = self.shape + 2*delta
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
        return Box(self.shape[i], origin=self.origin[i])

    def __repr__(self):
        result = "<Box shape={0}, origin={1}>"
        return result.format(self.shape, self.origin)

    def __iadd__(self, offset):
        self.origin = self.origin + offset
        return self

    def __add__(self, offset):
        return self.__copy__().__iadd__(offset)

    def __isub__(self, offset):
        self.origin = self.origin - offset
        return self

    def __sub__(self, offset):
        return self.__copy__().__isub__(offset)

    def __imatmul__(self, bbox):
        bounds = jnp.concatenate((self.bounds, bbox.bounds))
        self = Box.from_bounds(*bounds)
        return self

    def __matmul__(self, bbox):
        return self.__copy__().__imatmul__(bbox)

    def __copy__(self):
        return Box(self.shape, origin=self.origin)

    def copy(self):
        """Copy of the box
        """
        return self.__copy__()

    def __eq__(self, other):
        return self.shape == other.shape and self.origin == other.origin

    def __hash__(self):
        return hash((self.shape, self.origin))