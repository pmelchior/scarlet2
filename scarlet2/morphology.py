import equinox as eqx
import jax.numpy as jnp
import jax.scipy

from .bbox import Box
from .module import Module
from .wavelet import starlet_reconstruction
from .dusty_nn import pad_it, crop

class Morphology(Module):
    bbox: Box = eqx.field(static=True, init=False)

    def center_bbox(self, center):
        center_ = tuple(_.item() for _ in center.astype(int))
        self.bbox.set_center(center_)


class ArrayMorphology(Morphology):
    data: jnp.array

    def __init__(self, data):
        self.data = data
        super().__post_init__()
        self.bbox = Box(self.data.shape)

    def __call__(self):
        return self.data


class GaussianMorphology(Morphology):
    center: jnp.ndarray
    sigma: float

    def __init__(self, center, sigma, bbox=None):
        self.sigma = sigma
        self.center = center
        super().__post_init__()

        if bbox is None:
            max_sigma = jnp.max(self.sigma)
            # explicit call to int() to avoid bbox sizes being jax-traced
            size = 10 * int(jnp.ceil(max_sigma))
            if size % 2 == 0:
                size += 1
            center_int = jnp.floor(self.center)
            shape = (size, size)
            origin = (int(center_int[0]) - size // 2, int(center_int[1]) - size // 2)
            bbox = Box(shape, origin=origin)
        self.bbox = bbox

    def __call__(self):
        # grid positions in X/Y
        _Y = jnp.arange(self.bbox.shape[-2]) + self.bbox.origin[-2]
        _X = jnp.arange(self.bbox.shape[-1]) + self.bbox.origin[-1]

        # with pixel integration
        f = lambda x, s: 0.5 * (
                1 - jax.scipy.special.erfc((0.5 - x) / jnp.sqrt(2) / s) +
                1 - jax.scipy.special.erfc((0.5 + x) / jnp.sqrt(2) / s)
        )
        # # without pixel integration
        # f = lambda x, s: jnp.exp(-(x ** 2) / (2 * s ** 2)) / (jnp.sqrt(2 * jnp.pi) * s)

        return jnp.outer(f(_Y - self.center[0], self.sigma), f(_X - self.center[1], self.sigma))

class StarletMorphology(Morphology):
    coeffs: jnp.ndarray

    def __init__(self, coeffs):
        self.coeffs = coeffs
        super().__post_init__()
        self.bbox = Box(self.coeffs.shape[1:])

    def __call__(self):
        return starlet_reconstruction(self.coeffs)

import numpy as np
from scipy.special import gammaincinv
from astropy.modeling.models import Sersic2D

class Sersic2DMorphology(Morphology):

    x0: jnp.array
    y0: jnp.array
    X: jnp.ndarray
    Y: jnp.ndarray
    ellip: jnp.array
    theta: jnp.array
    r_eff: jnp.array
    n: jnp.array
    amplitude : jnp.array

    def __init__(self, x0, y0, X, Y,
                 ellip, theta, r_eff, n,
                 amplitude = jnp.array(1) ):
        
        self.x0 = x0
        self.y0 = y0
        self.X = X
        self.Y = Y
        self.ellip  = ellip
        self.theta  = theta
        self.r_eff  = r_eff
        self.n  = n
        self.amplitude = amplitude

        super().__post_init__()

        self.bbox = Box(self.X.shape)

    def evaluate(self, x, y, amplitude, r_eff, n, x_0, y_0, ellip, theta):
        """Two dimensional Sersic profile function."""

        # bn = gammaincinv(2.0 * n, 0.5)
        bn = 1.99930938*n -0.32678895
        a, b = r_eff, (1 - ellip) * r_eff
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        z = jnp.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

        return amplitude * jnp.exp(-bn * (z ** (1 / n) - 1))
    
    def __call__(self):
        return self.evaluate(self.X, self.Y, amplitude = self.amplitude, r_eff = self.r_eff, n = self.n, 
               x_0=self.x0, y_0=self.y0,
               ellip=self.ellip, theta=self.theta)

class DustyMorphologyNormed(Morphology):

    data: jnp.array
    amplitude: jnp.array

    def __init__(self, 
                 data,
                 amplitude = jnp.array(1),
                   ):
        
        self.data = data
        self.amplitude = amplitude

        super().__post_init__()

        self.bbox = Box(self.data.shape)

    def __call__(self):
        # return self.amplitude * self.data
        return self.data