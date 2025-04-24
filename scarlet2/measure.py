"""Measurement methods"""

import math

import astropy.units as u
import numpy as jnp
import numpy.ma as ma

from .frame import get_scale, get_angle, get_sign
from .source import Component


def max_pixel(component):
    """Determine pixel with maximum value

    Parameters
    ----------
    component: :py:class:`~scarlet2.Component` or array
        Component to analyze or its hyperspectral model

    Returns
    -------
    array
        Coordinates of the brightest pixel in pixel coordinates or in the model frame (if `component` has a `bbox`)
    """
    if isinstance(component, Component):
        model = component()
        origin = jnp.array(component.bbox.origin)
    else:
        model = component
        origin = 0

    return jnp.array(jnp.unravel_index(jnp.argmax(model), model.shape)) + origin


def flux(component):
    """Determine total flux in every channel

    Parameters
    ----------
    component: :py:class:`~scarlet2.Component` or array
        Component to analyze or its hyperspectral model

    Returns
    -------
    float
    """
    if isinstance(component, Component):
        model = component()
    else:
        model = component

    return model.sum(axis=(-2, -1))


def centroid(component):
    """Determine spatial centroid of model

    Parameters
    ----------
    component: :py:class:`~scarlet2.Component` or array
        Component to analyze or its hyperspectral model

    Returns
    -------
    array
        Coordinates of the centroid in pixel coordinates or in the model frame (if `component` has a `bbox`)
    """
    if isinstance(component, Component):
        model = component()
        origin = jnp.array(component.bbox.spatial.origin)
    else:
        model = component
        origin = 0, 0

    grid_y, grid_x = jnp.indices(model.shape[-2:])
    if len(model.shape) == 3:
        grid_y = grid_y[None, :, :]
        grid_x = grid_x[None, :, :]
    f = flux(model)
    c = (grid_y * model).sum(axis=(-2, -1)) / f + origin[0], (grid_x * model).sum(axis=(-2, -1)) / f + origin[1]
    return jnp.array(c)


def fwhm(component):
    """Determine the Full-width at half maximum in pixels

    Parameters
    ----------
    component: py:class:`~scarlet2.Component` or array
        Component to analyze or its hyperspectral model

    Returns
    -------
    float
    """
    if isinstance(component, Component):
        model = component()
    else:
        model = component
    peak_pixel = max_pixel(model)[-2:]  # only spatial location
    peak_value = model[:, peak_pixel[0], peak_pixel[1]]
    half_value = peak_value / 2
    num_pixels = jnp.count_nonzero(model >= half_value[:, None, None], axis=(1, 2))
    diameter = 2 * jnp.sqrt(num_pixels) / jnp.pi
    return diameter

def snr(component, observations):
    """Determine SNR with `component` as weight function

    Parameters
    ----------
    component: :py:class:`~scarlet2.Component` or array
        Component to analyze or its hyperspectral model
    observations: :py:class:`scarlet2.Observation` or list

    Returns
    -------
    float
    """
    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    if hasattr(component, "get_model"):
        frame = None
        model = component.get_model(frame=frame)
    else:
        model = component

    M = []
    W = []
    var = []
    # convolve model for every observation;
    # flatten in channel direction because it may not have all C channels; concatenate
    # do same thing for noise variance
    for obs in observations:
        noise_rms = 1 / jnp.sqrt(ma.masked_equal(obs.weights, 0))
        ma.set_fill_value(noise_rms, jnp.inf)
        model_ = obs.render(model)
        M.append(model_.reshape(-1))
        W.append((model_ / (model_.sum(axis=(-2, -1))[:, None, None])).reshape(-1))
        noise_var = noise_rms ** 2
        var.append(noise_var.reshape(-1))
    M = jnp.concatenate(M)
    W = jnp.concatenate(W)
    var = jnp.concatenate(var)

    # SNR from Erben (2001), eq. 16, extended to multiple bands
    # SNR = (I @ W) / sqrt(W @ Sigma^2 @ W)
    # with W = morph, Sigma^2 = diagonal variance matrix
    snr = (M * W).sum() / jnp.sqrt(((var * W) * W).sum())
    return snr


class Moments(dict):
    def __init__(self, component, N=2, center=None, weight=None):
        """Moments of the brightness distribution

        The dict is accessed by keys, which denote the power of y/x of the specific Moment:
        `m[p,q] = \int dx dy f(y,x) y^p x^q`.

        Notes
        -----
        Like all coordinates in scarlet2, moments are computed in (y,x) order.

        Parameters
        ----------
        component: :py:class:`~scarlet2.Component` or array
            Component to analyze or its hyperspectral model
        N: int >=0
            Moment order
        center: array
            2D coordinate in frame of `component`
        weight: array
            weight function with same shape as `component`
        """
        super().__init__()

        if isinstance(component, Component):
            model = component()
        else:
            model = component

        if weight is None:
            weight = 1

        grid_y, grid_x = jnp.indices(model.shape[-2:])

        if model.ndim == 3:
            grid_y = grid_y[None, :, :]
            grid_x = grid_x[None, :, :]

        self.N = N
        # moment code adapted from https://github.com/pmelchior/shapelens/blob/master/src/Moments.cc
        for n in range(self.N + 1):
            for m in range(n + 1):
                # moments ordered by power in y, then x
                self[m, n - m] = (grid_y ** m * grid_x ** (n - m) * model * weight).sum(axis=(-2, -1))

            if n == 1:
                # shift grid to produce centered moments
                if center is None:
                    center = self[1, 0] / self[0, 0], self[0, 1] / self[0, 0]  # centroid
                    self[1, 0] = jnp.zeros_like(self[1, 0])
                    self[0, 1] = jnp.zeros_like(self[0, 1])
                else:
                    center = jnp.asarray(center)
                    # centroid wrt given center
                    self[1, 0] -= center[0] * self[0, 0]
                    self[0, 1] -= center[1] * self[0, 0]
                if model.ndim == 3 and center[0].ndim == 1:
                    center = center[0][:, None, None], center[1][:, None, None]
                grid_y = grid_y - center[0]
                grid_x = grid_x - center[1]

    @property
    def order(self):
        """Moment order

        Returns
        -------
        int
        """
        # TODO: why is this not simply self.N? Probably left over from a dynamic computation of higher moments
        return max(key[0] for key in self.keys())

    @property
    def centroid(self):
        """Determine centroid from moments

        Returns
        -------
        array
            Coordinates of the centroid in the pixel frame of the data that defines these moments
        """
        return jnp.array((self[1, 0] / self[0, 0], self[0, 1] / self[0, 0]))

    @property
    def size(self):
        """Determine size from moments

        Returns
        -------
        float
        """
        flux = self[0, 0]
        T = (self[0, 2] / flux * self[2, 0] / flux - (self[1, 1] / flux) ** 2) ** (1 / 4)
        return T

    @property
    def ellipticity(self):
        """Determine complex ellipticity from moments

        Returns
        -------
        jnp.array
            Ellipticity (2D) of the data that defines these moments
        """
        ellipticity = (self[0, 2] - self[2, 0] + 2j * self[1, 1]) / (self[2, 0] + self[0, 2])
        return jnp.array((ellipticity.real, ellipticity.imag))

    def deconvolve(self, p):
        """Deconvolve moments from moments `p`

        The moments are changed in place.

        See Melchior et al. (2010), "Weak gravitational lensing with Deimos", Table 1

        Parameters
        ----------
        p: Moments
            Moments of the kernel to deconvolve from

        Returns
        -------
        None
        """
        g = self
        Nmin = min(p.order, g.order)

        # use explicit relations for up to 2nd moments
        g[0, 0] /= p[0, 0]
        if Nmin >= 1:
            g[0, 1] -= g[0, 0] * p[0, 1]
            g[1, 0] -= g[0, 0] * p[1, 0]
            g[0, 1] /= p[0, 0]
            g[1, 0] /= p[0, 0]
            if Nmin >= 2:
                g[0, 2] -= g[0, 0] * p[0, 2] + 2 * g[0, 1] * p[0, 1]
                g[1, 1] -= g[0, 0] * p[1, 1] + g[0, 1] * p[1, 0] + g[1, 0] * p[0, 1]
                g[2, 0] -= g[0, 0] * p[2, 0] + 2 * g[1, 0] * p[1, 0]
                if Nmin >= 3:
                    # use general formula
                    for n in range(3, Nmin + 1):
                        for i in range(n + 1):
                            for j in range(n - i):
                                for k in range(i):
                                    for l in range(j):
                                        g[i, j] -= (
                                                binomial(i, k)
                                                * binomial(j, l)
                                                * g[k, l]
                                                * p[i - k, j - l]
                                        )
                                for k in range(i):
                                    g[i, j] -= binomial(i, k) * g[k, j] * p[i - k, 0]
                                for l in range(j):
                                    g[i, j] -= binomial(j, l) * g[i, l] * p[0, j - l]
                        g[i, j] /= p[0, 0]

    def resize(self, c):
        """Change moments for a change of factor `c` of the size/spatial resolution of the defining frame

        This operation arises when one adjust the moments for a change in the size of pixels of the defining frame, e.g.
        when asking "what would the moments be if the pixels were factor c smaller (or the source c times larger)"?
        The moments are changed in place.

        See Teague (1980), "Image  analysis via the general  theory of moments", eq. 34 for details.

        Parameters
        ----------
        c: float or list
            Scaling factor for the size change. Can be different along x and y

        Returns
        -------
        None
        """
        # Teague (1980), eq. 34
        if jnp.isscalar(c):
            for e in self:
                self[e] = self[e] * c ** (2 + e[0] + e[1])
        elif len(c) == 2:
            for e in self:
                self[e] = self[e] * c[0] ** (e[0] + 1) * c[1] ** (e[1] + 1)
        else:
            raise AttributeError("c must be a scalar of a list or array of two components")

    def rotate(self, phi):
        """Change moments for rotation of angle `phi`.

        The moments are changed in place.

        See Teague (1980), "Image  analysis via the general  theory of moments", eq. 36 for details.

        Parameters
        ----------
        phi: :py:class:`astropy.Quantity`
            Rotation angle. Must be an astropy quantity of type="angle".

        Returns
        -------
        None
        """
        assert u.get_physical_type(phi) == "angle" # check that it's an angle with a suitable unit
        phi = phi.to(u.deg).value
        phi = phi * math.pi / 180 # radian

        mu_p = {}
        for n in range(self.N+1):
            for m in range(n+1):
                j = m
                k = n-m
                value = 0
                for r in range(j+1):
                    for s in range(k+1):
                        value += (-1)**(k-s) * binomial(j, r)*binomial(k, s) *\
                              jnp.cos(phi)**(j-r+s) * jnp.sin(phi)**(k+r-s) *\
                                  self[j+k-r-s, r+s]
                mu_p[m, n-m] = value
        
        for e in self:
            self[e] = mu_p[e]

    def transfer(self, wcs_in, wcs_out):
        """Compute rescaling and rotation from WCSs and apply to moments

        This methods adjusts moments measured with a frame defined by `wcs_in` to to the frame `wcs_out`.
        The moments are changed in place.

        Parameters
        ----------
        wcs_in: :py:class:`astropy.wcs.Wcsprm`
        wcs_out: :py:class:`astropy.wcs.Wcsprm`

        Returns
        -------
        None
        """

        flux_in = self[0,0]

        if (wcs_in is not None) and (wcs_out is not None):
            # Rescale moments (amplitude rescaling)
            scale_in = get_scale(wcs_in) * 60 ** 2  # arcsec
            scale_out = get_scale(wcs_out) * 60 ** 2  # arcsec
            c = jnp.array(scale_in) / jnp.array(scale_out)
            self.resize(c)

            # Rotate moments
            phi_in = get_angle(wcs_in)
            phi_out = get_angle(wcs_out)
            phi = (phi_out - phi_in) / jnp.pi * 180
            self.rotate(phi * u.deg)

            # Flip moments if WCSs don't share the same convention
            sign_in = get_sign(wcs_in)
            sign_out = get_sign(wcs_out)
            self.resize(sign_in * sign_out)

            flux_out = self[0,0]
            
            for key in self.keys():
                self[key]  /= flux_out/flux_in
                
# def moments(component, N=2, center=None, weight=None):
#    return Moments(component, N=N, center=center, weight=weight)

# adapted from  https://github.com/pmelchior/shapelens/blob/src/DEIMOS.cc
def binomial(n, k):
    """Binomial coefficient"""
    if k == 0:
        return 1
    if k > n // 2:
        return binomial(n, n - k)
    result = 1
    for i in range(1, k + 1):
        result *= n - i + 1
        result //= i
    return result



