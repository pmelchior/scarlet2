"""Measurement methods"""

import copy

import numpy as jnp
import numpy.ma as ma

from .frame import get_affine, get_scale_angle_flip_shift
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
        Coordinates of the brightest pixel in pixel coordinates or in the model
        frame (if `component` has a `bbox`)
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
    model = component() if isinstance(component, Component) else component

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
    c = (
        (grid_y * model).sum(axis=(-2, -1)) / f + origin[0],
        (grid_x * model).sum(axis=(-2, -1)) / f + origin[1],
    )
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
    model = component() if isinstance(component, Component) else component
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
        The observations to use for the SNR computation.

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

    m = []
    w = []
    var = []
    # convolve model for every observation;
    # flatten in channel direction because it may not have all C channels; concatenate
    # do same thing for noise variance
    for obs in observations:
        noise_rms = 1 / jnp.sqrt(ma.masked_equal(obs.weights, 0))
        ma.set_fill_value(noise_rms, jnp.inf)
        model_ = obs.render(model)
        m.append(model_.reshape(-1))
        w.append((model_ / (model_.sum(axis=(-2, -1))[:, None, None])).reshape(-1))
        noise_var = noise_rms**2
        var.append(noise_var.reshape(-1))
    m = jnp.concatenate(m)
    w = jnp.concatenate(w)
    var = jnp.concatenate(var)

    # SNR from Erben (2001), eq. 16, extended to multiple bands
    # SNR = (I @ W) / sqrt(W @ Sigma^2 @ W)
    # with W = morph, Sigma^2 = diagonal variance matrix
    snr = (m * w).sum() / jnp.sqrt(((var * w) * w).sum())
    return snr


class Moments(dict):
    """Base Moments class"""

    def __init__(self, component, N=2, center=None, weight=None):  # noqa: N803
        r"""Moments of the brightness distribution

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

        model = component() if isinstance(component, Component) else component

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
                self[m, n - m] = (grid_y**m * grid_x ** (n - m) * model * weight).sum(axis=(-2, -1))

            if n == 1:
                self._centroid = jnp.array((self[1, 0] / self[0, 0], self[0, 1] / self[0, 0]))
                # shift grid to produce centered moments
                if center is None:
                    center = self._centroid
                    self[1, 0] = jnp.zeros_like(self[1, 0])
                    self[0, 1] = jnp.zeros_like(self[0, 1])
                else:
                    center = jnp.asarray(center)
                    # centroid wrt given center
                    self._centroid[0] -= center[0] * self[0, 0]
                    self._centroid[1] -= center[1] * self[0, 0]
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
    def flux(self):
        """Determine flux from 0th moment

        Returns
        -------
        float
        """
        return self[0, 0]

    @property
    def centroid(self):
        """Determine centroid from moments

        Returns
        -------
        array
            Coordinates of the centroid in the pixel frame of the data that defines these moments
        """
        return self._centroid

    @property
    def size(self):
        """Determine size from moments

        Returns
        -------
        float
        """
        flux = self[0, 0]
        return (self[0, 2] / flux * self[2, 0] / flux - (self[1, 1] / flux) ** 2) ** (1 / 4)

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

    def normalize(self):
        """Normalize moments with respect to the flux

        Returns
        -------
        self
        """
        norm = self[0, 0]
        for key in self.keys():
            self[key] /= norm
        return self

    def convolve(self, p):
        """Convolve moments with moments `p`

        The moments are changed in place.

        See Melchior et al. (2010), "Weak gravitational lensing with Deimos", Equation 9

        Parameters
        ----------
        p: Moments
            Moments of the kernel to convolve with

        Returns
        -------
        self
        """

        g_ = self
        g = copy.deepcopy(g_)
        n_min = min(p.order, g.order)

        for n in range(n_min + 1):
            for i in range(n + 1):
                j = n - i
                g_[i, j] = jnp.zeros_like(g_[i, j])
                for k in range(i + 1):
                    for l in range(j + 1):  # noqa: E741
                        g_[i, j] += binomial(i, k) * binomial(j, l) * g[k, l] * p[i - k, j - l]
        return self

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
        self
        """
        g = self
        n_min = min(p.order, g.order)

        # use explicit relations for up to 2nd moments
        g[0, 0] /= p[0, 0]
        if n_min >= 1:
            g[0, 1] -= g[0, 0] * p[0, 1]
            g[1, 0] -= g[0, 0] * p[1, 0]
            g[0, 1] /= p[0, 0]
            g[1, 0] /= p[0, 0]
        if n_min >= 2:
            g[0, 2] -= g[0, 0] * p[0, 2] + 2 * g[0, 1] * p[0, 1]
            g[1, 1] -= g[0, 0] * p[1, 1] + g[0, 1] * p[1, 0] + g[1, 0] * p[0, 1]
            g[2, 0] -= g[0, 0] * p[2, 0] + 2 * g[1, 0] * p[1, 0]
            g[0, 2] /= p[0, 0]
            g[1, 1] /= p[0, 0]
            g[2, 0] /= p[0, 0]
        if n_min >= 3:
            # use general formula
            for n in range(3, n_min + 1):
                for i in range(n + 1):
                    for j in range(n - i):
                        for k in range(i):
                            for l in range(j):  # noqa: E741
                                g[i, j] -= binomial(i, k) * binomial(j, l) * g[k, l] * p[i - k, j - l]
                        for k in range(i):
                            g[i, j] -= binomial(i, k) * g[k, j] * p[i - k, 0]
                        for l in range(j):  # noqa: E741
                            g[i, j] -= binomial(j, l) * g[i, l] * p[0, j - l]
                g[i, j] /= p[0, 0]
        return self

    def resize(self, c):
        """Change moments for a change of factor `c` of the size/spatial resolution
        of the defining frame

        This operation arises when one adjust the moments for a change in the size
        of pixels of the defining frame, e.g. when asking "what would the moments
        be if the pixels were factor c smaller (or the source c times larger)"?
        The fluxes are unchanged, which corresponds to preservation of photons under resizing.

        The moments are changed in place.

        See Teague (1980), "Image  analysis via the general  theory of moments", eq. 34 for details.

        Parameters
        ----------
        c: float or list
            Scaling factor for the size change. Can be different along x and y

        Returns
        -------
        self
        """
        if jnp.isscalar(c):
            assert c > 0
            flux_change = c**2
            for e in self:
                self[e] = self[e] * c ** (2 + e[0] + e[1]) / flux_change
        elif len(c) == 2:
            assert c[0] > 0 and c[1] > 0
            flux_change = c[0] * c[1]
            for e in self:
                self[e] = self[e] * c[0] ** (e[0] + 1) * c[1] ** (e[1] + 1) / flux_change
        else:
            raise AttributeError("c must be a scalar of a list or array of two components")

        return self

    def fliplr(self):
        """Flip moments along the x-axis

        The moments are changed in place.

        Returns
        -------
        self
        """
        for e in self:
            if e[1] % 2 == 1:
                self[e] *= -1
        return self

    def flipud(self):
        """Flip moments along the y-axis

        The moments are changed in place.

        Returns
        -------
        self
        """
        for e in self:
            if e[0] % 2 == 1:
                self[e] *= -1
        return self

    def rotate(self, phi):
        """Change moments for rotation of angle `phi`.

        The moments are changed in place.

        See Teague (1980), "Image  analysis via the general  theory of moments", eq. 36 for details.

        Parameters
        ----------
        phi: float
            Rotation angle, in radian

        Returns
        -------
        self
        """

        mu_p = {}
        for n in range(self.N + 1):
            for j in range(n + 1):
                k = n - j
                value = 0
                for r in range(j + 1):
                    for s in range(k + 1):
                        value += (
                            (-1) ** (k - s)
                            * binomial(j, r)
                            * binomial(k, s)
                            * jnp.cos(phi) ** (j - r + s)
                            * jnp.sin(phi) ** (k + r - s)
                            * self[j + k - r - s, r + s]
                        )
                mu_p[j, k] = value

        for e in self:
            self[e] = mu_p[e]

        return self

    def translate(self, shift):
        """Change moments for translation `shift`

        The moments are changed in place.

        Note: This changes all the moments, not just the dipole, for the new reference center.

        See Teague (1980), "Image  analysis via the general  theory of moments", eq. 30 for details.

        Parameters
        ----------
        shift: (y, x)
            translation, in pixels

        Returns
        -------
        self
        """
        mu = {}
        for n in range(self.N + 1):
            for j in range(n + 1):
                k = n - j
                value = 0
                for r in range(j + 1):
                    for s in range(k + 1):
                        value += (
                            binomial(j, r)
                            * binomial(k, s)
                            * (shift[0]) ** (j - r)
                            * (shift[1]) ** (k - s)
                            * self[r, s]
                        )
                mu[j, k] = value

        for e in self:
            self[e] = mu[e]
        return self

    def transfer(self, wcs_in, wcs_out):
        """Compute rescaling and rotation from WCSs and apply to moments

        The method adjusts moments measured with a frame defined by `wcs_in` to the frame `wcs_out`.
        The moments are changed in place.

        Parameters
        ----------
        wcs_in: :py:class:`astropy.wcs.Wcsprm`
            WCS of the frame with original moments
        wcs_out: :py:class:`astropy.wcs.Wcsprm`
            WCS of the frame to which the moments should be adjusted

        Returns
        -------
        self
        """

        if (wcs_in is not None) and (wcs_out is not None):
            M_in = get_affine(wcs_in)  # noqa: N806
            M_out = get_affine(wcs_out)  # noqa: N806
            M = jnp.linalg.inv(M_out) @ M_in  # noqa: N806, transformation from in pixel -> sky -> out pixels
            scale, angle, flip, _ = get_scale_angle_flip_shift(M)
            # if flipped: go to right-handed coord first before applying rotation
            if flip == -1:
                self.flipud()  # our flip convention is along y-axis
            self.rotate(angle).resize(scale)

        return self


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


def forced_photometry(scene, obs):
    """Computes the spectra of every source in the scene to match the observations

    Computes the best-fit amplitude of the rendered model of all components in every
    channel of every observation as a linear inverse problem.

    If sources/sources components in `scene` have non-flat spectra, the output of this function is
    the correction factor that needs to be applied to those spectra to best match each channel of `obs`.

    Parameters
    ----------
    scene: :py:class:`scarlet2.scene.Scene`
        Scene for which the spectra should be computed
    obs: :py:class:`~scarlet2.Observation`
        The observation used to determine the spectra.

    Returns
    -------
    array
        Array of the spectra, in the order of the sources in the scene
    """

    # extract multi-channel model for every source
    models = []
    for i, src in enumerate(scene.sources):
        # evaluate the model for any source so that fit includes it even if its spectrum is not updated
        model = scene.evaluate_source(src)  # assumes all sources are single components

        # check for models with identical initializations, see scarlet repo issue #282
        # if duplicate: raise ValueError
        for model_indx in range(len(models)):
            if jnp.allclose(model, models[model_indx]):
                message = f"Source {i} has a model identical to source {model_indx}.\n"
                message += "This is likely not intended, and the second source should be deleted."
                raise ValueError(message)
        models.append(model)

    models = jnp.array(models)
    num_models = len(models)

    # independent channels, no mixing
    # solve the linear inverse problem of the amplitudes in every channel
    # given all the rendered morphologies
    # spectrum = (M^T Sigma^-1 M)^-1 M^T Sigma^-1 * im
    num_channels = obs.frame.C
    images = obs.data
    weights = obs.weights
    morphs = jnp.stack([obs.render(model) for model in models], axis=0)
    spectra = jnp.zeros((num_models, num_channels))
    for c in range(num_channels):
        im = images[c].reshape(-1)
        w = weights[c].reshape(-1)
        m = morphs[:, c, :, :].reshape(num_models, -1)
        mw = m * w[None, :]
        # check if all components have nonzero flux in c.
        # because of convolutions, flux can be outside the box,
        # so we need to compare weighted flux with unweighted flux,
        # which is the same (up to a constant) for constant weights.
        # so we check if *most* of the flux is from pixels with non-zero weight
        nonzero = jnp.sum(mw, axis=1) / jnp.sum(m, axis=1) / jnp.mean(w) > 0.1
        nonzero = jnp.flatnonzero(nonzero)
        if len(nonzero) == num_models:
            covar = jnp.linalg.inv(mw @ m.T)
            spectra = spectra.at[:, c].set(covar @ m @ (im * w))
        else:
            covar = jnp.linalg.inv(mw[nonzero] @ m[nonzero].T)
            spectra = spectra.at[nonzero, c].set(covar @ m[nonzero] @ (im * w))

    return spectra


def correlation_function(img, maxlength=2, threshold=0):
    """Computes the 2D correlation function of the image.

    Parameters
    ----------
    img: :py:class:`numpy.ndarray`
        Image array, 2D or 3D. Masked pixels must be set to 0 in `img`.
    maxlength: int
        Maximum length of the correlation function
    threshold: float
        Minimum correlation coefficient to maintain

    Returns
    -------
    dict, with keys (dy,dx) specifying the 2D offset in image pixels
    """
    xi = dict()
    n = dict()
    # expand to image cubes for faster ellipsis
    img_ = img[None, :, :] if img.ndim == 2 else img
    height, width = img_.shape[-2:]
    for dy in range(maxlength + 1):
        for dx in range(maxlength + 1):
            overlap = img_[..., dy:, dx:] * img_[..., : height - dy, : width - dx]
            xi[dy, dx] = jnp.sum(overlap, axis=(-2, -1))
            n[dy, dx] = jnp.sum(overlap != 0, axis=(-2, -1))

    # normalize and filter correlations below threshold
    # Note: possibly safer to set the largest negative correlation (which should not exist) as threshold
    for k in xi:
        xi[k] = jnp.maximum(xi[k] / jnp.maximum(n[k], 1), threshold)  # prevent division by 0

    # fill in the symmetric negative offsets
    offsets = list(xi.keys())
    for k in offsets:
        dy, dx = k
        if dy > 0:
            dy *= -1
        if dx > 0:
            dx *= -1
        xi[dy, dx] = xi[k]

    return xi
