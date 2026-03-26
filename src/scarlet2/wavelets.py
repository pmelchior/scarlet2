"""Wavelet functions"""
# from https://github.com/pmelchior/scarlet/blob/master/scarlet/wavelet.py

import jax
import jax.numpy as jnp


class Starlet:
    """Wavelet transform of a images (2D or 3D) with the 'a trou' algorithm.

    The transform is performed by convolving the image by a seed starlet: the transform of an all-zero
    image with its central pixel set to one. This requires 2-fold padding of the image and an odd pad
    shape. The fft of the seed starlet is cached so that it can be reused in the transform of other
    images that have the same shape.
    """

    def __init__(self, image, coefficients, generation, convolve2d):
        """
        Parameters
        ----------
        image: array
            Image in real space.
        coefficients: array
            Starlet transform of the image.
        generation: int
            The generation of the starlet transform (either `1` or `2`).
        convolve2d: array
            The filter used to convolve the image and create the wavelets.
            When `convolve2d` is `None` this uses a cubic bspline.
        """
        self._image = image
        self._coeffs = coefficients
        self._generation = generation
        self._convolve2d = convolve2d
        self._norm = None

    @property
    def image(self):
        """The real space image"""
        return self._image

    @property
    def coefficients(self):
        """Starlet coefficients"""
        return self._coeffs

    @staticmethod
    def from_image(image, scales=None, generation=2, convolve2d=None):
        """Generate a set of starlet coefficients for an image

        Parameters
        ----------
        image: array-like
            The image that is converted into starlet coefficients
        scales: int
            The number of starlet scales to use.
            If `scales` is `None` then the maximum number of scales is used.
            Note: this is the length of the coefficients-1, as in the notation
            of `Starck et al. 2011`.
        generation: int
            The generation of the starlet transform (either `1` or `2`).
        convolve2d: array-like
            The filter used to convolve the image and create the wavelets.
            When `convolve2D` is `None` this uses a cubic bspline.

        Returns
        -------
        result: Starlet
            The resulting `Starlet` that contains the image, starlet coefficients,
            as well as the parameters used to generate the coefficients.
        """
        if scales is None:
            scales = get_scales(image.shape)
        coefficients = starlet_transform(image, scales, generation, convolve2d)
        return Starlet(image, coefficients, generation, convolve2d)


def bspline_convolve(image, scale):
    """Convolve an image with a bpsline at a given scale.

    This uses the spline
    `h1d = jnp.array([1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16])`
    from Starck et al. 2011.

    Parameters
    ----------
    image: 2D array
        The image or wavelet coefficients to convolve.
    scale: int
        The wavelet scale for the convolution. This sets the
        spacing between adjacent pixels with the spline.

    """
    h1d = jnp.array([1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16])
    step = 2**scale
    ny, nx = image.shape

    row_idx = jnp.arange(ny)
    col_idx = jnp.arange(nx)

    def reflect(idx, size):
        # reflect indices at boundaries into [0, size-1]
        idx = jnp.abs(idx)
        # Map into [0, 2*size - 2] period, then fold back
        idx = idx % (2 * size - 2)
        return jnp.where(idx >= size, 2 * size - 2 - idx, idx)

        # a simpler version: clamp pixels beyond the edge to the edge pixels
        #         return jnp.clip(idx, 0, size - 1)  # clamp — or use true reflection below

    # Row convolution
    col = jnp.zeros_like(image)
    for k, offset in enumerate([-2 * step, -step, 0, step, 2 * step]):
        reflected = reflect(row_idx + offset, ny)
        shifted = jnp.take(image, reflected, axis=0)
        col += shifted * h1d[k]

    # Column convolution
    result = jnp.zeros_like(col)
    for k, offset in enumerate([-2 * step, -step, 0, step, 2 * step]):
        reflected = reflect(col_idx + offset, nx)
        shifted = jnp.take(col, reflected, axis=1)
        result += shifted * h1d[k]

    return result


def starlet_transform(image, scales=None, generation=2, convolve2d=None):
    """Perform a scarlet transform, or 2nd gen starlet transform.

    Parameters
    ----------
    image: 2D array
        The image to transform into starlet coefficients.
    scales: int
        The number of scale to transform with starlets.
        The total dimension of the starlet will have
        `scales+1` dimensions, since it will also hold
        the image at all scales higher than `scales`.
    generation: int
        The generation of the transform.
        This must be `1` or `2`. Default is `2`.
    convolve2d: function
        The filter function to use to convolve the image
        with starlets in 2D.

    Returns
    -------
    starlet: array with dimension (scales+1, Ny, Nx)
        The starlet dictionary for the input `image`.
    """
    assert len(image.shape) == 2, f"Image should be 2D, got {len(image.shape)}"
    assert generation in (1, 2), f"generation should be 1 or 2, got {generation}"

    scales = get_scales(image.shape, scales)
    c = image
    if convolve2d is None:
        convolve2d = bspline_convolve

    ## wavelet set of coefficients.
    starlet = jnp.zeros((scales + 1,) + image.shape)
    for j in range(scales):
        gen1 = convolve2d(c, j)

        if generation == 2:
            gen2 = convolve2d(gen1, j)
            starlet = starlet.at[j].set(c - gen2)
        else:
            starlet = starlet.at[j].set(c - gen1)

        c = gen1

    starlet = starlet.at[-1].set(c)
    return starlet


def starlet_reconstruction(starlets, generation=2, convolve2d=None, scales=None):
    """Reconstruct an image from a dictionary of starlets

    Parameters
    ----------
    starlets: array with dimension (scales+1, Ny, Nx)
        The starlet dictionary used to reconstruct the image.
    generation: int
        The generation of the transform.
        This must be `1` or `2`. Default is `2`.
    convolve2d: function
        The filter function to use to convolve the image
        with starlets in 2D.
    scales: list of int
        The scales to include in the reconstruction (0 being the smallest)

    Returns
    -------
    image: 2D array
        The image reconstructed from the input `starlet`.
    """
    if generation == 1:
        return jnp.sum(starlets, axis=0)
    if convolve2d is None:
        convolve2d = bspline_convolve

    # scales sorted in reverse order: from largest to smallest
    max_scale = len(starlets) - 1
    if scales is None:
        scales = tuple(max_scale - i for i in range(1, max_scale + 1))
    else:
        scales = sorted(tuple(scale for scale in scales if scale <= max_scale), reverse=True)

    # reconstruct: initialize from largest, go to smallest
    c = starlets[scales[0]]
    for j in scales[1:]:
        cj = convolve2d(c, j)
        c = cj + starlets[j]
    return c


def get_multiresolution_support(image, starlets, sigma, K=3, epsilon=1e-1, max_iter=20, image_type="ground", rng_key=None):
    """Calculate the multi-resolution support for a dictionary of starlet coefficients.

    This is different for ground and space based telescopes.
    For space-based telescopes the procedure in Starck and Murtagh 1998
    iteratively calculates the multi-resolution support.
    For ground based images, where the PSF is much wider and there are no
    pixels with no signal at all scales, we use a modified method that
    estimates support at each scale independently.

    Parameters
    ----------
    image: 2D array
        The image to transform into starlet coefficients.
    starlets: array with dimension (scales+1, Ny, Nx)
        The starlet dictionary used to reconstruct `image`.
    sigma: float
        The standard deviation of the `image`.
    K: float
        The multiple of `sigma` to use to calculate significance.
        Coefficients `w` where `|w| > K*sigma_j`, where `sigma_j` is
        the standard deviation at the jth scale, are considered significant.
    epsilon: float
        The convergence criteria of the algorithm.
        Once ``|new_sigma_j - sigma_j| / new_sigma_j < epsilon`` the
        algorithm has completed.
    max_iter: int
        Maximum number of iterations to fit `sigma_j` at each scale.
    image_type: str
        The type of image that is being used.
        This should be ``"ground"`` for ground based images with wide PSFs or
        ``"space"`` for images from space-based telescopes with a narrow PSF.
    rng_key: jax.random.PRNGKey, optional
        Random key used only when ``image_type="space"`` to generate a noise
        realisation for estimating per-scale standard deviations.
        Defaults to ``jax.random.PRNGKey(0)`` if not provided.

    Returns
    -------
    M : array of int
        Mask with significant coefficients in `starlets` set to ``1``.
    sigma_j : array, shape (scales+1,)
        Converged per-scale noise estimate used for thresholding.  For the
        ``"ground"`` branch this is the iteratively refined ``sigma_j``; for
        ``"space"`` it is ``sigma * sigma_je`` (the product of the input sigma
        and the per-scale noise factor derived from a noise realisation).
    """
    assert image_type in ("ground", "space")

    ny, nx = starlets.shape[-2:]
    n_scales = len(starlets)

    # Per-scale interior masks: exclude the 4*2^j-pixel border at scale j.
    # The 2nd-generation bspline convolution is applied twice per scale, so
    # boundary influence reaches 4*2^j pixels inward.  Excluding these pixels
    # from sigma estimation prevents inflated edge coefficients from biasing
    # the per-scale noise threshold, while still allowing the threshold to be
    # applied (and sources detected) all the way to the image edge.
    def _interior(j):
        b = min(4 * (2 ** j), ny // 4, nx // 4)
        row_ok = (jnp.arange(ny) >= b) & (jnp.arange(ny) < ny - b)
        col_ok = (jnp.arange(nx) >= b) & (jnp.arange(nx) < nx - b)
        return row_ok[:, None] & col_ok[None, :]

    interior = jnp.stack([_interior(j) for j in range(n_scales)])

    if image_type == "space":
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        # Calculate sigma_je, the standard deviation at each scale due to gaussian noise
        noise_img = jax.random.normal(rng_key, shape=image.shape)
        noise_starlet = starlet_transform(noise_img, scales=get_scales(image.shape), generation=1)
        sigma_je = jnp.array([jnp.std(star) for star in noise_starlet])
        noise = image - starlets[-1]

        last_sigma_i = sigma
        M = None
        for _ in range(max_iter):
            M = (jnp.abs(starlets) > K * sigma * sigma_je[:, None, None])
            S = jnp.sum(M, axis=0) == 0
            mask_2d = S & interior[-1]
            n = mask_2d.sum().clip(1)
            mean = (noise * mask_2d).sum() / n
            sigma_i = jnp.sqrt(((noise - mean) ** 2 * mask_2d).sum() / n)
            if jnp.abs(sigma_i - last_sigma_i) / sigma_i < epsilon:
                break
            last_sigma_i = sigma_i
        # effective 1-sigma threshold level at each scale
        sigma_j = sigma * sigma_je
    else:
        # Sigma to use for significance at each scale.
        # Initially we use the input `sigma`.
        sigma_j = jnp.ones((len(starlets),), dtype=image.dtype) * sigma
        last_sigma_j = sigma_j
        M = None
        for _ in range(max_iter):
            M = (jnp.abs(starlets) > K * sigma_j[:, None, None])
            # Compute std only over insignificant interior pixels to avoid both
            # boundary-convolution artifacts and bias from zeroing excluded pixels.
            mask = (~M) & interior
            n = mask.sum(axis=(1, 2)).clip(1)
            mean = (starlets * mask).sum(axis=(1, 2)) / n
            sigma_j = jnp.sqrt(((starlets - mean[:, None, None]) ** 2 * mask).sum(axis=(1, 2)) / n)
            # At lower scales all of the pixels may be significant,
            # so sigma is effectively zero. To avoid infinities we
            # only check the scales with non-zero sigma
            cut = sigma_j > 0
            if jnp.all(jnp.abs(sigma_j[cut] - last_sigma_j[cut]) / sigma_j[cut] < epsilon):
                break
            last_sigma_j = sigma_j
    return M.astype(int), sigma_j


def get_scales(image_shape, scales=None):
    """Get the number of scales to use in the starlet transform.

    Parameters
    ----------
    image_shape: tuple
        The 2D shape of the image that is being transformed
    scales: int
        The number of scale to transform with starlets.
        The total dimension of the starlet will have
        `scales+1` dimensions, since it will also hold
        the image at all scales higher than `scales`.
    """
    # Number of levels for the Starlet decomposition
    max_scale = int(jnp.log2(min(image_shape[-2:]))) - 1
    if (scales is None) or scales > max_scale:
        scales = max_scale
    return int(scales)
