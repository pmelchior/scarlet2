"""Wavelet functions"""
# from https://github.com/pmelchior/scarlet/blob/master/scarlet/wavelet.py

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
