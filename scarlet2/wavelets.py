# from https://github.com/pmelchior/scarlet/blob/master/scarlet/wavelet.py

import jax.numpy as jnp
import numpy as np

class Starlet:
    """ A class used to create the Wavelet transform of a cube of images from the 'a trou' algorithm.

        The transform is performed by convolving the image by a seed starlet: the transform of an all-zero
        image with its central pixel set to one. This requires 2-fold padding of the image and an odd pad
        shape. The fft of the seed starlet is cached so that it can be reused in the transform of other
        images that have the same shape.
    """

    def __init__(self, image, coefficients, generation, convolve2D):
        """ Initialise the Starlet object

        Parameters
        ----------
        image: numpy ndarray
            Image in real space.
        coefficients: array
            Starlet transform of the image.
        generation: int
            The generation of the starlet transform (either `1` or `2`).
        convolve2D: array-like
            The filter used to convolve the image and create the wavelets.
            When `convolve2D` is `None` this uses a cubic bspline.
        """
        self._image = image
        self._coeffs = coefficients
        self._generation = generation
        self._convolve2D = convolve2D
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
    def from_image(image, scales=None, generation=2, convolve2D=None):
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
        convolve2D: array-like
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
        coefficients = starlet_transform(image, scales, generation, convolve2D)
        return Starlet(image, coefficients, generation, convolve2D)
    
def bspline_convolve(image, scale):
    """Convolve an image with a bpsline at a given scale.

    This uses the spline
    `h1D = np.array([1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16])`
    from Starck et al. 2011.

    Parameters
    ----------
    image: 2D array
        The image or wavelet coefficients to convolve.
    scale: int
        The wavelet scale for the convolution. This sets the
        spacing between adjacent pixels with the spline.

    """
    # Filter for the scarlet transform. Here bspline
    h1D = np.array([1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16])
    j = scale

    slice0 = slice(None, -2**(j+1))
    slice1 = slice(None, -2**j)
    slice3 = slice(2**j, None)
    slice4 = slice(2**(j+1), None)
    # row
    col = image * h1D[2]
    col[slice4] += image[slice0] * h1D[0]
    col[slice3] += image[slice1] * h1D[1]
    col[slice1] += image[slice3] * h1D[3]
    col[slice0] += image[slice4] * h1D[4]

    # column
    result = col * h1D[2]
    result[:, slice4] += col[:, slice0] * h1D[0]
    result[:, slice3] += col[:, slice1] * h1D[1]
    result[:, slice1] += col[:, slice3] * h1D[3]
    result[:, slice0] += col[:, slice4] * h1D[4]
    return result

def starlet_transform(image, scales=None, generation=2, convolve2D=None):
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
        This must be `1` or `2`.
    convolve2D: function
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
    if convolve2D is None:
        convolve2D = bspline_convolve

    ## wavelet set of coefficients.
    starlet = np.zeros((scales + 1,) + image.shape)
    for j in range(scales):
        gen1 = convolve2D(c, j)

        if generation == 2:
            gen2 = convolve2D(gen1, j)
            starlet[j] = c - gen2
        else:
            starlet[j] = c - gen1

        c = gen1

    starlet[-1] = c
    return starlet

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