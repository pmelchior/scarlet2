# import autograd.numpy as np
import jax
import numpy as np
import jax.numpy as jnp
from autograd.extend import defvjp, primitive

class Starlet(object):
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

    @staticmethod
    def from_coefficients(coefficients, generation=2, convolve2D=None):
        """Generate an image from a set of starlet coefficients

        Parameters
        ----------
        coefficients: array-like
            The starlet coefficients used to generate the image
        generation: int
            The generation of the starlet transform (either `1` or `2`).
        convolve2D: array-like
            The filter used to convolve the image and create the wavelets.
            When `convolve2D` is `None` this uses a cubic bspline.

        Returns
        -------
        result: Starlet
            The resulting `Starlet` that contains the image, starlet coefficients,
            as well as the parameters used to generate the image.
        """
        image = starlet_reconstruction(coefficients, generation, convolve2D)
        return Starlet(image, coefficients, generation, convolve2D)

    @property
    def image(self):
        """The real space image"""
        return self._image

    @image.setter
    def image(self, image):
        """Update the coefficients if the image has changed"""
        self._image = image
        self._coeffs = starlet_transform(self.image, self.generation, self.convolve2D)

    @property
    def coefficients(self):
        """Starlet coefficients"""
        return self._coeffs

    @coefficients.setter
    def coefficients(self, coeffs):
        """Update the image if the coefficients have changed"""
        self._coeffs = coeffs
        self._image = starlet_reconstruction(self.coefficients, self.generation, self.convolve2D)

    @property
    def scales(self):
        """Number of starlet scales"""
        return len(self.coefficients)-1

    @property
    def generation(self):
        """The generation of the starlet transform"""
        return self._generation

    @generation.setter
    def generation(self, value):
        """Update the generation of a starlet transform, which involve recalculating the coefficients"""
        if value != self.generation:
            self._generation = value
            self._coeffs = starlet_transform(self.image, self.generation, self.convolve2D)
            self._norm = None

    @property
    def convolve2D(self):
        """Filter used to create starlet coefficients"""
        return self._convolve2D

    @convolve2D.setter
    def convolve2D(self, value):
        """Update the filter used to calculate starlet coefficients, which also involves recreating the coefficients"""
        if value != self.convolve2D:
            self._convolve2D = value
            self._coeffs = starlet_transform(self.image, self.generation, self.convolve2D)
            self._norm = None

    @property
    def norm(self):
        """The norm of a convolved dirac"""
        if self._norm is None:
            cy, cx = jnp.array(self.image.shape[-2:])//2
            dirac = jnp.zeros(self.image.shape[-2:])
            dirac[cy, cx] = 1
            seed = starlet_transform(dirac, generation=self.generation, convolve2D=self.convolve2D)
            self._norm = jnp.sqrt(jnp.sum(seed**2, axis=(-2, -1)))
        return self._norm

def set_up_col(image, scale):

    axis   = 1
    j = scale

    yindex = jnp.tile( jnp.arange(image.shape[1]), (image.shape[0], 1))

    zeroes = jnp.zeros(image.shape)

    col0 = jnp.roll(image, 2**(j+1), axis = axis)
    col0 = jnp.where( (yindex < 2**(j+1)).astype(int) , zeroes, col0 )

    col1 = jnp.roll(image, 2**j, axis = axis)
    col1 = jnp.where( (yindex < 2**j).astype(int) , zeroes, col1 )

    col3 = jnp.roll(image, -2**j, axis = axis)
    col3 = jnp.where( (yindex >= image.shape[1] - 2**j).astype(int) , zeroes, col3 )

    col4 = jnp.roll(image, -2**(j+1) , axis = axis)
    col4 = jnp.where( (yindex >= image.shape[1] - 2**(j+1) ).astype(int) , zeroes, col4 )

    return [col0, col1, col3, col4]

def set_up_row(image, scale):

    axis   = 0
    yindex = jnp.array([jnp.arange( image.shape[0] )]).T

    j = scale

    zeroes = jnp.zeros(image.shape)

    col0 = jnp.roll(image, 2**(j+1), axis = axis)
    col0 = jnp.where( (yindex < 2**(j+1)).astype(int) , zeroes, col0 )

    col1 = jnp.roll(image, 2**j, axis = axis)
    col1 = jnp.where( (yindex < 2**j).astype(int) , zeroes, col1 )

    col3 = jnp.roll(image, -2**j, axis = axis)
    col3 = jnp.where( (yindex >= image.shape[0] - 2**j).astype(int) , zeroes, col3 )

    col4 = jnp.roll(image, -2**(j+1) , axis = axis)
    col4 = jnp.where( (yindex >= image.shape[0] - 2**(j+1) ).astype(int) , zeroes, col4 )

    return [col0, col1, col3, col4]

def bspline_convolve_support(image, cols):

    # Filter for the scarlet transform. Here bspline
    h1D = jnp.array([1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16])

    col = image * h1D[2]

    col = col + cols[0] * h1D[0]
    col = col + cols[1] * h1D[1]
    col = col + cols[2] * h1D[3]
    col = col + cols[3] * h1D[4]

    return col

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
    cols = set_up_row(image, scale)
    col  = bspline_convolve_support(image, cols)

    cols = set_up_col(col, scale)
    col  = bspline_convolve_support(col, cols)

    return col

@jax.jit
def starlet_transform_support(c, j):
    gen1 = bspline_convolve(c, j)
    gen2 = bspline_convolve(gen1, j)
    
    return gen1, gen2

def starlet_transform(image, scales=None, generation=2):
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

    Returns
    -------
    starlet: array with dimension (scales+1, Ny, Nx)
        The starlet dictionary for the input `image`.
    """
    assert len(image.shape) == 2, f"Image should be 2D, got {len(image.shape)}"
    assert generation in (1, 2), f"generation should be 1 or 2, got {generation}"

    scales = get_scales(image.shape, scales)

    ## wavelet set of coefficients.
    starlet = []
    for j in range(scales):

        gen1, gen2 = starlet_transform_support(image, j)

        if generation == 2:
            starlet.append(image-gen2)
        else:
            starlet.append(image-gen1)

        image = gen1

    starlet.append(image)
    return jnp.array(starlet)

def _grad_bspline_convolve(input_grad, image, scale):
    return lambda input_grad: bspline_convolve(input_grad, scale)

defvjp(bspline_convolve, _grad_bspline_convolve)

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
    max_scale = int(np.log2( np.min(image_shape[-2:]))) - 1
    if (scales is None) or scales > max_scale:
        scales = max_scale
    return int(scales)

def multiband_starlet_transform(image, scales=None, generation=2, convolve2D=None):
    """Perform a starlet transform of a multiband image.

    See `starlet_transform` for a description of the parameters.
    """
    assert len(image.shape) == 3, f"Image should be 3D (bands, height, width), got shape {len(image.shape)}"
    assert generation in (1, 2), f"generation should be 1 or 2, got {generation}"
    scales = get_scales(image.shape, scales)

    wavelets = jnp.empty((scales+1,)+image.shape, dtype=image.dtype)
    for b, image in enumerate(image):
        wavelets[:, b] = starlet_transform(image, scales=scales, generation=generation, convolve2D=convolve2D)
    return wavelets

@jax.jit
def starlet_reconstruction_gen2(starlets):

    convolve2D = bspline_convolve
    scales = len(starlets) - 1

    c = starlets[-1]
    for i in range(1, scales + 1):
        j = scales - i
        cj = convolve2D(c, j)
        c = cj + starlets[j]
    return c

def starlet_reconstruction(starlets, generation=2):
    """Reconstruct an image from a dictionary of starlets

    Parameters
    ----------
    starlets: array with dimension (scales+1, Ny, Nx)
        The starlet dictionary used to reconstruct the image.
    
    generation: int
        The generation of the transform.
        This must be `1` or `2`.

    Returns
    -------
    image: 2D array
        The image reconstructed from the input `starlet`.
    """
    if generation == 1:
        return jnp.sum(starlets, axis=0)
    else:
        return starlet_reconstruction_gen2(starlets)

def multiband_starlet_reconstruction(starlets, generation=2, convolve2D=None):
    """Reconstruct a multiband image

    See `starlet_reconstruction` for a description of the
    remainder of the parameters.
    """
    scales, bands, width, height = starlets.shape
    result = jnp.array((bands, width, height), dtype=starlets.dtype)
    for band in bands:
        result[:, band] = starlet_reconstruction(
            starlets[:, band],
            generation=generation,
            convolve2D=convolve2D
        )
    return result


def get_multiresolution_support(image, starlets, sigma, K=3, epsilon=1e-1, max_iter=20, image_type="ground"):
    """Calculate the multi-resolution support for a dictionary of starlet coefficients

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
        standard deviation at the jth scale, are considered significant.
    epsilon: float
        The convergence criteria of the algorithm.
        Once `|new_sigma_j - sigma_j|/new_sigma_j < epsilon` the
        algorithm has completed.
    max_iter: int
        Maximum number of iterations to fit `sigma_j` at each scale.
    image_type: str
        The type of image that is being used.
        This should be "ground" for ground based images with wide PSFs or
        "space" for images from space-based telescopes with a narrow PSF.

    Returns
    -------
    M: array of `int`
        Mask with significant coefficients in `starlets` set to `True`.
    """
    assert image_type in ("ground")

    if image_type == "space":
        # Calculate sigma_je, the standard deviation at
        # each scale due to gaussian noise
        shape = (get_scales(image.shape),) + image.shape

        noise_img = jnp.random.normal(size=image.shape)
        noise_starlet = starlet_transform(shape, noise_img, generation=1)
        sigma_je = jnp.zeros((len(noise_starlet),))
        for j, star in enumerate(noise_starlet):
            # sigma_je[j] = jnp.std(star)
            sigma_je = sigma_je.at[j].set( jnp.std(star) )

        noise = image - starlets[-1]

        last_sigma_i = sigma
        for it in range(max_iter):
            M = (jnp.abs(starlets) > K * sigma * sigma_je[:, None, None])
            S = jnp.sum(M, axis=0) == 0
            sigma_i = jnp.std(noise * S)
            if jnp.abs(sigma_i-last_sigma_i)/sigma_i < epsilon:
                break
            last_sigma_i = sigma_i
    else:
        # Sigma to use for significance at each scale
        # Initially we use the input `sigma`
        sigma_j = jnp.ones((len(starlets),), dtype=image.dtype) * sigma
        last_sigma_j = sigma_j
        for it in range(max_iter):
            M = (jnp.abs(starlets) > K * sigma_j[:, None, None])
            # Take the standard deviation of the current insignificant coeffs at each scale
            S = ~M
            sigma_j = jnp.std(starlets * S.astype(int), axis=(1, 2))
            # At lower scales all of the pixels may be significant,
            # so sigma is effectively zero. To avoid infinities we
            # only check the scales with non-zero sigma
            cut = sigma_j > 0
            if jnp.all(jnp.abs(sigma_j[cut] - last_sigma_j[cut]) / sigma_j[cut] < epsilon):
                break

            last_sigma_j = sigma_j
    return M.astype(int)


class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


def apply_wavelet_denoising(image, sigma=None, k=3, epsilon=1e-1, max_iter=20, image_type="ground", positive=True):
    """Apply wavelet denoising

    Uses the algorithm and notation from Starck et al. 2011, section 4.1

    Parameters
    ----------
    image: array-like
        The image to denoise
    sigma: float
        The standard deviation of the image
    k: float
        The threshold in units of sigma to declare a coefficient significant
    epsilon: float
        Convergence criteria for determining the support
    max_iter: int
        The maximum number of iterations. This applies to both finding the support
        and the denoising loop.
    image_type: str
        The type of image that is being used.
        This should be "ground" for ground based images with wide PSFs or
        "space" for images from space-based telescopes with a narrow PSF.
    positive: bool
        Whether or not the expected result should be positive

    Returns
    -------
    result: jnp.ndarray
        The resulting denoised image after `max_iter` iterations.
    """
    image_coeffs = starlet_transform(image)
    if sigma is None:
        sigma = jnp.median(jnp.absolute(image - jnp.median(image)))
    coeffs = image_coeffs.copy()
    support = get_multiresolution_support(image, coeffs, sigma, k, epsilon, max_iter, image_type)
    x = starlet_reconstruction(coeffs)

    for n in range(max_iter):
        coeffs = starlet_transform(x)
        x = x + starlet_reconstruction(support * (image_coeffs - coeffs))
        if positive:
            x[x<0] = 0
    return x
