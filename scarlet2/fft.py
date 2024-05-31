import operator
import math

import jax.lax
import jax.numpy as jnp
from scipy import fftpack


def transform(image, fft_shape, axes=None):
    """The FFT of an image for a given `fft_shape` along desired `axes`

    Parameters
    ----------
    image: array
        The real-space image.
    fft_shape: tuple
        "Fast" shape of the image used to generate the FFT.
    axes: int or tuple
        The dimension(s) of the array that will be transformed.
    """

    # TODO: allow fft_shape = None to automatically determine it

    if axes is None:
        axes = range(len(image.shape))
    else:
        try:
            iter(axes)
        except TypeError:
            axes = (axes,)

    if len(fft_shape) != len(axes):
        msg = (
            "fft_shape self.axes must have the same number of dimensions, got {0}, {1}"
        )
        raise ValueError(msg.format(fft_shape, axes))

    image = _pad(image, fft_shape, axes)
    image = jnp.fft.ifftshift(image, axes)
    image_fft = jnp.fft.rfftn(image, axes=axes)
    return image_fft


def inverse(image_fft, fft_shape, image_shape, axes=None):
    """Generate image from its FFT

    Parameters
    ----------
    image_fft: array
        The FFT of the image.
    fft_shape: tuple
        "Fast" shape of the image used to generate the FFT.
    image_shape: tuple
        The actual shape of the image *before padding* and using fast shapes.
        This will regenerate the image with the extra padding trimmed off.
    axes: int or tuple
        The dimension(s) of the array that will be transformed.
    """

    # TODO: allow fft_shape = None to automatically determine it

    if axes is None:
        axes = range(len(image_fft.shape))
    else:
        try:
            iter(axes)
        except TypeError:
            axes = (axes,)

    image = jnp.fft.irfftn(image_fft, fft_shape, axes=axes)
    # Shift the center of the image from the bottom left to the center
    image = jnp.fft.fftshift(image, axes=axes)
    # Trim the image to remove the padding added
    # to reduce fft artifacts
    image = _trim(image, image_shape)
    return image


def convolve(image, kernel, padding=3, axes=None, fft_shape=None, return_fft=False):
    """Convolve image with a kernel

    Parameters
    ----------
    image: array
        Image array
    kernel: array
        Convolution kernel array
    padding: int
        Additional padding to use when generating the FFT
        to supress artifacts.
    axes: tuple or None
        Axes that contain the spatial information for the PSFs.
    """
    return _kspace_op(
        image,
        kernel,
        operator.mul,
        padding=padding,
        axes=axes,
        fft_shape=fft_shape,
        return_fft=return_fft,
    )


def deconvolve(image, kernel, padding=3, axes=None, fft_shape=None, return_fft=False):
    """Deconvolve image with a kernel

    This is usually unstable. Treat with caution!

    Parameters
    ----------
    image: array
        Image array
    kernel: array
        Convolution kernel array
    padding: int
        Additional padding to use when generating the FFT
        to supress artifacts.
    axes: tuple or None
        Axes that contain the spatial information for the PSFs.
    """

    return _kspace_op(
        image,
        kernel,
        operator.truediv,
        padding=padding,
        fft_shape=fft_shape,
        axes=axes,
        return_fft=return_fft,
    )


def _kspace_op(
    image, kernel, f, padding=3, axes=None, fft_shape=None, return_fft=False
):
    if axes is None:
        axes = range(len(image.shape))
    else:
        try:
            iter(axes)
        except TypeError:
            axes = (axes,)

    # assumes kernel FFT has been computed with large enough shape to cover also image
    if kernel.dtype in (jnp.complex64, jnp.complex128):
        fft_shape = [kernel.shape[ax] for ax in axes]
        fft_shape[-1] = 2 * (
            fft_shape[-1] - 1
        )  # real-valued FFT has 1/2 of the frequencies
        kernel_fft = kernel
    else:
        if fft_shape is None:
            fft_shape = _get_fast_shape(
                image.shape, kernel.shape, padding=padding, axes=axes
            )
        kernel_fft = transform(kernel, fft_shape, axes=axes)

    image_fft = transform(image, fft_shape, axes=axes)
    image_fft_ = f(image_fft, kernel_fft)
    if return_fft:
        return image_fft_
    image_ = inverse(image_fft_, fft_shape, image.shape, axes=axes)
    return image_


def _get_fast_shape(im_or_shape1, im_or_shape2, axes=None, padding=3, max_shape=False):
    """Return the fast fft shapes for each spatial axis

    Calculate the fast fft shape for each dimension in
    axes.
    """
    if hasattr(im_or_shape1, "shape"):
        shape1 = im_or_shape1.shape
    else:
        shape1 = im_or_shape1
    if hasattr(im_or_shape2, "shape"):
        shape2 = im_or_shape2.shape
    else:
        shape2 = im_or_shape2

    # Make sure the shapes are the same size
    if len(shape1) != len(shape2):
        msg = (
            "img1 and img2 must have the same number of dimensions, but got {0} and {1}"
        )
        raise ValueError(msg.format(len(shape1), len(shape2)))

    if axes is None:
        axes = range(len(shape1))
    else:
        try:
            iter(axes)
        except TypeError:
            axes = (axes,)

    # Set the combined shape based on the total dimensions
    combine_shapes = lambda s1, s2: max(s1, s2) if max_shape else s1 + s2
    shape = [combine_shapes(shape1[ax], shape2[ax]) + padding for ax in axes]
    # Use the next fastest shape in each dimension
    # TODO: check what jnp actually uses for FFTs
    shape = [fftpack.helper.next_fast_len(s) for s in shape]
    return shape


def _trim(arr, newshape):
    """Return the center newshape portion of the array.

    Note: If the array shape is odd and the target is even,
    the center of `arr` is shifted to the center-right
    pixel position.
    This is slightly different than the scipy implementation,
    which uses the center-left pixel for the array center.
    The reason for the difference is that we have
    adopted the convention of `np.fft.fftshift` in order
    to make sure that changing back and forth from
    fft standard order (0 frequency and position is
    in the bottom left) to 0 position in the center.
    """
    startind = tuple((c - s + 1) // 2 for c, s in zip(arr.shape, newshape))
    return jax.lax.dynamic_slice(arr, startind, newshape)


def _fast_zero_pad(arr, pad_width):
    """Fast version of numpy.pad when `mode="constant"`

    Executing `numpy.pad` with zeros is ~1000 times slower
    because it doesn't make use of the `zeros` method for padding.

    Parameters
    ---------
    arr: array
        The array to pad
    pad_width: tuple
        Number of values padded to the edges of each axis.
        See numpy docs for more.

    Returns
    -------
    result: array
        The array padded with `constant_values`
    """
    newshape = tuple([a + ps[0] + ps[1] for a, ps in zip(arr.shape, pad_width)])
    result = jnp.zeros(newshape, dtype=arr.dtype)
    start = tuple(start for start, end in pad_width)
    result = jax.lax.dynamic_update_slice(result, arr, start_indices=start)
    return result


def _pad(arr, newshape, axes=None, mode="constant", constant_values=0):
    """Pad an array to fit into newshape

    Pad `arr` with zeros to fit into newshape,
    which uses the `np.fft.fftshift` convention of moving
    the center pixel of `arr` (if `arr.shape` is odd) to
    the center-right pixel in an even shaped `newshape`.
    """
    if axes is None:
        newshape = jnp.asarray(newshape)
        currshape = jnp.array(arr.shape)
        dS = newshape - currshape
        startind = (dS + 1) // 2
        endind = dS - startind
        pad_width = list(zip(startind, endind))
    else:
        # only pad the axes that will be transformed
        pad_width = [(0, 0) for axis in arr.shape]
        try:
            len(axes)
        except TypeError:
            axes = [axes]
        for a, axis in enumerate(axes):
            dS = newshape[a] - arr.shape[axis]
            startind = (dS + 1) // 2
            endind = dS - startind
            pad_width[axis] = (startind, endind)

    # if mode == "constant" and constant_values == 0:   
    # result = _fast_zero_pad(arr, pad_width)
    # else:
    result = jnp.pad(arr, pad_width, mode=mode)
    return result


def good_fft_size(input_size):
    # Code from JAX-Galsim
    # https://github.com/GalSim-developers/JAX-GalSim/blob/4b12d6b3520938cd823ae3978c400bb9a2b454d3/jax_galsim/image.py#L830
    # Reference from GalSim C++
    # https://github.com/GalSim-developers/GalSim/blob/ece3bd32c1ae6ed771f2b489c5ab1b25729e0ea4/src/Image.cpp#L1009
    # Reduce slightly to eliminate potential rounding errors:
    insize = (1.0 - 1.0e-5) * input_size
    log2n = math.log(2.0) * math.ceil(math.log(insize) / math.log(2.0))
    log2n3 = math.log(3.0) + math.log(2.0) * math.ceil(
        (math.log(insize) - math.log(3.0)) / math.log(2.0)
    )
    log2n3 = max(log2n3, math.log(6.0))  # must be even number
    Nk = max(int(math.ceil(math.exp(min(log2n, log2n3)) - 1.0e-5)), 2)
    return Nk

def wrap_hermitian_x(im, im_xmin, im_ymin, wrap_xmin, wrap_ymin, wrap_nx, wrap_ny):
    """
    Bernstein & Gruen (2014) arxiv:1401.2636
    This function is taken from JAX-Galsim wrap_image utils written by Matthew R. Becker
    https://github.com/GalSim-developers/JAX-GalSim/blob/4b12d6b3520938cd823ae3978c400bb9a2b454d3/jax_galsim/core/wrap_image.py#L6C1-L54C40
    """

    def wrap_nonhermitian(im, xmin, ymin, nxwrap, nywrap):
        def _body_j(j, vals):
            i, im = vals

            ii = (i - ymin) % nywrap + ymin
            jj = (j - xmin) % nxwrap + xmin

            im = jax.lax.cond(
                # weird way to say if ii != i and jj != j
                # I tried other ways and got test failures
                jnp.abs(ii - i) + jnp.abs(jj - j) != 0,
                lambda im, i, j, ii, jj: im.at[ii, jj].add(im[i, j]),
                lambda im, i, j, ii, jj: im,
                im,
                i,
                j,
                ii,
                jj,
            )

            return [i, im]

        def _body_i(i, vals):
            im = vals
            _, im = jax.lax.fori_loop(0, im.shape[1], _body_j, [i, im])
            return im

        im = jax.lax.fori_loop(0, im.shape[0], _body_i, im)
        return im


    def expand_hermitian_x(im):
        return jnp.concatenate([im[:, 1:][::-1, ::-1].conjugate(), im], axis=1)


    def contract_hermitian_x(im):
        return im[:, im.shape[1] // 2 :]

    
    im_exp = expand_hermitian_x(im)
    im_exp = wrap_nonhermitian(
        im_exp, wrap_xmin - im_xmin, wrap_ymin - im_ymin, wrap_nx, wrap_ny
    )
    return contract_hermitian_x(im_exp)