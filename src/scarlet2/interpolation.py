"""Interpolation methods

Some of the code to perform interpolation in Fourier space as been adapted from
https://github.com/GalSim-developers/JAX-GalSim/blob/main/jax_galsim/interpolant.py
https://github.com/GalSim-developers/JAX-GalSim/blob/main/jax_galsim/interpolatedimage.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from .frame import _flip_matrix, _rot_matrix


### Interpolant class
class Interpolant(eqx.Module):
    """Base class for interpolants"""

    extent: int
    """Size of the interpolation kernel"""

    def __call__(self):
        """Code to execute when the class is called"""
        raise NotImplementedError

    def kernel(self, x):
        """Evaluate the kernel in configuration space at location `x`

        Parameters
        ----------
        x: float or array
            Position in real space

        Returns
        -------
        float or array
        """
        raise NotImplementedError

    def uval(self, u):
        """Evaluate the kernel in Fourier space at frequency `u`

        Parameters
        ----------
        u: complex or array
            Position in Fourier space

        Returns
        -------
        complex or array
        """
        raise NotImplementedError


### Quintic interpolant
class Quintic(Interpolant):
    """Quintic interpolation from Gruen & Bernstein (2014)"""

    def __init__(self):
        self.extent = 3

    def _f_0_1_q(self, x):
        return 1 + x * x * x * (-95 + 138 * x - 55 * x * x) / 12

    def _f_1_2_q(self, x):
        return (x - 1) * (x - 2) * (-138 + 348 * x - 249 * x * x + 55 * x * x * x) / 24

    def _f_2_3_q(self, x):
        return (x - 2) * (x - 3) * (x - 3) * (-54 + 50 * x - 11 * x * x) / 24

    def _f_3_q(self, x):
        return jnp.zeros_like(x, dtype=x.dtype)

    def kernel(self, x):
        """See parent class"""
        x = jnp.abs(x)  # quitic kernel is even

        b1 = x <= 1
        b2 = x <= 2
        b3 = x <= 3

        return jnp.piecewise(
            x, [b1, (~b1) & b2, (~b2) & b3], [self._f_0_1_q, self._f_1_2_q, self._f_2_3_q, self._f_3_q]
        )

    def uval(self, u):
        """See parent class"""
        u = jnp.abs(u)
        s = jnp.sinc(u)
        piu = jnp.pi * u
        c = jnp.cos(piu)
        ssq = s * s
        piusq = piu * piu

        return s * ssq * ssq * (s * (55.0 - 19.0 * piusq) + 2.0 * c * (piusq - 27.0))


class Lanczos(Interpolant):
    """Lanczos interpolation class"""

    def __init__(self, n):
        """Lanczos interpolant

        Parameters
        ----------
        n: int
            Lanczos order
        """
        self.extent = n

    def _f_1(self, x, n):
        # Approximation from galsim
        # res = n/(pi x)^2 * sin(pi x) * sin(pi x / n)
        #     ~= (1 - 1/6 pix^2) * (1 - 1/6 pix^2 / n^2)
        #     = 1 - 1/6 pix^2 ( 1 + 1/n^2 )
        px = jnp.pi * x
        temp = 1.0 / 6.0 * px * px
        res = 1.0 - temp * (1.0 + 1.0 / (n * n))
        return res

    def _f_2(self, x, n):
        px = jnp.pi * x
        return n * jnp.sin(px) * jnp.sin(px / n) / px / px

    def _lanczos_n(self, x, n=3):
        small_x = jnp.abs(x) <= 1e-4
        window_n = jnp.abs(x) <= n

        return jnp.piecewise(
            x, [small_x, (~small_x) & window_n], [self._f_1, self._f_2, lambda x, n: jnp.array(0)], n
        )

    def kernel(self, x):
        """See parent class"""
        return self._lanczos_n(x, self.extent)


### Resampling function


def resample2d(signal, coords, warp, interpolant=Lanczos(3)):  # noqa: B008
    """Resample a 2-dimensional image using a Lanczos kernel

    Parameters
    ----------
    signal: array
        2d array containing the signal. We assume here that the coordinates of
        the signal. Shape: `[Ny, Nx]`
    coords: array
        Coordinates on which the signal is sampled.
        Shape: `[Ny, Nx, 2]`
        y-coordinates are `coords[0,:,0]`, x-coordinates are `coords[:,0,1]`.
    warp: array
        Coordinates on which to resample the signal.
        Shape:[ny, nx, 2]
        [
        [[0,  0], [0,  1], ...,  [0,  N-1]],
        [ ... ],
        [[N-1,0], [N-1,1], ...,  [N-1,N  ]]
        ]
    interpolant: Interpolant
        Instance of interpolant

    Returns
    -------
    array
        Resampled `signal` at the location indicated by `warp`
    """
    y = warp[..., 0].flatten()
    x = warp[..., 1].flatten()

    coords_y = coords[0, :, 0]
    coords_x = coords[:, 0, 1]

    h = coords_x[1] - coords_x[0]

    xi = jnp.floor((x - coords_x[0]) / h).astype(jnp.int32)
    yi = jnp.floor((y - coords_y[0]) / h).astype(jnp.int32)

    n_y = coords.shape[0]
    n_x = coords.shape[1]

    def body_fun_x(i, args):
        res, yind, ky, masky = args

        xind = xi + i
        maskx = (xind >= 0) & (xind < n_x)

        kx = interpolant.kernel((x - coords_x[xind]) / h)

        k = kx * ky
        mask = maskx & masky
        res += jnp.where(mask, k * signal[yind, xind], 0)

        return res, yind, ky, masky

    def body_fun_y(i, args):
        res = args

        yind = yi + i
        masky = (yind >= 0) & (yind < n_y)

        ky = interpolant.kernel((y - coords_y[yind]) / h)

        res = jax.lax.fori_loop(
            -interpolant.extent, interpolant.extent + 1, body_fun_x, (res, yind, ky, masky)
        )[0]

        return res

    res = jax.lax.fori_loop(
        -interpolant.extent, interpolant.extent + 1, body_fun_y, jnp.zeros_like(x).astype(signal.dtype)
    )

    return res.reshape(warp[..., 0].shape)


# @partial(jax.jit, static_argnums=(3))
def resample3d(signal, coords, warp, interpolant):
    """Resample a 3-dimensional image using a Lanczos kernel

    Parameters
    ----------
    signal: array
        3d array containing the signal. We assume here that the coordinates of
        the signal. Shape: `[C, Ny, Nx]`
    coords: array
        Coordinates on which the signal is sampled.
        Shape: `[Ny, Nx, 2]`
        y-coordinates are `coords[0,:,0]`, x-coordinates are `coords[:,0,1]`.
    warp: array
        Coordinates on which to resample the signal.
        Shape:[ny, nx, 2]
        [
        [[0,  0], [0,  1], ...,  [0,  N-1]],
        [ ... ],
        [[N-1,0], [N-1,1], ...,  [N-1,N  ]]
        ]
    interpolant: Interpolant
        Instance of interpolant

    Returns
    -------
    array
        Resampled `signal` at the location indicated by `warp`

    See Also
    --------
    resample2d
    """
    _resample2d = lambda s: resample2d(s, coords, warp, interpolant=interpolant)
    return jax.vmap(_resample2d, in_axes=0, out_axes=0)(signal)


def resample_hermitian(signal, warp, x_min, y_min, interpolant=Quintic()):  # noqa: B008
    """Resample a 2-dimensional image using an interpolation kernel

    This is assuming that the signal is Hermitian and starting at 0 on axis=2,
    i.e. f(-x, -y) = conjugate(f(x, y))

    Parameters
    ----------
    signal: array
        2d array containing the signal. We assume here that the coordinates of
        the signal
        shape: [Nx, Ny]
    warp: array
        Coordinates on which to resample the signal, in the grid of signal
        coordinates [[0 ... signal.shape[0]], [0 ... signal.shape[1]]
        shape:[nx, ny, 2]
        [
        [[0,  0], [0,  1], ...,  [0,  N-1]],
        [ ... ],
        [[N-1,0], [N-1,1], ...,  [N-1,N  ]]
        ]
    x_min: float
        Left coordinate of corner of bounding box that defines the location of `signal`
    y_min: float
        Low coordinate of corner of bounding box that defines the location of `signal`
    interpolant: Interpolant
        Instance of interpolant

    Returns
    -------
    array
        Resampled `signal` at the location indicated by `warp`
    """

    x = warp[..., 0].flatten()
    y = warp[..., 1].flatten()

    xi = jnp.floor(x - x_min).astype(jnp.int32)
    yi = jnp.floor(y - y_min).astype(jnp.int32)

    xp = xi + x_min
    yp = yi + y_min

    nkx_2 = signal.shape[1] - 1
    nkx = signal.shape[0]

    def body_fun_x(i, args):
        res, yind, ky = args

        xind = (xi + i) % nkx

        kx = interpolant.kernel(x - (xp + i))

        k = kx * ky

        tmp = jnp.where(
            xind < nkx_2,
            signal[(nkx - yind) % nkx, nkx - xind - nkx_2].conjugate(),
            signal[yind, xind - nkx_2],
        )

        res += tmp * k

        return res, yind, ky

    def body_fun_y(i, args):
        res = args

        yind = yi + i

        ky = interpolant.kernel(y - (yp + i))

        res = jax.lax.fori_loop(-interpolant.extent, interpolant.extent + 1, body_fun_x, (res, yind, ky))[0]

        return res

    res = jax.lax.fori_loop(
        -interpolant.extent, interpolant.extent + 1, body_fun_y, jnp.zeros_like(x).astype(signal.dtype)
    )

    return res.reshape(warp[..., 0].shape)


def resample_ops(
    kimage,
    shape_in,
    shape_out,
    scale=1,
    angle=0,
    flip=1,
    shift=(0, 0),
    interpolant=Quintic(),  # noqa: B008
):
    """Resampling operation

    This method uses the Fourier space resampling technique from Gruen & Bernstein (2014).
    It assumes that the signal is Hermitian and starting at 0 on axis=2,
    i.e. f(-x, -y) = conjugate(f(x, y))

    Parameters
    ----------
    kimage: array
        Complex array of image in Fourier space
    shape_in: tuple
        Shape of input image in configuration space
    shape_out: tuple
        Shape of output image in configuration space
    scale: float
        Scaling factor (= pixel_size_in / pixel_size_out)
    angle: float
        Angle of rotation (in radians, counter-clockwise) around the center of the image
    flip: -1 or 1
        Flip of the y-axis (if negative) to get an improper rotation
    shift: tuple
        Shift of the output image (in units of output pixels)
    interpolant: Interpolant
        Interpolation kernel function

    Returns
    -------
    array
    """

    # Apply rescaling to the frequencies
    # [0, Fe/2+1]
    # [-Fe/2+1, Fe/2]
    kcoords_out = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, shape_in / 2, shape_out // 2 + 1),
            jnp.linspace(-shape_in / 2, shape_in / 2, shape_out),
        ),
        -1,
    )
    # Apply rotation to the frequencies
    R = _rot_matrix(-angle) @ _flip_matrix(flip)  # noqa: N806
    b_shape = kcoords_out.shape
    kcoords_out = (R @ kcoords_out.reshape((-1, 2)).T).T.reshape(b_shape)

    # k interpolation of original signal
    k_resampled = jax.vmap(resample_hermitian, in_axes=(0, None, None, None, None))(
        kimage, kcoords_out * scale, -shape_in / 2, -shape_in / 2, interpolant
    )
    # fft of x-interpolant
    uscale = 1.0 / shape_in  # (2.0 * jnp.pi)
    xint_val = interpolant.uval(kcoords_out[..., 0] * uscale) * interpolant.uval(kcoords_out[..., 1] * uscale)

    # # apply shift
    # shift_ = shift[::-1]  # x,y needed here
    # pfac = jnp.exp(-1j * (kcoords_out[..., 0] * shift_[0] + kcoords_out[..., 1] * shift_[1]))

    return k_resampled * jnp.expand_dims(xint_val, 0)  # * pfac
