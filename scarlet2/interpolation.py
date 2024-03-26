import jax
import jax.numpy as jnp

### Quintic interpolant

def f_0_1_q(x):
    return 1 + x*x*x * (-95 + 138*x - 55*x*x) / 12

def f_1_2_q(x):
    return (x-1)*(x-2) * (-138 + 348*x - 249*x*x + 55*x*x*x) / 24


def f_2_3_q(x):
    return (x-2)*(x-3)*(x-3)*(-54 + 50*x - 11*x*x) / 24

def f_3_q(x):
    return jnp.zeros_like(x, dtype=x.dtype)

def quintic(x):
    x = jnp.abs(x) # quitic kernel is even

    b1 = x <= 1
    b2 = x <= 2
    b3 = x <= 3 

    return jnp.piecewise(
        x, 
        [b1, (~b1) & b2, (~b2) & b3], 
        [f_0_1_q, f_1_2_q, f_2_3_q, f_3_q]
        )

### Lanczos interpolant

def f_1(x, n):
    """
    Approximation from galsim
    // res = n/(pi x)^2 * sin(pi x) * sin(pi x / n)
    //     ~= (1 - 1/6 pix^2) * (1 - 1/6 pix^2 / n^2)
    //     = 1 - 1/6 pix^2 ( 1 + 1/n^2 )
    """
    px = jnp.pi * x
    temp = 1./6. * px * px
    res = 1. - temp * (1. + 1. / (n * n))
    return res


def f_2(x, n):
    px = jnp.pi * x
    return n * jnp.sin(px) * jnp.sin(px / n) / px / px

def lanczos_n(x, n=3):
    """
    Lanczos interpolation kernel

    Parameters
    ----------
    n: Lanczos order
    """
    small_x = jnp.abs(x) <= 1e-4
    window_n = jnp.abs(x) <= n

    return jnp.piecewise(
        x, [small_x, (~small_x) & window_n], [f_1, f_2, lambda x, n: jnp.array(0)], n
    )

### Resampling function

def resample2d(signal, coords, warp, n=3):
    """
    Resample a 2-dimensional image using a Lanczos kernel

    Parameters
    ----------
    signal: array
        2d array containing the signal. We assume here that the coordinates of
        the signal
        shape: [Nx, Ny]
    coords: array
        Coordinates on which the signal is sampled.
        shape: [Nx, Ny, 2]
            - x-coordinates are coords[0,:,0]
            - y-coordinates are coords[:,0,1]
    warp: array
        Coordinates on which to resample the signal.
        shape:[nx, ny, 2]
        [ [[0,  0], [0,  1], ...,  [0,  N-1]],
                           [ ... ],
          [[N-1,0], [N-1,1], ...,  [N-1,N  ]] ]
    n: Lanczos order

    Returns
    -------
    resampled_signal: array
    """

    x = warp[..., 0].flatten()
    y = warp[..., 1].flatten()

    coords_x = coords[0, :, 0]
    coords_y = coords[:, 0, 1]

    h = coords_x[1] - coords_x[0]

    xi = jnp.floor((x - coords_x[0]) / h).astype(jnp.int32)
    yi = jnp.floor((y - coords_y[0]) / h).astype(jnp.int32)

    Nx = coords.shape[0]
    Ny = coords.shape[1]

    def body_fun_x(i, args):
        res, yind, ky, masky = args

        xind = xi + i
        maskx = (xind >= 0) & (xind < Ny)

        kx = quintic((x - coords_x[xind]) / h)
        # kx = lanczos_n((x - coords_x[xind]) / h, n)

        k = kx * ky
        mask = maskx & masky
        res += jnp.where(mask, k * signal[yind, xind], 0)

        return res, yind, ky, masky

    def body_fun_y(i, args):
        res = args

        yind = yi + i
        masky = (yind >= 0) & (yind < Nx)

        ky = quintic((y - coords_y[yind]) / h)
        # ky = lanczos_n((y - coords_y[yind]) / h, n)

        res = jax.lax.fori_loop(-n, n + 1, body_fun_x, (res, yind, ky, masky))[0]

        return res

    res = jax.lax.fori_loop(
        -n, n + 1, body_fun_y, jnp.zeros_like(x).astype(signal.dtype)
    )

    return res.reshape(warp[..., 0].shape)
