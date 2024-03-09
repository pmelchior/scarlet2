import jax
import jax.numpy as jnp

def f_1(x,n):
    """
    We should Taylor expand instead
    (see galsim code)
    """
    return x*0.+1.

def f_2(x,n):
    px = jnp.pi * x
    return n * jnp.sin(px) * jnp.sin(px / n) / px / px

from functools import partial

def lanczos_n(x, n=3):
    """
    Lanczos interpolation kernel
    
    Parameters
    ----------
    n: Lanczos order
    """
    small_x = jnp.abs(x) <= 1e-4
    window_n = jnp.abs(x) <= n

    return jnp.piecewise(x, 
                         [small_x, (~small_x) & window_n], 
                         [f_1, f_2, lambda x,n: jnp.array(0)], n
                        )

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

    x = warp[...,0].flatten()
    y = warp[...,1].flatten()
 
    coords_x = coords[0,:,0]
    coords_y = coords[:,0,1]

    h = coords_x[1] - coords_x[0]
    
    xi = jnp.floor((x-coords_x[0])/h).astype(jnp.int32)
    yi = jnp.floor((y-coords_y[0])/h).astype(jnp.int32)

    Nx = coords.shape[0]
    Ny = coords.shape[1]
    
    def body_fun_x(i, args):
        res, yind, ky, masky = args
        
        xind = xi + i
        maskx = (xind >= 0) & (xind < Nx)

        kx = lanczos_n(
            (x - coords_x[xind])/h,
            n)

        k = kx*ky
        mask = maskx & masky
        res += jnp.where(mask, k * signal[yind, xind], 0)
        
        return res, yind, ky, masky

    def body_fun_y(i, args):
        res = args
        
        yind = yi + i
        masky = (yind >= 0) & (yind < Ny)

        ky = lanczos_n(
            (y - coords_y[yind])/h, 
            n)
        
        res = jax.lax.fori_loop(-n, n+1, body_fun_x, (res, yind, ky, masky))[0]
        
        return res

    res = jax.lax.fori_loop(-n, n+1, body_fun_y, jnp.zeros_like(x).astype(signal.dtype))
    
    return res.reshape(warp[...,0].shape)