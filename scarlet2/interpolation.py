import jax
import jax.numpy as jnp
import equinox as eqx

"""
Some of the code to perform interpolation in Fourier space as been adapted from
https://github.com/GalSim-developers/JAX-GalSim/blob/main/jax_galsim/interpolant.py
https://github.com/GalSim-developers/JAX-GalSim/blob/main/jax_galsim/interpolatedimage.py
"""

### Interpolant class

class Interpolant(eqx.Module):
    extent: int
    def __call__(
        self):
        raise NotImplementedError

    def kernel(self, x):
        raise NotImplementedError
    
    def uval(self, u):
        raise NotImplementedError

### Quintic interpolant

class Quintic(Interpolant):
    def __init__(self):
        self.extent = 3

    def f_0_1_q(self, x):
        return 1 + x*x*x * (-95 + 138*x - 55*x*x) / 12

    def f_1_2_q(self, x):
        return (x-1)*(x-2) * (-138 + 348*x - 249*x*x + 55*x*x*x) / 24


    def f_2_3_q(self, x):
        return (x-2)*(x-3)*(x-3)*(-54 + 50*x - 11*x*x) / 24

    def f_3_q(self, x):
        return jnp.zeros_like(x, dtype=x.dtype)

    def kernel(self, x):
        """
        Quintic kernel values in direct space
        """
        x = jnp.abs(x) # quitic kernel is even

        b1 = x <= 1
        b2 = x <= 2
        b3 = x <= 3 

        return jnp.piecewise(
            x, 
            [b1, (~b1) & b2, (~b2) & b3], 
            [self.f_0_1_q, self.f_1_2_q, self.f_2_3_q, self.f_3_q]
            )
    
    def uval(self, u):
        """
        Quintic kernel values in Fourier space
        """
        u = jnp.abs(u)
        s = jnp.sinc(u)
        piu = jnp.pi*u
        c = jnp.cos(piu)
        ssq = s*s
        piusq = piu*piu
        
        return s * ssq * ssq * (s * (55. - 19. * piusq) + 2. * c * (piusq - 27.))

### Lanczos interpolant

class Lanczos(Interpolant):
    
    def __init__(self, n):
        """
        Lanczos interpolant

        Parameters
        ----------
        n: Lanczos order
        """

        self.extent = n
    
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
        Lanczos interpolation kernel in direct space
        """
        small_x = jnp.abs(x) <= 1e-4
        window_n = jnp.abs(x) <= n

        return jnp.piecewise(
            x, 
            [small_x, (~small_x) & window_n], 
            [f_1, f_2, lambda x, n: jnp.array(0)], n
        )

    def kernel(self, x):
        return self.lanczos_n(x, self.extent)

### Resampling function

def resample2d(signal, coords, warp, interpolant=Quintic()):
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

        kx = interpolant.kernel((x - coords_x[xind]) / h)

        k = kx * ky
        mask = maskx & masky
        res += jnp.where(mask, k * signal[yind, xind], 0)

        return res, yind, ky, masky

    def body_fun_y(i, args):
        res = args

        yind = yi + i
        masky = (yind >= 0) & (yind < Nx)

        ky = interpolant.kernel((y - coords_y[yind]) / h)

        res = jax.lax.fori_loop(-interpolant.extent, interpolant.extent + 1, body_fun_x, (res, yind, ky, masky))[0]

        return res

    res = jax.lax.fori_loop(
        -interpolant.extent, interpolant.extent + 1, body_fun_y, jnp.zeros_like(x).astype(signal.dtype)
    )

    return res.reshape(warp[..., 0].shape)

def resample_hermitian(signal, warp, x_min, y_min, interpolant=Quintic()):
    """
    Resample a 2-dimensional image using a Lanczos kernel
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

    xi = jnp.floor(x-x_min).astype(jnp.int32)
    yi = jnp.floor(y-y_min).astype(jnp.int32)

    xp = xi + x_min
    yp = yi + y_min

    nkx_2 = signal.shape[1]-1
    nkx = signal.shape[0]

    def body_fun_x(i, args):
        res, yind, ky = args

        xind = (xi + i) % nkx
        
        kx = interpolant.kernel(x - (xp+i))

        k = kx * ky
        
        tmp = jnp.where(xind < nkx_2,
                        signal[(nkx - yind) % nkx, nkx - xind - nkx_2].conjugate(),
                        signal[yind, xind - nkx_2]
                        )
        
        res += tmp * k

        return res, yind, ky

    def body_fun_y(i, args):
        res = args

        yind = yi + i

        ky = interpolant.kernel(y - (yp+i))

        res = jax.lax.fori_loop(-interpolant.extent, interpolant.extent + 1, body_fun_x, (res, yind, ky))[0]

        return res

    res = jax.lax.fori_loop(
        -interpolant.extent, interpolant.extent + 1, body_fun_y, jnp.zeros_like(x).astype(signal.dtype)
    )

    return res.reshape(warp[..., 0].shape)

def resample_ops(kimage, shape_in, shape_out, res_in, res_out, phi=None, flip_sign=None, interpolant=Quintic()):
    """
    Resampling operation used in the Multiresolution Renderers
    This is assuming that the signal is Hermitian and starting at 0 on axis=2,
    i.e. f(-x, -y) = conjugate(f(x, y))
    """
    
    # Apply rescaling to the frequencies
    # [0, Fe/2+1]
    # [-Fe/2+1, Fe/2]
    kcoords_out = jnp.stack(jnp.meshgrid(
        jnp.linspace(0, 
                     shape_in / 2 / res_out * res_in, 
                     shape_out // 2 + 1),
        jnp.linspace(-shape_in / 2 / res_out * res_in, 
                     shape_in / 2 / res_out * res_in, 
                     shape_out)
        ), -1)

    
    # Apply rotation to the frequencies
    if phi is not None:
        R = jnp.array([[jnp.cos(phi), jnp.sin(phi)],
                       [-jnp.sin(phi), jnp.cos(phi)]])
        
        b_shape = kcoords_out.shape
        kcoords_out = (R @ kcoords_out.reshape((-1, 2)).T).T.reshape((b_shape))

    k_resampled = jax.vmap(resample_hermitian, in_axes=(0,None,None,None,None))(
        kimage,
        kcoords_out,
        -shape_in/2,
        -shape_in/2,
        interpolant
        )
    
    kx = jnp.linspace(0, jnp.pi, shape_out//2 + 1) * res_in/res_out
    ky = jnp.linspace(-jnp.pi, jnp.pi, shape_out)
    coords = jnp.stack(jnp.meshgrid(kx, ky),-1) / 2 / jnp.pi

    xint_val = interpolant.uval(coords[...,0]) * interpolant.uval(coords[...,1])
    
    return k_resampled * jnp.expand_dims(xint_val, 0)