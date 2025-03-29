import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
import numpy as np

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
    
    def f_1(self, x, n):
        """
        Approximation from galsim
        // res = n/(pi x)^2 * sin(pi x) * sin(pi x / n)
        //     ~= (1 - 1/6 pix^2) * (1 - 1/6 pix^2 / n^2)
        //     = 1 - 1/6 pix^2 ( 1 + 1/n^2 )
        """
        px = jnp.pi * x
        temp = 1./6. * px * px
        res = 1. - temp * (1. + 1. / (n * n))
        return jnp.ones_like(x) #res

    def f_2(self, x, n):
        px = jnp.pi * x
        return n * jnp.sin(px) * jnp.sin(px / n) / px / px

    def lanczos_n(self, x, n=3):
        """
        Lanczos interpolation kernel in direct space
        """
        small_x = jnp.abs(x) <= 1e-3
        window_n = jnp.abs(x) <= n

        return jnp.piecewise(
            x, 
            [small_x, (~small_x) & window_n], 
            [self.f_1, self.f_2, lambda x, n: jnp.array(0)], n
        )

    # def kernel(self, x):
    #     return self.lanczos_n(x, self.extent)
    
    def kernel(self, x):
        radius = self.extent
        y = radius * jnp.sin(np.pi * x) * jnp.sin(np.pi * x / radius)
        #  out = y / (np.pi ** 2 * x ** 2) where x >1e-3, 1 otherwise
        # out = jnp.where(jnp.abs(x) > 1e-3, jnp.divide(y, jnp.where(x != 0, np.pi**2 * x**2, 1)), 1)
        # return jnp.where(jnp.abs(x) > radius, 0., out)
        out = jnp.where(jnp.abs(x) > 1e-3, jnp.divide(y, jnp.where(x != 0, np.pi**2 * x**2, 1)), 1)
        return jnp.where(jnp.abs(x) > radius, 0., out)

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
        print("kx", kx)
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

"""
Rewritting the hermitian sampling operation without any for loops
"""

@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
@partial(jax.jit, static_argnames=("interp",))
def _interp_weight_1d_xval(ioff, xi, xp, x, nx, interp):
    xind = xi + ioff
    mskx = (xind >= 0) & (xind < nx)
    _x = x - (xp + ioff)
    wx = interp.kernel(_x)
    wx = jnp.where(mskx, wx, 0)
    return wx, xind.astype(jnp.int32)

def resample_2d_vectorized(signal, warp, xmin, ymin, interp=Quintic()):
    print("in the vmap resample ops")
    print(interp)
    orig_shape = warp[...,0].shape

    x = warp[..., 0].flatten()
    y = warp[..., 1].flatten()
    zp = signal

    x = x.ravel()
    xi = jnp.floor(x - xmin).astype(jnp.int32)
    xp = xi + xmin
    nx = zp.shape[1]

    y = y.ravel()
    yi = jnp.floor(y - ymin).astype(jnp.int32)
    yp = yi + ymin
    ny = zp.shape[0]

    irange = interp.extent #interp.ixrange // 2
    iinds = jnp.arange(-irange, irange + 1)

    wx, xind = _interp_weight_1d_xval(
        iinds,
        xi,
        xp,
        x,
        nx,
        interp,
    )

    wy, yind = _interp_weight_1d_xval(
        iinds,
        yi,
        yp,
        y,
        ny,
        interp,
    )

    z = jnp.sum(
        wx[None, :, :] * wy[:, None, :] * zp[yind[:, None, :], xind[None, :, :]],
        axis=(0, 1),
    )

    # we reshape on the way out to match the input shape
    return z.reshape(orig_shape)
    # return signal

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

# @partial(jax.jit, static_argnums=(2,3))
def resample_image(image, target_coords, interp, hermitian=False):
  """Resamples an image onto a target coordinate grid using Lanczos interpolation.

  Args:
    image: A 2D JAX array representing the image.
    target_coords: A 2D JAX array of shape (H_out, W_out, 2) where
      target_coords[i, j] contains the (x, y) coordinates in the
      original image to sample from for the output pixel at (i, j).

  Returns:
    A 2D JAX array of shape (H_out, W_out) representing the resampled image.
  """
  h_in, w_in = image.shape

  # Extract y and x coordinates from target_coords
  y_coords = target_coords[:, :, 1]
  x_coords = target_coords[:, :, 0]

  # Calculate integer and fractional parts of the coordinates
  y_floor = jnp.floor(y_coords).astype(int)
  x_floor = jnp.floor(x_coords).astype(int)
  y_frac = y_coords - y_floor
  x_frac = x_coords - x_floor

  # Define the support of the Lanczos kernel (a=3)
  kernel_radius = interp.extent

  # Create indices for the neighboring pixels to sample
  y_indices = jnp.arange(-kernel_radius, kernel_radius+1)
  x_indices = jnp.arange(-kernel_radius, kernel_radius+1)

  # Create a meshgrid of offsets
  dx, dy = jnp.meshgrid(x_indices, y_indices)

  # Calculate base neighbor coordinates
  y_neighbors = y_floor[:, :, None, None] + dy[None, None, :, :]
  x_neighbors = x_floor[:, :, None, None] + dx[None, None, :, :]

  # Check if neighbor indices are within bounds
  y_valid_mask = (y_neighbors >= 0) & (y_neighbors < h_in)
  x_valid_mask = (x_neighbors >= 0) & (x_neighbors < w_in)
  valid_mask = y_valid_mask & x_valid_mask

  # Clip neighbor indices to be within bounds for gathering
  y_neighbors_clipped = jnp.clip(y_neighbors, 0, h_in - 1)
  x_neighbors_clipped = jnp.clip(x_neighbors, 0, w_in - 1)

  # if hermitian:
    

  # Gather pixel values using the clipped indices
  neighbor_pixels = image[y_neighbors_clipped, x_neighbors_clipped]

  # Calculate weights based on the original offsets from the target coordinate
  wy = interp.kernel(dy - y_frac[:, :, None, None])
  wx = interp.kernel(dx - x_frac[:, :, None, None])
  weights = wy * wx

  # Mask out the weights of invalid neighbors
  weights = jnp.where(valid_mask, weights, 0.0)

  # Resample the image by taking a weighted sum of the neighboring pixels
  resampled_pixel = jnp.sum(weights * neighbor_pixels, axis=(2, 3))

  return resampled_pixel

def resample_ops(kimage, shape_in, shape_out, res_in, res_out, phi=None, flip_sign=None, interpolant=Quintic(),
                 v2=False):
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

    print("kimage.shape", kimage.shape)

    if v2:
        # kcoords_out += -shape_in/2
        print(kcoords_out.shape)
        k_resampled = jax.vmap(resample_image, in_axes=(0,None,None,None))(
            kimage,
            kcoords_out,
            interpolant,
            True
            )

    else:
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



"""
Rewritting the hermitian sampling operation without any for loops
"""

@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
@partial(jax.jit, static_argnames=("interpolant",))
def _interp_weight_1d_kval(ioff, kxi, kxp, kx, nkx, interpolant):

    kxind = (kxi + ioff) % nkx
    _kx = kx - (kxp + ioff)
    wkx = interpolant.kernel(_kx)

    return wkx, kxind.astype(jnp.int32)

def resample_hermitian_vmap(signal, warp, x_min, y_min, interpolant=Quintic()):
    
    orig_shape = warp[...,0].shape

    x = warp[..., 0].flatten()
    y = warp[..., 1].flatten()
    
    xi = jnp.floor(x-x_min).astype(jnp.int32)
    yi = jnp.floor(y-y_min).astype(jnp.int32)

    xp = xi + x_min
    yp = yi + y_min

    nkx_2 = signal.shape[1] - 1
    nkx = nkx_2 * 2
    
    nky = signal.shape[0]
    
    irange = interpolant.extent
    iinds = jnp.arange(-irange, irange + 1)

    wkx, kxind = _interp_weight_1d_kval(
        iinds,
        xi,
        xp,
        x,
        nkx,
        interpolant,
    )

    wky, kyind = _interp_weight_1d_kval(
        iinds,
        yi,
        yp,
        y,
        nky,
        interpolant,
    )

    wkx = wkx[None, :, :]
    kxind = kxind[None, :, :]
    wky = wky[:, None, :]
    kyind = kyind[:, None, :]

    val = jnp.where(
        kxind < nkx_2,
        signal[(nky - kyind) % nky, nkx - kxind - nkx_2].conjugate(),
        signal[kyind, kxind - nkx_2],
    )
    z = jnp.sum(
        val * wkx * wky,
        axis=(0, 1),
    )

    return z.reshape(orig_shape)


def resample_ops_new(kimage, shape_in, shape_out, res_in, res_out, phi=None, flip_sign=None, interpolant=Quintic()):
    """
    Resampling operation used in the Multiresolution Renderers
    This is assuming that the signal is Hermitian and starting at 0 on axis=2,
    i.e. f(-x, -y) = conjugate(f(x, y))
    """
    
    print("I am using the new stuff")

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

    
    # Apply inverse rotation to the frequencies
    if phi is not None:
        R = jnp.array([[jnp.cos(phi), jnp.sin(phi)],
                       [-jnp.sin(phi), jnp.cos(phi)]])
        
        b_shape = kcoords_out.shape
        kcoords_out = (R @ kcoords_out.reshape((-1, 2)).T).T.reshape((b_shape))

    
    k_resampled = jax.vmap(resample_hermitian_vmap, in_axes=(0,None,None,None,None))(
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


# def resample_hermitian_new(kimage, warp, x_min, y_min, interpolant=Quintic()):
