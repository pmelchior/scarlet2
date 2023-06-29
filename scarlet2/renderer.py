import equinox as eqx
import jax.numpy as jnp

from .fft import convolve, deconvolve, _get_fast_shape


class Renderer(eqx.Module):
    pass

class NoRenderer(Renderer):
    def __call__(self, model):
        return model


class ConvolutionRenderer(Renderer):

    def __init__(self, frame, obs_frame):
        # create PSF model
        psf = frame.psf()
        if len(psf.shape) == 2:  # only one image for all bands
            psf_model = jnp.tile(psf, (obs_frame.bbox.shape[0], 1, 1))
        else:
            psf_model = psf
        # make sure fft uses a shape large enough to cover the convolved model
        fft_shape = _get_fast_shape(frame.bbox.shape, psf_model.shape, padding=3, axes=(-2, -1))
        # compute and store diff kernel in Fourier space
        diff_kernel_fft = deconvolve(obs_frame.psf(), psf_model, axes=(-2, -1), fft_shape=fft_shape, return_fft=True)
        object.__setattr__(self, "_diff_kernel_fft", diff_kernel_fft)

    def __call__(self, model):
        # TODO: including slices
        return convolve(model, self._diff_kernel_fft, axes=(-2, -1))
