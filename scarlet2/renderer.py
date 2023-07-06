import jax.numpy as jnp

from .fft import convolve, deconvolve, _get_fast_shape
from .module import Module

class Renderer(Module):
    def __call__(self, model):
        raise NotImplementedError

    def get_channel_map(self, model_frame, obs_frame):
        if list(obs_frame.channels) == list(model_frame.channels):
            return None

        channel_map = [
            list(model_frame.channels).index(c) for c in list(obs_frame.channels)
        ]
        min_channel = min(channel_map)
        max_channel = max(channel_map)
        if max_channel + 1 - min_channel == len(channel_map):
            channel_map = slice(min_channel, max_channel + 1)
        return channel_map

    def map_channels(self, model):
        """Map to model channels onto the observation channels
        Parameters
        ----------
        model: array
            The hyperspectral model
        Returns
        -------
        obs_model: array
            `model` mapped onto the observation channels
        """
        if self.channel_map is None:
            return model
        if isinstance(self.channel_map, slice):
            return model[self.channel_map]
        return jnp.dot(self.channel_map, model)


class NoRenderer(Renderer):
    def __call__(self, model):
        return model

class ConvolutionRenderer(Renderer):
    def __init__(self, frame, obs_frame, *parameters):
        # create PSF model
        psf_model = jnp.tile(frame.psf(), (obs_frame.bbox.shape[0], 1, 1))
        # make sure fft uses a shape large enough to cover the convolved model 
        fft_shape = _get_fast_shape(frame.bbox.shape, psf_model.shape, padding=3, axes=(-2, -1))
        # compute and store diff kernel in Fourier space
        diff_kernel_fft = deconvolve(obs_frame.psf(), psf_model, axes=(-2, -1), fft_shape=fft_shape, return_fft=True)
        object.__setattr__(self, "_diff_kernel_fft", diff_kernel_fft)
        object.__setattr__(self, "channel_map", self.get_channel_map(frame, obs_frame))
        
    def __call__(self, model):
        # TODO: including slices 
        model_ = self.map_channels(model)
        return convolve(model_, self._diff_kernel_fft, axes=(-2, -1))

