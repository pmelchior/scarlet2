import equinox as eqx
import jax.numpy as jnp
import jax

from .fft import convolve, deconvolve, _get_fast_shape, transform, inverse

from .interpolation import resample2d
# from jii import _lanczos_interp2d, resample2d_

class Renderer(eqx.Module):
    def __call__(self, model, key=None):  # key is needed to chain renderers with eqx.nn.Sequential
        raise NotImplementedError


class NoRenderer(Renderer):
    def __call__(self, model, key=None):
        return model


class ChannelRenderer(Renderer):
    channel_map: (None, list, slice) = None

    def __init__(self, model_frame, obs_frame):
        if obs_frame.channels == model_frame.channels:
            channel_map = None
        else:
            try:
                channel_map = [
                    list(model_frame.channels).index(c) for c in list(obs_frame.channels)
                ]
            except ValueError:
                msg = "Cannot match channels between model and observation.\n"
                msg += f"Got {model_frame.channels} and {obs_frame.channels}."
                raise ValueError(msg)

            min_channel = min(channel_map)
            max_channel = max(channel_map)
            if max_channel + 1 - min_channel == len(channel_map):
                channel_map = slice(min_channel, max_channel + 1)
        self.channel_map = channel_map

    def __call__(self, model, key=None):
        """Map model channels onto the observation channels

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
        if isinstance(self.channel_map, (slice, list)):
            return model[self.channel_map, :, :]
        # not yet used by any renderer: full matrix mapping between model and observation channels
        return jnp.dot(self.channel_map, model)

class ConvolutionRenderer(Renderer):
    def __init__(self, model_frame, obs_frame):
        # create PSF model
        psf = model_frame.psf()
        if len(psf.shape) == 2:  # only one image for all bands
            psf_model = jnp.tile(psf, (obs_frame.bbox.shape[0], 1, 1))
        else:
            psf_model = psf
        # make sure fft uses a shape large enough to cover the convolved model
        fft_shape = _get_fast_shape(model_frame.bbox.shape, psf_model.shape, padding=3, axes=(-2, -1))

        # compute and store diff kernel in Fourier space
        diff_kernel_fft = deconvolve(obs_frame.psf(), psf_model, axes=(-2, -1), fft_shape=fft_shape, return_fft=True)
        object.__setattr__(self, "_diff_kernel_fft", diff_kernel_fft)

    def __call__(self, model, key=None):
        return convolve(model, self._diff_kernel_fft, axes=(-2, -1))

class KDeconvRenderer(Renderer):
    """
    Perform decovolution in Fourier space and return the Fourier decovolved
    model
    """
    def __init__(self, model_frame, obs_frame):
        # create PSF model
        psf = model_frame.psf()
        if len(psf.shape) == 2:  # only one image for all bands
            # print("I am tiling the psf")
            psf_model = jnp.tile(psf, (model_frame.bbox.shape[0], 1, 1))
        else:
            psf_model = psf
        object.__setattr__(self, "_psf_model", psf_model)

    def __call__(self, model, key=None):
        # print(model)
        psf_model = self._psf_model
        # print(model.shape)
        # print(psf_model.shape)
        fft_shape = _get_fast_shape(model.shape, psf_model.shape, padding=1000, axes=(-2, -1))
        print()
        deconv_ = deconvolve(model, psf_model, axes=(-2, -1), fft_shape=fft_shape, return_fft=True)
        # print(deconv_.dtype)
        return deconv_

class KResampleRenderer(Renderer):
    """
    Perform resampling in Fourier space
    
    Should get as input a Fourier image sampled at the model frame resolution
    Needs to return a Fourier image sampled at the observation frame resolution
    """ 
    def __init__(self, model_frame, obs_frame): 
        object.__setattr__(self, "_in_res", model_frame.pixel_size)
        object.__setattr__(self, "_out_res", obs_frame.pixel_size)

        # find on what grid we will interpolate
        fft_out_shape = _get_fast_shape(obs_frame.bbox.shape, obs_frame.psf().shape, padding=300, axes=(-2, -1))
        object.__setattr__(self, "_fft_out_shape", fft_out_shape)

        # compute resolution ratio for flux conservation
        object.__setattr__(self, "_resolution_ratio", obs_frame.pixel_size/model_frame.pixel_size)


    def __call__(self, kimage, key=None):

        # compute model k-coordinates of the fourier input image
        ky_in = jnp.linspace(-.5, .5, kimage.shape[1]) / self._in_res
        # kx_in = jnp.linspace(0, .5, kimage.shape[1]//2+1) / self._in_res
        kx_in = jnp.linspace(-.5, .5, kimage.shape[1]) / self._in_res

        ky_out = jnp.linspace(-.5, .5, self._fft_out_shape[0]) / self._out_res
        # kx_out = jnp.linspace(0, .5, self._fft_out_shape[0]//2+1) / self._out_res
        kx_out = jnp.linspace(-.5, .5, self._fft_out_shape[0]) / self._out_res
        # print(kx_in)
        # print(kx_out)
        kimage = jnp.fft.fftshift(kimage, (-2, -1))

        kcoords_in = jnp.stack(
            jnp.meshgrid(kx_in, 
                         ky_in
                         ),
              -1
              )
        
        kcoords_out = jnp.stack(
            jnp.meshgrid(
                kx_out,
                ky_out
            ),
            -1
        )
        
        # import jii
        k_resampled = jax.vmap(
            resample2d,
            in_axes=(0, None, None, None))(
                kimage,
                kcoords_in,
                kcoords_out,
                3
        )                       
        
        k_resampled = jnp.fft.ifftshift(k_resampled, (-2, -1))

        # conserve flux
        
        k_resampled = k_resampled * self._resolution_ratio

        return k_resampled
    

class KConvolveRenderer(Renderer):
    """
    Convolve with obs PSF and return real image
    """
    def __init__(self, model_frame, obs_frame):
        object.__setattr__(self, "_obs_shape", obs_frame.bbox.shape)

        # get PSF from obs
        psf = obs_frame.psf()
        if len(psf.shape) == 2:  # only one image for all bands
            # print("I am tiling the psf")
            psf_model = jnp.tile(psf, (obs_frame.bbox.shape[0], 1, 1))
        else:
            psf_model = psf
        object.__setattr__(self, "_psf_model", psf_model)

        fft_out_shape = _get_fast_shape(obs_frame.bbox.shape, obs_frame.psf().shape, padding=300, axes=(-2, -1))
        object.__setattr__(self, "_fft_out_shape", fft_out_shape)

    def __call__(self, kimage, key=None):

        # Convolve with obs PSF
        # fft_shape = _get_fast_shape(model.shape, psf_model.shape, padding=3, axes=(-2, -1))

        kpsf = transform(self._psf_model, self._fft_out_shape, (-2, -1))
        # print("kimage.shape", kimage.shape)
        # print("kpsf.shape", kpsf.shape)
        kconv = kimage * kpsf

        img = inverse(kconv, self._fft_out_shape, self._obs_shape, axes=(-2,-1))

        return img