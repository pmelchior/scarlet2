import equinox as eqx
import jax.numpy as jnp
import jax

from .fft import convolve, deconvolve, _get_fast_shape, transform, good_fft_size, _trim, _pad

from .interpolation import resample_ops

from .fft import wrap_hermitian_x

from .measure import get_angle, get_sign

class Renderer(eqx.Module):
    def __call__(
        self, model, key=None
    ):  # key is needed to chain renderers with eqx.nn.Sequential
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
                    list(model_frame.channels).index(c)
                    for c in list(obs_frame.channels)
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
        fft_shape = _get_fast_shape(
            model_frame.bbox.shape, psf_model.shape, padding=3, axes=(-2, -1)
        )

        # compute and store diff kernel in Fourier space
        diff_kernel_fft = deconvolve(
            obs_frame.psf(),
            psf_model,
            axes=(-2, -1),
            fft_shape=fft_shape,
            return_fft=True,
        )
        object.__setattr__(self, "_diff_kernel_fft", diff_kernel_fft)

    def __call__(self, model, key=None):
        return convolve(model, self._diff_kernel_fft, axes=(-2, -1))

"""
Preprocess:
    - padd img, psf_in and psf_out on the according goodfftsize
    - return kimages

Resample:
    - resample the three kimages on the target kgrid
    - return these kimages

Postprocess:
    - Deconvolve model PSF and convolve obs PSF in Fourier space
    - kwrapping
    - ifft and cropping to obs frame
"""

class PreprocessMultiresRenderer(Renderer):
    """
    Perform padding and Fourier transform of the model image, model psf and
    observed psf
    """

    def __init__(self, model_frame, obs_frame, padding=None):

        if padding == None:
            # use 4 times padding
            padding = 4
        object.__setattr__(self, "_padding", padding)

        # create PSF model
        psf_model = model_frame.psf()
        
        if len(psf_model.shape) == 2:  # only one image for all bands
            psf_model = jnp.tile(psf_model, (obs_frame.bbox.shape[0], 1, 1))
        
        object.__setattr__(self, "_psf_model", psf_model)

        psf_obs = obs_frame.psf()

        object.__setattr__(self, "_psf_obs", psf_obs)

        fft_shape_model_im = good_fft_size(self._padding*max(model_frame.bbox.shape))
        fft_shape_model_psf = good_fft_size(self._padding*max(psf_model.shape))
        fft_shape_obs_psf = good_fft_size(self._padding*max(psf_obs.shape))

        object.__setattr__(self, "fft_shape_model_im", fft_shape_model_im)
        object.__setattr__(self, "fft_shape_model_psf", fft_shape_model_psf)
        object.__setattr__(self, "fft_shape_obs_psf", fft_shape_obs_psf)

    def __call__(self, model, key=None):

        psf_model = self._psf_model
        psf_obs = self._psf_obs

        model_kim = jnp.fft.fftshift(
            transform(model, (self.fft_shape_model_im, self.fft_shape_model_im), (-2, -1)),
            (-2))
        model_kpsf = jnp.fft.fftshift(
            transform(psf_model, (self.fft_shape_model_psf, self.fft_shape_model_psf), (-2, -1)),
            (-2))
        obs_kpsf = jnp.fft.fftshift(
            transform(psf_obs, (self.fft_shape_obs_psf, self.fft_shape_obs_psf), (-2, -1)),
            (-2))

        return model_kim, model_kpsf, obs_kpsf
    
class ResamplingMultiresRenderer(Renderer):
    """
    Perform the interpolation of the model, model psf and obs psf on the same
    target grid
    """

    def __init__(self, model_frame, obs_frame, padding=None):
        
        if padding == None:
            # use 4 times padding
            padding = 4
        object.__setattr__(self, "_padding", padding)

        fft_shape_model_im = good_fft_size(self._padding*max(model_frame.bbox.shape))
        fft_shape_obs_psf = good_fft_size(self._padding*max(obs_frame.psf().shape))

        # getting the smallest grid to perform the interpolation
        # odd shape is required for k-wrapping later
        fft_shape_target = min(fft_shape_model_im, fft_shape_obs_psf) + 1
        object.__setattr__(self, "fft_shape_target", fft_shape_target)

        object.__setattr__(self, "res_in", model_frame.pixel_size)
        object.__setattr__(self, "res_out", obs_frame.pixel_size)

        # Extract rotation angle between WCSs using jacobian matrices
        angle_in = get_angle(model_frame.wcs)
        angle_out = get_angle(obs_frame.wcs)
        if angle_out - angle_in == 0:
            object.__setattr__(self, "rotation_angle", None)
        else:
            object.__setattr__(self, "rotation_angle", angle_out - angle_in)
        
        # Get flip sign between WCSs using jacobian matrices
        sign_in = get_sign(model_frame.wcs)
        sign_out = get_sign(obs_frame.wcs)
        if (sign_in != sign_out).any():
            raise ValueError("model and observation WCSs have different sign conventions, which is not yet handled by scarlet2")

        object.__setattr__(self, "flip_sign", sign_in*sign_out)

    def __call__(self, kimages, key=None):

        model_kim, model_kpsf, obs_kpsf = kimages

        model_kim_interp = resample_ops(model_kim, model_kim.shape[-2], 
                                        self.fft_shape_target, self.res_in, self.res_out,
                                        phi=self.rotation_angle,
                                        flip_sign=self.flip_sign)

        model_kpsf_interp = resample_ops(model_kpsf, model_kpsf.shape[-2], 
                                        self.fft_shape_target, self.res_in, self.res_out,
                                        phi=self.rotation_angle,
                                        flip_sign=self.flip_sign)
        
        obs_kpsf_interp = resample_ops(obs_kpsf, obs_kpsf.shape[-2], 
                                        self.fft_shape_target, self.res_out, self.res_out)
        
        return model_kim_interp, model_kpsf_interp, obs_kpsf_interp
    

class PostprocessMultiresRenderer(Renderer):
    def __init__(self, model_frame, obs_frame, padding=None):
        if padding == None:
            # use 4 times padding
            padding = 4
        object.__setattr__(self, "_padding", padding)

        fft_shape_model_im = good_fft_size(self._padding*max(model_frame.bbox.shape))
        fft_shape_obs_psf = good_fft_size(self._padding*max(obs_frame.psf().shape))
        object.__setattr__(self, "real_shape_target", obs_frame.bbox.shape)
        
        # getting the smallest grid to perform the interpolation
        # odd shape is required for k-wrapping later
        fft_shape_target = min(fft_shape_model_im, fft_shape_obs_psf) + 1
        object.__setattr__(self, "fft_shape_target", fft_shape_target)

    def __call__(self, kimages, key=None):

        model_kim, model_kpsf, obs_kpsf = kimages
        kimage_final = model_kim / model_kpsf * obs_kpsf
        
        kimage_final_wrap = jax.vmap(wrap_hermitian_x, in_axes=(0, None, None, None, None, None, None))(
                            kimage_final,
                            -self.fft_shape_target//2,
                            -self.fft_shape_target//2,
                            -self.fft_shape_target//2+1,
                            -self.fft_shape_target//2,
                            self.fft_shape_target-1,
                            self.fft_shape_target-1
        )

        kimage_final_wrap = kimage_final_wrap[:, :-1, :]
    
        kimg_shift = jnp.fft.ifftshift(kimage_final_wrap, axes=(-2,))

        real_image_arr = jnp.fft.fftshift(
            jnp.fft.irfft2(kimg_shift, 
                           [self.fft_shape_target-1, self.fft_shape_target-1], (-2, -1)), (-2, -1)
        )

        img_trimed = _trim(real_image_arr, [real_image_arr.shape[0], self.real_shape_target[-2], self.real_shape_target[-1]])

        return img_trimed