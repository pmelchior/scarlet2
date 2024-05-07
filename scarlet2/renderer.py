import equinox as eqx
import jax.numpy as jnp
import jax

from .fft import convolve, deconvolve, _get_fast_shape, transform, inverse, good_fft_size, _pad, _trim

from .interpolation import resample_ops, Quintic

from .fft import wrap_hermitian_x

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


class KDeconvRenderer(Renderer):
    """
    Perform decovolution in Fourier space and return the Fourier decovolved
    model
    """

    def __init__(self, model_frame, padding=None):

        if padding == None:
            # use 4 times padding
            padding = 4
        object.__setattr__(self, "_padding", padding)

        # create PSF model
        psf = model_frame.psf()
        if len(psf.shape) == 2:  # only one image for all bands
            psf_model = jnp.tile(psf, (model_frame.bbox.shape[0], 1, 1))
        else:
            psf_model = psf
        object.__setattr__(self, "_psf_model", psf_model)

    def __call__(self, model, key=None):
        psf_model = self._psf_model

        if self._padding==None:
            fft_shape = _get_fast_shape(
                model.shape, psf_model.shape, padding=self.padding, axes=(-2, -1)
            )
        else:
            fft_shape_ = max(good_fft_size(self._padding*max(model.shape)),
                             good_fft_size(self._padding*max(psf_model.shape)))
            
            fft_shape = [fft_shape_, fft_shape_]

        deconv_ = deconvolve(
            model, psf_model, axes=(-2, -1), fft_shape=fft_shape, return_fft=True
        )

        return deconv_


class KResampleRenderer(Renderer):
    """
    Perform resampling in Fourier space

    Should get as input a Fourier image sampled at the model frame resolution
    Needs to return a Fourier image sampled at the observation frame resolution
    """

    def __init__(self, model_frame, obs_frame, padding=None):
        object.__setattr__(self, "_in_res", model_frame.pixel_size)
        object.__setattr__(self, "_out_res", obs_frame.pixel_size)

        if padding == None:
            # use 4 times padding
            padding = 4

        # find on what grid we will interpolate
        # fft_out_shape = _get_fast_shape(
        #     obs_frame.bbox.shape, obs_frame.psf().shape, padding=padding, axes=(-2, -1)
        # )

        # getting the smallest grid to perform the interpolation
        # odd shape is required for k-wrapping later
        _fft_out_shape = min(good_fft_size(padding*max(obs_frame.bbox.shape)),
                             good_fft_size(padding*max(obs_frame.psf().shape))) + 1

        object.__setattr__(self, "_fft_out_shape", _fft_out_shape)

        # compute resolution ratio for flux conservation
        object.__setattr__(
            self, "_resolution_ratio", obs_frame.pixel_size / model_frame.pixel_size
        )

    def __call__(self, kimage, key=None):

        interpolant = Quintic()

        # compute model k-coordinates of the fourier input image
        # ky_in = jnp.linspace(-0.5, 0.5, kimage.shape[1]) / self._in_res
        # kx_in = jnp.linspace(0, 0.5, kimage.shape[1] // 2 + 1) / self._in_res

        # ky_out = jnp.linspace(-0.5, 0.5, self._fft_out_shape[0]) / self._out_res
        # kx_out = jnp.linspace(0, 0.5, self._fft_out_shape[0] // 2 + 1) / self._out_res
        shape_in = kimage.shape[1]
        shape_out = self._fft_out_shape

        kcoords_out = jnp.stack(jnp.meshgrid(
            jnp.linspace(0, 
                        shape_in/2/self._out_res*self._in_res, 
                         shape_out//2+1),
            jnp.linspace(-shape_in/2/self._out_res*self._in_res, 
                         shape_in/2/self._out_res*self._in_res, 
                         shape_out)
                         ), -1)

        kimage = jnp.fft.fftshift(kimage, -2)

        # kcoords_in = jnp.stack(jnp.meshgrid(kx_in, ky_in), -1)
        # kcoords_out = jnp.stack(jnp.meshgrid(kx_out, ky_out), -1)

        # k_resampled = jax.vmap(resample2d, in_axes=(0, None, None))(
        #     kimage, kcoords_in, kcoords_out
        # )

        k_resampled = jax.vmap(resample_hermitian, in_axes=(0, None, None, None,None))(
            kimage, kcoords_out, -shape_in/2, -shape_in/2, interpolant)
        
        print(k_resampled.shape)

        # multiply with the FFT of the kernel
        kx = jnp.linspace(0, jnp.pi, shape_out//2 + 1) * self._in_res/self._out_res
        ky = jnp.linspace(-jnp.pi, jnp.pi, shape_out)
        coords = jnp.stack(jnp.meshgrid(kx, ky),-1) / 2 / jnp.pi

        xint_val = interpolant.uval(coords[...,0]) * interpolant.uval(coords[...,1])
        print(xint_val.shape)
        k_resampled = k_resampled * xint_val

        k_resampled = jnp.fft.ifftshift(k_resampled, -2)

        return k_resampled


class KConvolveRenderer(Renderer):
    """
    Convolve with obs PSF and return real image
    """

    def __init__(self, obs_frame, padding=None):
        object.__setattr__(self, "_obs_shape", obs_frame.bbox.shape)

        if padding == None:
            # use 4 times padding
            padding = 4
            # padding = pad_factor * max(obs_frame.bbox.shape[1], obs_frame.bbox.shape[2])
        # object.__setattr__(self, "_pad_size", padding)
        object.__setattr__(self, "_padding", padding)

        # get PSF from obs
        psf = obs_frame.psf()
        if len(psf.shape) == 2:  # only one image for all bands
            psf_model = jnp.tile(psf, (obs_frame.bbox.shape[0], 1, 1))
        else:
            psf_model = psf
        object.__setattr__(self, "_psf_model", psf_model)


        # fft_out_shape = _get_fast_shape(
        #     obs_frame.bbox.shape, obs_frame.psf().shape, padding=padding, axes=(-2, -1)
        # )

        _fft_out_shape = good_fft_size(padding*max(obs_frame.psf().shape)) + 1
        fft_out_shape = [_fft_out_shape, _fft_out_shape]

        object.__setattr__(self, "_fft_out_shape", fft_out_shape)

    def __call__(self, kimage, key=None):

        # Convolve with obs PSF

        kpsf = transform(self._psf_model, self._fft_out_shape, (-2, -1))

        kconv = kimage * kpsf

        from jax_galsim.core.wrap_image import wrap_hermitian_x
        kconv = wrap_hermitian_x(
                            kconv,
                            -64,
                            -64,
                            -63,
                            -64,
                            128,
                            128
        )

        kconv = kconv[:,:-1, :]

        fft_out_shape = [self._fft_out_shape[0]-1, self._fft_out_shape[1]]
        print(kconv.shape)
        print(self._obs_shape)

        img = inverse(kconv, fft_out_shape, self._obs_shape, axes=(-2, -1))

        return img
    

"""
Preprocess:
    - padd img, psf_in and psf_out on the according goodfftsize
    - return kimages

Resample:
    - resample the three kimage on the target kgrid
    - return these images

Postprocess:
    - multiply these k images
    - kwrapping
    - ifft
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
            psf_model = jnp.tile(psf_model, (model_frame.bbox.shape[0], 1, 1))
        
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
    Perform the intepolation of the model, model psf and obs psf on the same
    target grid
    """

    def __init__(self, model_frame, obs_frame, padding=None):
        
        if padding == None:
            # use 4 times padding
            padding = 4
        object.__setattr__(self, "_padding", padding)

        fft_shape_model_im = good_fft_size(self._padding*max(model_frame.bbox.shape))
        fft_shape_model_psf = good_fft_size(self._padding*max(model_frame.psf().shape))
        fft_shape_obs_psf = good_fft_size(self._padding*max(obs_frame.psf().shape))

        # getting the smallest grid to perform the interpolation
        # odd shape is required for k-wrapping later
        fft_shape_target = min(fft_shape_model_im, fft_shape_model_psf, fft_shape_obs_psf) + 1
        object.__setattr__(self, "fft_shape_target", fft_shape_target)

        object.__setattr__(self, "res_in", model_frame.pixel_size)
        object.__setattr__(self, "res_out", obs_frame.pixel_size)

    def __call__(self, kimages, key=None):

        model_kim, model_kpsf, obs_kpsf = kimages

        model_kim_interp = resample_ops(model_kim, model_kim.shape[-2], 
                                        self.fft_shape_target, self.res_in, self.res_out)

        model_kpsf_interp = resample_ops(model_kpsf, model_kpsf.shape[-2], 
                                        self.fft_shape_target, self.res_in, self.res_out)
        
        obs_kpsf_interp = resample_ops(obs_kpsf, obs_kpsf.shape[-2], 
                                        self.fft_shape_target, self.res_out, self.res_out)
        
        return model_kim_interp, model_kpsf_interp, obs_kpsf_interp
    

class PostprocessMultiresRenderer(Renderer):
    def __init__(self):
        pass
    def __call__(self, kimages, key=None):

        model_kim, model_kpsf, obs_kpsf = kimages

        kimage_final = model_kim / model_kpsf * obs_kpsf
        
        kimage_final_wrap = jax.vmap(wrap_hermitian_x, in_axes=(0, None, None, None, None, None, None))(
                            kimage_final,
                            -64,
                            -64,
                            -63,
                            -64,
                            128,
                            128
        )

        kimage_final_wrap = kimage_final_wrap[:, :-1, :]
    
        kimg_shift = jnp.fft.ifftshift(kimage_final_wrap, axes=(-2,))

        real_image_arr = jnp.fft.fftshift(
            jnp.fft.irfft2(kimg_shift, 
                           [128, 128], (-2, -1)), (-2, -1)
        )

        img_trimed = _trim(real_image_arr, [real_image_arr.shape[0], 50,50])

        return img_trimed