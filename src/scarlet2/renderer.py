"""Renderer classes"""

import equinox as eqx
import jax
import jax.numpy as jnp

from .bbox import Box, overlap_slices
from .fft import _get_fast_shape, _trim, _wrap_hermitian_x, convolve, deconvolve, good_fft_size, transform
from .interpolation import resample_ops
from .measure import get_angle, get_sign


class Renderer(eqx.Module):
    """Renderer base class

    Renderers are (potentially parameterized) transformations between the model
    frame and the observation frame, or elements of such a transformation.
    """

    def __call__(self, model, key=None):  # key is needed to chain renderers with eqx.nn.Sequential
        """What to run when Renderer is called"""
        raise NotImplementedError


class NoRenderer(Renderer):
    """Inactive renderer that does not change the model"""

    def __call__(self, model, key=None):
        """Just return the model as is"""
        return model


#
class HashableSlice(eqx.Module):
    """A slice version that is hashable (for python < 3.12)"""

    start: int
    stop: int
    step: int = eqx.field(default=None)

    @classmethod
    def from_slice(cls, slice):
        """Create HashableSlice from slice"""
        return cls(slice.start, slice.stop, slice.step)

    def get_slice(self):
        """Return standard python slice"""
        return slice(self.start, self.stop, self.step)


class ChannelRenderer(Renderer):
    """Map model to observed channels

    This renderer only affects to spectral dimension of the model. It needs to
    be combined with spatial renderers for a full transformation to the observed frame.
    """

    channel_map: (None, list, HashableSlice, jnp.array) = None
    """Lookup table or transformation matrix

    For every channel in the observed frame, this map contained the index or
    weights of the model channels.
    """

    def __init__(self, model_frame, obs_frame):
        """Initialize channel mapping

        This method will attempt to find the index in `model_frame.channels` for
        every item `obs_frame.channels`. For this to work, the identifiers of the
        channels need to be the same, e.g. `channels=['g','r','i']` or
        `channels=[0,1,2,3,4]`.

        Parameters
        ----------
        model_frame: :py:class:`~scarlet.Frame`
            The model frame to be resampled
        obs_frame: :py:class:`~scarlet.Frame`
            The observation frame to which the model frame is resampled

        Raises
        ------
        ValueError
            If observed channel(s) are not found in `model_frame`
        """
        if obs_frame.channels == model_frame.channels:
            channel_map = None
        else:
            try:
                channel_map = [list(model_frame.channels).index(c) for c in list(obs_frame.channels)]
            except ValueError as err:
                msg = "Cannot match channels between model and observation.\n"
                msg += f"Got {model_frame.channels} and {obs_frame.channels}."
                raise ValueError(msg) from err

            min_channel = min(channel_map)
            max_channel = max(channel_map)
            if max_channel + 1 - min_channel == len(channel_map):
                channel_map = HashableSlice(min_channel, max_channel + 1)
        self.channel_map = channel_map

    def __call__(self, model, key=None):
        """Map model channels onto the observation channels

        Parameters
        ----------
        model: array
            The hyperspectral model
        key: optional
            Key is needed to chain renderers with eqx.nn.Sequential
        Returns
        -------
        obs_model: array
            `model` mapped onto the observation channels
        """
        if self.channel_map is None:
            return model
        if isinstance(self.channel_map, HashableSlice):
            return model[self.channel_map.get_slice(), :, :]
        if isinstance(self.channel_map, list):
            return model[self.channel_map, :, :]
        # not yet used by any renderer: full matrix mapping between model and observation channels
        return jnp.dot(self.channel_map, model)


class ConvolutionRenderer(Renderer):
    """Convolve model with observed PSF

    The convolution is performed in Fourier space and applies the difference kernel
    between model PSF and observed PSF.
    """

    _diff_kernel_fft: jnp.array = eqx.field(repr=False)

    def __init__(self, model_frame, obs_frame):
        """Initialize convolution renderer with difference kernel between `model_frame` and `obs_frame`

        Parameters
        ----------
        model_frame: :py:class:`~scarlet.Frame`
            The model frame to be resampled
        obs_frame: :py:class:`~scarlet2.Frame`
            The observation frame to which the model frame is resampled
        """
        # create PSF model
        psf = model_frame.psf()
        psf_model = jnp.tile(psf, (obs_frame.bbox.shape[0], 1, 1)) if len(psf.shape) == 2 else psf

        # make sure fft uses a shape large enough to cover the convolved model
        fft_shape = _get_fast_shape(model_frame.bbox.shape, psf_model.shape, padding=3, axes=(-2, -1))

        # compute and store diff kernel in Fourier space
        diff_kernel_fft = deconvolve(
            obs_frame.psf(),
            psf_model,
            axes=(-2, -1),
            fft_shape=fft_shape,
            return_fft=True,
        )
        self._diff_kernel_fft = diff_kernel_fft

    def __call__(self, model, key=None):
        """What to run when ConvolutionRenderer is called"""
        return convolve(model, self._diff_kernel_fft, axes=(-2, -1))


class AdjustToFrame(Renderer):
    """Extract cutout the observation box from the model frame box"""

    slices: HashableSlice

    def __init__(self, model_frame, obs_frame):
        obs_coord = obs_frame.convert_pixel_to(model_frame)
        y_min = jnp.floor(jnp.min(obs_coord[:, 0]))
        x_min = jnp.floor(jnp.min(obs_coord[:, 1]))
        y_max = jnp.ceil(jnp.max(obs_coord[:, 0]))
        x_max = jnp.ceil(jnp.max(obs_coord[:, 1]))
        num_channels = obs_frame.bbox.shape[0]
        this_box = Box.from_bounds(
            (0, num_channels), (int(y_min) + 1, int(y_max) + 1), (int(x_min) + 1, int(x_max) + 1)
        )

        im_slices, sub_slice = overlap_slices(model_frame.bbox, this_box)
        self.slices = (
            HashableSlice.from_slice(im_slices[0]),  # channels
            HashableSlice.from_slice(im_slices[1]),  # height
            HashableSlice.from_slice(im_slices[2]),  # width
        )

    def __call__(self, model, key=None):
        """What to run when AdjustToFrame is called"""
        sub = model[self.slices[0].get_slice(), self.slices[1].get_slice(), self.slices[2].get_slice()]
        return sub


class ResamplingRenderer(Renderer):
    """Renderer to resample image to different image placing or resolution

    The renderer comprises three steps

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

    padding: int
    fft_shape_target: int = eqx.field(repr=False)
    fft_shape_model_im: int = eqx.field(repr=False)
    res_in: float
    res_out: float
    rotation_angle: (None, float)
    flip_sign: jnp.array
    model_kpsf_interp: jnp.array = eqx.field(repr=False)
    obs_kpsf_interp: jnp.array = eqx.field(repr=False)
    real_shape_target: tuple = eqx.field(repr=False)

    def __init__(self, model_frame, obs_frame, padding=4):
        """Initialize preprocess renderer in multi-resolution mapping

        Parameters
        ----------
        model_frame: :py:class:`~scarlet2.Frame`
            The model frame to be resampled
        obs_frame: :py:class:`~scarlet2.Frame`
            The observation frame to which the model frame is resampled
        padding: int, optional
            How many times to input image if padded to reduce FFT artifacts.
        """
        self.padding = padding

        # create PSF model
        psf_model = model_frame.psf()
        if len(psf_model.shape) == 2:
            psf_model = psf_model[None, ...]

        if len(psf_model.shape) == 2:  # only one image for all bands
            psf_model = jnp.tile(psf_model, (obs_frame.bbox.shape[0], 1, 1))

        psf_obs = obs_frame.psf()
        if len(psf_obs.shape) == 2:
            psf_obs = psf_obs[None, ...]

        self.fft_shape_model_im = good_fft_size(padding * max(model_frame.bbox.shape))
        fft_shape_model_psf = good_fft_size(padding * max(psf_model.shape))
        fft_shape_obs_psf = good_fft_size(padding * max(psf_obs.shape))

        # Fourier transform model and observation PSFs
        model_kpsf = jnp.fft.fftshift(
            transform(psf_model, (fft_shape_model_psf, fft_shape_model_psf), (-2, -1)), (-2)
        )
        obs_kpsf = jnp.fft.fftshift(
            transform(psf_obs, (fft_shape_obs_psf, fft_shape_obs_psf), (-2, -1)), (-2)
        )

        # Get maximum of the fft shapes to interpolate on the highest resolved FFT image
        # odd shape is required for k-wrapping later
        self.fft_shape_target = max(self.fft_shape_model_im, fft_shape_obs_psf) + 1
        self.res_in = model_frame.pixel_size
        self.res_out = obs_frame.pixel_size

        # Extract rotation angle between WCSs using jacobian matrices
        angle_in = get_angle(model_frame.wcs)
        angle_out = get_angle(obs_frame.wcs)
        if angle_out - angle_in == 0:
            self.rotation_angle = None
        else:
            self.rotation_angle = angle_out - angle_in

        # Get flip sign between WCSs using jacobian matrices
        sign_in = get_sign(model_frame.wcs)
        sign_out = get_sign(obs_frame.wcs)
        if (sign_in != sign_out).any():
            raise ValueError(
                "model and observation WCSs have different sign conventions, \
                    which is not yet handled by scarlet2"
            )

        self.flip_sign = sign_in * sign_out

        self.model_kpsf_interp = resample_ops(
            model_kpsf,
            model_kpsf.shape[-2],
            self.fft_shape_target,
            self.res_in,
            self.res_out,
            phi=self.rotation_angle,
            flip_sign=self.flip_sign,
        )

        self.obs_kpsf_interp = resample_ops(
            obs_kpsf, obs_kpsf.shape[-2], self.fft_shape_target, self.res_out, self.res_out
        )

        self.real_shape_target = obs_frame.bbox.shape

    def __call__(self, model, key=None):
        """What to run when MultiresolutionRenderer is called"""
        # Fourier transform model
        model_kim = jnp.fft.fftshift(
            transform(model, (self.fft_shape_model_im, self.fft_shape_model_im), (-2, -1)), (-2)
        )

        # resample on target grid
        model_kim_interp = resample_ops(
            model_kim,
            model_kim.shape[-2],
            self.fft_shape_target,
            self.res_in,
            self.res_out,
            phi=self.rotation_angle,
            flip_sign=self.flip_sign,
        )

        # deconvolve with model psf, re-convolve with observation psf and Fourier transform back to real space
        model_kim = model_kim_interp
        model_kpsf = self.model_kpsf_interp
        obs_kpsf = self.obs_kpsf_interp

        kimage_final = model_kim / model_kpsf * obs_kpsf

        kimage_final_wrap = jax.vmap(_wrap_hermitian_x, in_axes=(0, None, None, None, None, None, None))(
            kimage_final,
            -self.fft_shape_target // 2,
            -self.fft_shape_target // 2,
            -self.fft_shape_target // 2 + 1,
            -self.fft_shape_target // 2,
            self.fft_shape_target - 1,
            self.fft_shape_target - 1,
        )

        kimage_final_wrap = kimage_final_wrap[:, :-1, :]

        kimg_shift = jnp.fft.ifftshift(kimage_final_wrap, axes=(-2,))

        real_image_arr = jnp.fft.fftshift(
            jnp.fft.irfft2(kimg_shift, [self.fft_shape_target - 1, self.fft_shape_target - 1], (-2, -1)),
            (-2, -1),
        )

        img_trimed = _trim(
            real_image_arr, [real_image_arr.shape[0], self.real_shape_target[-2], self.real_shape_target[-1]]
        )

        return img_trimed
