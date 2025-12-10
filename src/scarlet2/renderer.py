"""Renderer classes"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from .bbox import Box, overlap_slices
from .fft import _get_fast_shape, _trim, _wrap_hermitian_x, convolve, deconvolve, good_fft_size, transform
from .frame import _minmax_int, get_relative_jacobian_shift, get_scale_angle_flip_shift
from .interpolation import Interpolant, Lanczos, resample3d, resample_fourier


class Renderer(eqx.Module):
    """Renderer base class

    Renderers are (potentially parameterized) transformations between the model
    frame and the observation frame, or elements of such a transformation.
    """

    def __call__(self, model, key=None):  # key is needed to chain renderers with eqx.nn.Sequential
        """What to run when Renderer is called"""
        raise NotImplementedError


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
    _fft_shape: jnp.array = eqx.field(repr=False)

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
        self._fft_shape = _get_fast_shape(model_frame.bbox.shape, psf_model.shape, padding=3, axes=(-2, -1))

        # compute and store diff kernel in Fourier space
        diff_kernel_fft = deconvolve(
            obs_frame.psf(),
            psf_model,
            axes=(-2, -1),
            fft_shape=self._fft_shape,
            return_fft=True,
        )
        self._diff_kernel_fft = diff_kernel_fft

    def __call__(self, model, key=None):
        """What to run when ConvolutionRenderer is called"""
        return convolve(model, self._diff_kernel_fft, axes=(-2, -1), fft_shape=self._fft_shape)


class TrimSpatialBox(Renderer):
    """Extract cutout the observation box from the model frame box"""

    slices: HashableSlice

    def __init__(self, model_frame, obs_frame):
        obs_coord = obs_frame.convert_pixel_to(model_frame)
        y_min = jnp.floor(jnp.min(obs_coord[:, 0]))
        x_min = jnp.floor(jnp.min(obs_coord[:, 1]))
        y_max = jnp.ceil(jnp.max(obs_coord[:, 0]))
        x_max = jnp.ceil(jnp.max(obs_coord[:, 1]))
        this_box = Box.from_bounds(
            # (int(y_min) + 1, int(y_max) + 1), (int(x_min) + 1, int(x_max) + 1)
            (int(y_min), int(y_max) + 1),
            (int(x_min), int(x_max) + 1),
        )

        im_slices, sub_slice = overlap_slices(model_frame.bbox.spatial, this_box)
        self.slices = (
            HashableSlice.from_slice(im_slices[-2]),  # height
            HashableSlice.from_slice(im_slices[-1]),  # width
        )

    def __call__(self, model, key=None):
        """What to run when TrimSpatialBox is called"""
        sub = model[:, self.slices[-2].get_slice(), self.slices[-1].get_slice()]
        return sub


class ResamplingRenderer(Renderer):
    """Renderer to resample image to different pixel grid (subpixel position, resolution, orientation)"""

    padding: int
    scale: float
    angle: float
    handedness: int
    shift: tuple
    has_psf_in: bool
    has_psf_out: bool
    fft_shape_target: int = eqx.field(repr=False)
    fft_shape_model_im: int = eqx.field(repr=False)
    jacobian: jnp.array = eqx.field(repr=False)
    model_kpsf_interp: jnp.array = eqx.field(repr=False, default=None)
    obs_kpsf_interp: jnp.array = eqx.field(repr=False, default=None)
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

        # TODO: Check for SIP distortions, which are not covered by this code!
        # If those exists:
        # 1) Use ConvolutionRenderer in model frame (obs PSF needs to be resampled to this frame)
        # 2) Apply Lanczos resampling to observed frame
        #
        # This should be much more flexible than the Kspace resampler and more accurate than
        # resampling to obs frame, followed by a convolution in obs frame because the difference
        # kernel would be expressed in obs pixel and can thus easily undersample the model PSF.

        # store linear transformation and shift
        self.jacobian, self.shift = get_relative_jacobian_shift(model_frame, obs_frame)
        # store these properties for convenience and printing
        # (ignore shift because it doesn't include CRPIX/CRVAL changes)
        self.scale, self.angle, self.handedness, _ = get_scale_angle_flip_shift(self.jacobian)

        # Get maximum of the fft shapes to interpolate on the highest resolved FFT image
        self.real_shape_target = obs_frame.bbox.shape
        self.fft_shape_model_im = good_fft_size(padding * max(model_frame.bbox.spatial.shape))
        self.fft_shape_target = self.fft_shape_model_im
        if obs_frame.psf is not None:
            fft_shape_obs_psf = good_fft_size(padding * max(obs_frame.psf.shape))
            self.fft_shape_target = max(self.fft_shape_model_im, fft_shape_obs_psf)
        # odd shape is required for k-wrapping later
        if self.fft_shape_target % 2 == 0:
            self.fft_shape_target += 1

        # PSF models in Fourier space
        if model_frame.psf is None:
            self.has_psf_in = False
        else:
            self.has_psf_in = True
            psf_model = model_frame.psf()
            if len(psf_model.shape) == 2:  # only one image for all bands
                psf_model = jnp.tile(psf_model, (obs_frame.bbox.shape[0], 1, 1))

            # Fourier transform model PSF
            fft_shape_model_psf = good_fft_size(padding * max(psf_model.shape))
            model_kpsf = jnp.fft.fftshift(
                transform(psf_model, (fft_shape_model_psf, fft_shape_model_psf), (-2, -1)), (-2)
            )
            # resample with warp
            self.model_kpsf_interp = resample_fourier(
                model_kpsf,
                model_kpsf.shape[-2],
                self.fft_shape_target,
                jacobian=self.jacobian,
            )

        if obs_frame.psf is None:
            self.has_psf_out = False
        else:
            self.has_psf_out = True
            psf_obs = obs_frame.psf()
            if len(psf_obs.shape) == 2:
                psf_obs = psf_obs[None, ...]

            obs_kpsf = jnp.fft.fftshift(
                transform(psf_obs, (fft_shape_obs_psf, fft_shape_obs_psf), (-2, -1)), (-2)
            )
            # resample without warp
            self.obs_kpsf_interp = resample_fourier(
                obs_kpsf,
                obs_kpsf.shape[-2],
                self.fft_shape_target,
            )

    def __call__(self, model, key=None):
        """What to run when ResamplingRenderer is called"""
        # Fourier transform model
        model_kim = jnp.fft.fftshift(
            transform(model, (self.fft_shape_model_im, self.fft_shape_model_im), (-2, -1)), (-2)
        )

        # resample on target grid
        model_kim_interp = resample_fourier(
            model_kim,
            model_kim.shape[-2],
            self.fft_shape_target,
            jacobian=self.jacobian,
            shift=self.shift,
        )

        # deconvolve with model psf, re-convolve with observation psf and Fourier transform back to real space
        kimage_final = model_kim_interp
        if self.has_psf_in:
            kimage_final /= self.model_kpsf_interp
        if self.has_psf_out:
            kimage_final *= self.obs_kpsf_interp

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


class LanczosResamplingRenderer(Renderer):
    """Renderer to resample image to different pixel grid with a Lanczos kernel."""

    interpolant: Interpolant
    scale: float
    angle: float
    handedness: int
    shift: tuple
    _coords: jnp.ndarray = eqx.field(repr=False)
    _warp: jnp.ndarray = eqx.field(repr=False)
    _diff_kernel_fft: jnp.array = eqx.field(repr=False, default=None)
    _fft_shape: int = eqx.field(repr=False, default=None)

    def __init__(self, model_frame, obs_frame, lanczos_order=5):
        self.interpolant = Lanczos(lanczos_order)
        model_shape = model_frame.bbox.spatial.shape
        self._coords = jnp.stack(
            jnp.meshgrid(jnp.arange(model_shape[0]), jnp.arange(model_shape[1])), -1
        ).astype(jnp.float32)  # x/y
        obs_shape = obs_frame.bbox.spatial.shape
        self._warp = obs_frame.convert_pixel_to(model_frame).reshape(obs_shape[0], obs_shape[1], 2)

        # linear transformation and shift between frames
        jacobian, self.shift = get_relative_jacobian_shift(model_frame, obs_frame)
        # store these properties for convenience and printing
        # (ignore shift because it doesn't include CRPIX/CRVAL changes)
        self.scale, self.angle, self.handedness, _ = get_scale_angle_flip_shift(jacobian)

        if model_frame.psf is not None and model_frame.wcs is not None:
            # construct diff kernel in model_space
            # create PSF model
            psf = model_frame.psf()
            psf_model = jnp.tile(psf, (obs_frame.bbox.shape[0], 1, 1)) if len(psf.shape) == 2 else psf

            # resample obs psf in model pixel
            psf_obs = obs_frame.psf()
            # TODO: what is different between indices and meshgrid???
            # coords_ = jnp.stack(jnp.indices(obs_psf.shape[-2:]), axis=-1).astype(jnp.float32)
            coords_ = jnp.stack(
                jnp.meshgrid(jnp.arange(psf_obs.shape[-2]), jnp.arange(psf_obs.shape[-1])), -1
            ).astype(jnp.float32)
            coords_in_model_space = obs_frame.convert_pixel_to(model_frame, pixel=coords_)
            ylims = _minmax_int(coords_in_model_space[..., 0])
            xlims = _minmax_int(coords_in_model_space[..., 1])
            warp_ = jnp.stack(
                jnp.meshgrid(
                    jnp.arange(ylims[0] - 1, ylims[1] - 1),
                    jnp.arange(xlims[0], xlims[1]),
                ),
                -1,
            ).astype(jnp.float32)
            warp__ = model_frame.convert_pixel_to(obs_frame, pixel=warp_).reshape(
                warp_.shape[0], warp_.shape[1], 2
            )
            # interpolate observed to model pixels
            psf_obs_interp = resample3d(psf_obs, coords=coords_, warp=warp__, interpolant=self.interpolant)

            # make sure fft uses a shape large enough to cover the convolved model
            padding = self.interpolant.extent
            self._fft_shape = good_fft_size(max(max(psf_obs_interp.shape[-2:]), max(model_shape)) + padding)

            # compute and store diff kernel in Fourier space
            self._diff_kernel_fft = deconvolve(
                psf_obs_interp,
                psf_model,
                axes=(-2, -1),
                fft_shape=(self._fft_shape, self._fft_shape),
                return_fft=True,
            )

    def __call__(self, model, key=None, warp=None):
        """What to run when renderer is called"""
        if warp is None:
            warp = self._warp
        _resample3d = partial(
            resample3d,
            coords=self._coords,
            warp=warp,
            interpolant=self.interpolant,
        )
        if self._diff_kernel_fft is not None:
            model_ = convolve(
                model, self._diff_kernel_fft, axes=(-2, -1), fft_shape=(self._fft_shape, self._fft_shape)
            )
        else:
            model_ = model

        return _resample3d(model_) / self.scale**2  # conservation of surface brightness / photons
