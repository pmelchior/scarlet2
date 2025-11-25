import equinox as eqx
import jax
import jax.numpy as jnp

from .bbox import Box, overlap_slices
from .frame import Frame, get_affine
from .module import Module
from .renderer import (
    ChannelRenderer,
    ConvolutionRenderer,
    NoRenderer,
    Renderer,
    ResamplingRenderer,
    TrimSpatialBox,
)
from .validation_utils import (
    ValidationError,
    ValidationInfo,
    ValidationMethodCollector,
    ValidationResult,
    ValidationWarning,
    print_validation_results,
)


class Observation(Module):
    """Content and definition of an observation"""

    data: jnp.ndarray
    """Observed data"""
    weights: jnp.ndarray
    """Statistical weights (usually inverse variance) for :py:meth:`log_likelihood`"""
    frame: Frame
    """Metadata to describe what view of the sky `data` amounts to"""
    renderer: (Renderer, eqx.nn.Sequential)
    """Renderer to translate from the model frame the observation frame"""

    def __init__(self, data, weights, psf=None, wcs=None, channels=None, renderer=None):
        # TODO: automatic conversion to jnp arrays
        self.data = data
        if self.data.ndim == 2:
            # add a channel dimension if it is missing
            self.data = self.data[None, ...]

        self.weights = weights
        if self.weights is not None and self.weights.ndim == 2:
            # add a channel dimension if it is missing
            self.weights = self.weights[None, ...]

        self.frame = Frame(Box(self.data.shape), psf, wcs, channels=channels)
        if renderer is None:
            renderer = NoRenderer()
        self.renderer = renderer

        # (re)-import `VALIDATION_SWITCH` at runtime to avoid using a static/old value
        from .validation_utils import VALIDATION_SWITCH

        if VALIDATION_SWITCH:
            from .validation import check_observation

            validation_results = check_observation(self)
            print_validation_results("Observation validation results", validation_results)

    def render(self, model):
        """Render `model` in the frame of this observation

        Parameters
        ----------
        model: array
            The (pre-rendered) predicted data cube, typically from evaluating :py:class:`~scarlet2.Scene`

        Returns
        -------
        array
            Prediction of the observation given the `model`. Has the same shape as :py:attr:`data`.
        """
        return self.renderer(model)

    def log_likelihood(self, model):
        """The logarithm the likelihood of :py:attr:`data` given `model`

        Parameters
        ----------
        model: array
            The (pre-rendered) predicted data cube, typically from evaluating :py:class:`~scarlet2.Scene`

        Returns
        -------
        float
        """

        return self._log_likelihood(model, self.data)

    def _log_likelihood(self, model, data):
        # rendered model
        model_ = self.render(model)

        # normalization of the single-pixel likelihood:
        # 1 / [(2pi)^1/2 (sigma^2)^1/2]
        # with inverse variance weights: sigma^2 = 1/weight
        # full likelihood is sum over all (unmasked) pixels in data
        n = jnp.prod(jnp.asarray(data.shape)) - jnp.sum(self.weights == 0)
        log_norm = n / 2 * jnp.log(2 * jnp.pi)
        log_like = -jnp.sum(self.weights * (model_ - data) ** 2) / 2
        return log_like - log_norm

    def goodness_of_fit(self, model):
        """Evaluate the goodness of the model fit to the data

        For a Gaussian noise model, the gof is defined as the averaged squared deviation of the model from the
        data, scaled by the variance of the data, aka mean chi squared
        :math:`\frac{1}{N}\\sum_i=1^N w_i (m_i - d_i)^2` with inverse variance weights :math:`w_i`.

        Up to a normalization, the gof is identical to :py:class:`~scarlet2.Observation.log-likelihood`.

        Parameters
        ----------
        model: array
            The (pre-rendered) predicted data cube, typically from evaluating :py:class:`~scarlet2.Scene`

        Returns
        -------
        float
        """
        # only use unmasked pixels in the data
        n = jnp.prod(jnp.asarray(self.data.shape)) - jnp.sum(self.weights == 0)
        return (self.weights * (self.render(model) - self.data) ** 2).sum() / n

    def match(self, frame, renderer=None):
        """Construct the mapping between `frame` (from the model) and this observation frame

        Parameters
        ----------
        frame: Frame
            Model frame, typically :py:attr:`scarlet2.Scene.frame` for the current scene.
        renderer: Renderer, optional
            Custom transformation to translate the `frame` (from the model) to this observation frame.
            If not set, this method will attempt to create the mapping from the information in both frames.

        Returns
        -------
        self
        """
        # choose the renderer
        if renderer is None:
            renderers = []

            # note the order of renderers!
            # 1) match channels of frame
            if self.frame.channels != frame.channels:
                renderers.append(ChannelRenderer(frame, self.frame))

            # 2) match spatial properties of frame
            # if image has pixel grid (modulo an integer shift), avoid resampling
            m_self = get_affine(self.frame.wcs)
            m_frame = get_affine(frame.wcs)
            same_matrix = jnp.allclose(m_self, m_frame)

            ref_pixel = jnp.array(self.frame.bbox.spatial.origin)
            shift = frame.get_pixel(self.frame.get_sky_coord(ref_pixel)) - ref_pixel
            integer_shift = jnp.allclose(shift, jnp.round(shift), atol=1e-3)

            if same_matrix and integer_shift:
                if self.frame.psf != frame.psf:
                    renderers.append(ConvolutionRenderer(frame, self.frame))
                if self.frame.bbox.spatial != frame.bbox.spatial:
                    renderers.append(TrimSpatialBox(frame, self.frame))
            else:
                renderers.append(ResamplingRenderer(frame, self.frame))

            if len(renderers) == 0:
                renderer = NoRenderer()
            elif len(renderers) == 1:
                renderer = renderers[0]
            else:
                renderer = eqx.nn.Sequential(renderers)
        else:
            assert isinstance(renderer, (Renderer, eqx.nn.Sequential))
            # TODO: avoid call to renderer, use validator instead
            assert renderer(jnp.zeros(frame.bbox.shape)).shape == self.frame.bbox.shape, (
                "Renderer does not map model frame to observation frame"
            )
        object.__setattr__(self, "renderer", renderer)
        return self

    def eval_chi_square_in_box_and_border(self, scene, border_width=3):
        """
        Evaluate the weighted mean (weighted by the inverse variance weights) of the squared residuals
        for each source. Chi square is also computed for the perimeter outside the box of with `border_width`.

        Parameters
        ----------
        scene: :py:class:`~scarlet2.Scene`
            Scene containing the sources
        border_width: int
            width of the border around the source box

        Returns
        -------
        Dict of sources indices and their corresponding Dict of residuals inside and outside source box.
        """
        # TODO: combine with chi_square_in_box_and_border and move this to output validation tests (#148)
        residuals = self.render(scene()) - self.data

        chi_dict = {}
        for i, src in enumerate(scene.sources):
            bbox, _ = overlap_slices(self.frame.bbox, src.bbox, return_boxes=True)
            chi_in, chi_out = chi_square_in_box_and_border(residuals, self.weights, bbox, border_width)
            chi_dict[i] = {"in": chi_in, "out": chi_out}

        return chi_dict


class CorrelatedObservation(Observation):
    """Content and definition of an observation with pixel correlations

    The noise model is still assumed to be Gaussian, but with correlations between pixels.
    The implementation computes the goodness of fit in Fourier space from the noise power spectrum to avoid
    the expensive computation of/with an inverse banded matrix in configuration space.
    """

    power_spectrum: jnp.ndarray
    """Noise power spectrum for :py:meth:`log_likelihood`"""

    def __init__(
        self,
        data,
        weights,
        psf=None,
        wcs=None,
        channels=None,
        renderer=None,
        power_spectrum=None,
        correlation_function=None,
    ):
        super().__init__(data, weights, psf=psf, wcs=wcs, channels=channels, renderer=renderer)

        assert power_spectrum is not None or correlation_function is not None, (
            "Provide either power_spectrum or correlation_function"
        )

        if power_spectrum is None:

            def noise_kernel(xi):
                channels = len(xi[0, 0])
                maxlength = max(max(k) for k in xi)
                kernel = jnp.zeros((channels, 2 * maxlength + 1, 2 * maxlength + 1))
                for k in xi:
                    dy, dx = k
                    kernel = kernel.at[:, dy + maxlength, dx + maxlength].set(xi[k])
                return kernel

            def pad_kernel(kernel, shape):
                pads = ((0, 0),) + tuple(
                    ((s - l) // 2, (s - l) // 2 + 1)
                    for s, l in zip(shape[-2:], kernel.shape[-2:], strict=False)  # noqa: E741
                )
                kernel_padded = jnp.pad(kernel, pads)
                return kernel_padded

            def power_spectrum_from(xi, shape):
                # NOTE: this conversion is not ideal because the correlation function is likely undersampled
                # Better would be a pure correlated noise field to measure the power spectrum directly
                kernel = noise_kernel(xi)
                kernel_padded = pad_kernel(kernel, shape)
                kernel_fft = jnp.fft.rfft2(kernel_padded, axes=(-2, -1))
                ps = jnp.abs(kernel_fft)
                return ps

            power_spectrum = power_spectrum_from(correlation_function, data.shape)

        self.power_spectrum = power_spectrum

    def _log_likelihood(self, model, data):
        # rendered model
        model_ = self.render(model)

        # compute log_likelihood, without normalization factor
        res_fft = jnp.fft.rfft2(model_ - data, axes=(-2, -1))
        log_like = -0.5 * jnp.sum((res_fft * jnp.conjugate(res_fft)).real / self.power_spectrum)

        return log_like


def chi_square_in_box_and_border(residuals, weights, bbox, border_width):
    """
    helper function for :py:meth:`eval_chi_square_in_box_and_border`

    Parameters
    ----------
    residuals: array
        residual image
    weights: array
        observation weights (inverse variance)
    bbox: :py:class:`~scarlet2.Box
        source box`
    border_width: int
        width of the border around the source box
    """
    bbox_out = bbox.grow([0, border_width, border_width])

    sub_res_in = jax.lax.dynamic_slice(residuals, bbox.start, bbox.shape)
    sub_res_out = jax.lax.dynamic_slice(residuals, bbox_out.start, bbox_out.shape)
    weights_in = jax.lax.dynamic_slice(weights, bbox.start, bbox.shape)
    weights_out = jax.lax.dynamic_slice(weights, bbox_out.start, bbox_out.shape)

    border = jax.lax.dynamic_update_slice(
        jnp.ones_like(sub_res_out), jnp.zeros_like(sub_res_in), (0, 3, 3)
    ).astype("bool")

    chi_square_box = (weights_in * (sub_res_in**2)).mean()
    chi_square_border = (weights_out * (sub_res_out**2))[border].mean()

    return chi_square_box, chi_square_border


class ObservationValidator(metaclass=ValidationMethodCollector):
    """A class containing all of the validation checks for Observation objects.
    Note that the metaclass is defined as `MethodCollector`, which collects all
    validation methods in this class into a single class attribute list called
    `validation_checks`. This allows for easy iteration over all checks."""

    def __init__(self, observation: Observation):
        self.observation = observation

    def check_weights_non_negative(self) -> ValidationResult:
        """Check that the weights in the observation are non-negative.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        if self.observation.weights is not None and (self.observation.weights < 0).any():
            return ValidationError(
                message="Weights in the observation must be non-negative.",
                check=self.__class__.__name__,
                context={"observation.weights": self.observation.weights},
            )
        else:
            return ValidationInfo(
                message="Weights in the observation are non-negative.",
                check=self.__class__.__name__,
            )

    def check_weights_finite(self) -> ValidationResult:
        """Check that the weights in the observation are finite.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        if self.observation.weights is not None and jnp.isinf(self.observation.weights).any():
            return ValidationError(
                message="Weights in the observation must be finite.",
                check=self.__class__.__name__,
                context={"observation.weights": self.observation.weights},
            )
        else:
            return ValidationInfo(
                message="Weights in the observation are finite.",
                check=self.__class__.__name__,
            )

    def check_data_and_weights_shape(self) -> ValidationResult:
        """Check that the data and weights have the same shape.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        if (
            self.observation.weights is not None
            and self.observation.data.shape != self.observation.weights.shape
        ):
            return ValidationError(
                message="Data and weights must have the same shape.",
                check=self.__class__.__name__,
                context={
                    "observation.data.shape": self.observation.data.shape,
                    "observation.weights.shape": self.observation.weights.shape,
                },
            )
        else:
            return ValidationInfo(
                message="Data and weights have the same shape.",
                check=self.__class__.__name__,
            )

    def check_num_channels_matches_data(self) -> ValidationResult:
        """Check that the number of channels in the observation matches the data.

        NOTE: It is unlikely that this check will ever fail because there are many assertions
        in place around Frame and BBox that will raise an error if the number of channels
        does not match the data shape.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        num_channels = len(self.observation.frame.channels)
        if num_channels != self.observation.data.shape[0]:
            return ValidationError(
                message="Number of channels in the observation does not match the data.",
                check=self.__class__.__name__,
                context={
                    "observation.frame.channels": self.observation.frame.channels,
                    "observation.data.shape": self.observation.data.shape,
                },
            )
        else:
            return ValidationInfo(
                message="Number of channels in the observation matches the data.",
                check=self.__class__.__name__,
            )

    def check_data_finite_for_non_zero_weights(self) -> ValidationResult:
        """Check that the data in the observation is finite where weights are greater
        than zero.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        if self.observation.weights is not None and self.observation.data is not None:
            # Mask self.observation.data where self.observation.weights is 0
            if jnp.isinf(self.observation.data[self.observation.weights > 0]).any():
                return ValidationError(
                    message="Data in the observation must be finite.",
                    check=self.__class__.__name__,
                    context={"observation.data": self.observation.data},
                )
            else:
                return ValidationInfo(
                    message="Data in the observation is finite where weights are greater than zero.",
                    check=self.__class__.__name__,
                )
        else:
            return ValidationWarning(
                message="Observation data or weights are not defined.",
                check=self.__class__.__name__,
                context={
                    "observation.data": self.observation.data,
                    "observation.weights": self.observation.weights,
                },
            )

    def check_psf_has_3_dimensions(self) -> ValidationResult:
        """Check that the PSF in the observation is 3-dimensional.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        if self.observation.frame.psf is not None:
            psf = self.observation.frame.psf()
            if psf.ndim != 3:
                return ValidationError(
                    message="PSF must be 3-dimensional.",
                    check=self.__class__.__name__,
                    context={"observation.frame.psf.shape": psf.shape},
                )
            else:
                return ValidationInfo(
                    message="PSF is 3-dimensional.",
                    check=self.__class__.__name__,
                )
        else:
            return ValidationWarning(
                message="Observation PSF is not defined.",
                check=self.__class__.__name__,
                context={"observation.frame.psf": self.observation.frame.psf},
            )

    def check_number_of_psf_channels(self) -> ValidationResult:
        """Check that the number of PSF channels matches the number of data channels and
        that the PSF and data have the same number of dimensions. The PSF should be
        3-dimensional, and number of channels should match the data.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        return_value: ValidationResult = ValidationInfo(
            message="Number of PSF channels matches the number of data channels.",
            check=self.__class__.__name__,
        )
        if self.observation.frame.psf is not None:
            num_psf_channels = self.observation.frame.psf().shape[0]
            num_data_channels = self.observation.data.shape[0]

            # The number of bands is different between the PSF and data
            if num_psf_channels != num_data_channels:
                return_value = ValidationError(
                    message="Number of PSF channels does not match the number of data channels.",
                    check=self.__class__.__name__,
                    context={
                        "observation.frame.psf.shape": self.observation.frame.psf().shape,
                        "observation.data.shape": self.observation.data.shape,
                    },
                )

        else:
            return_value = ValidationWarning(
                message="Observation PSF is not defined.",
                check=self.__class__.__name__,
                context={"observation.frame.psf": self.observation.frame.psf},
            )

        return return_value

    def check_psf_centroid_consistent(self) -> ValidationResult:
        """Check that the pixel location of the PSF centroid is consistent across
        channels.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        return_value: ValidationResult = ValidationInfo(
            message="PSF centroid is consistent across channels.",
            check=self.__class__.__name__,
        )

        if self.observation.frame.psf is not None:
            from .measure import Moments

            psf_shape = self.observation.frame.psf().shape
            psf_center_y = psf_shape[-2] // 2
            psf_center_x = psf_shape[-1] // 2
            moments = Moments(self.observation.frame.psf(), N=1, center=[psf_center_y, psf_center_x])
            psf_centroid = moments.centroid

            psf_centroid_y, psf_centroid_x = psf_centroid

            tolerance = 1e-3
            if not jnp.allclose(psf_centroid_y, psf_centroid_y[0], atol=tolerance) or not jnp.allclose(
                psf_centroid_x, psf_centroid_x[0], atol=tolerance
            ):
                return_value = ValidationError(
                    message="PSF centroid is not the same in all channels.",
                    check=self.__class__.__name__,
                    context={
                        "psf_centroid": psf_centroid,
                    },
                )
        else:
            return_value = ValidationWarning(
                message="Observation PSF is not defined.",
                check=self.__class__.__name__,
                context={"observation.frame.psf": self.observation.frame.psf},
            )
        return return_value
