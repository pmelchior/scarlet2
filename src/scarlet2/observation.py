from typing import Optional

import equinox as eqx
import jax.numpy as jnp

from .bbox import Box
from .frame import Frame
from .module import Module
from .renderer import (
    AdjustToFrame,
    ChannelRenderer,
    ConvolutionRenderer,
    MultiresolutionRenderer,
    NoRenderer,
    Renderer,
)
from .validation import ValidationError, ValidationMethodCollector


class Observation(Module):
    """Content and definition of an observation"""

    data: jnp.ndarray
    """Observed data"""
    weights: jnp.ndarray
    """Statistical weights (usually inverse variance) for :py:meth:`log_likelihood`"""
    frame: Frame
    """Metadata to describe what view of the sky `data` amounts to"""
    renderer: (Renderer, eqx.nn.Sequential) = eqx.field(static=True)
    """Renderer to translate from the model frame the observation frame"""

    def __init__(
        self, data, weights, psf=None, wcs=None, channels=None, renderer=None, check_observation=False
    ):
        self.data = data
        self.weights = weights
        if channels is None:
            channels = range(data.shape[0])
        self.frame = Frame(Box(data.shape), psf, wcs, channels)
        if renderer is None:
            renderer = NoRenderer()
        self.renderer = renderer

        if check_observation:
            from .validation import check_observation

            validation_errors = check_observation(self)
            if validation_errors:
                #! We can raise this as a ValueError or assign it to self.validation_errors
                raise ValueError(
                    "Observation validation failed with the following errors:\n"
                    + "\n".join(str(error) for error in validation_errors)
                )

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
        d = jnp.prod(jnp.asarray(data.shape)) - jnp.sum(self.weights == 0)
        log_norm = d / 2 * jnp.log(2 * jnp.pi)
        log_like = -jnp.sum(self.weights * (model_ - data) ** 2) / 2
        return log_like - log_norm

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
            # 1) collapse channels that are not needed
            if self.frame.channels != frame.channels:
                renderers.append(ChannelRenderer(frame, self.frame))

            if self.frame.bbox != frame.bbox:
                renderers.append(AdjustToFrame(frame, self.frame))

            if self.frame.psf != frame.psf:
                if frame.wcs != self.frame.wcs:
                    # 2) Pad model, model psf and obs psf and Fourier transform
                    # 3)a) rotate and resample to obs orientation
                    # 3)b) Resample at the obs resolution
                    # 3)c) deconvolve with model PSF and re-convolve with obs PSF
                    # 4) Wrap the Fourier image and crop to obs frame
                    renderers.append(MultiresolutionRenderer(frame, self.frame))

                else:
                    renderers.append(ConvolutionRenderer(frame, self.frame))

            if len(renderers) == 0:
                renderer = NoRenderer()
            elif len(renderers) == 1:
                renderer = renderers[0]
            else:
                renderer = eqx.nn.Sequential(renderers)
        else:
            assert isinstance(renderer, (Renderer, eqx.nn.Sequential))
            assert (
                renderer(jnp.zeros(frame.bbox.shape)).shape == self.frame.bbox.shape
            ), "Renderer does not map model frame to observation frame"
        object.__setattr__(self, "renderer", renderer)
        return self


class ObservationValidator(metaclass=ValidationMethodCollector):
    """A class containing all of the validation checks for Observation objects.
    Note that the metaclass is defined as `MethodCollector`, which collects all
    validation methods in this class into a single class attribute list called
    `validation_checks`. This allows for easy iteration over all checks."""

    def __init__(self, observation: Observation):
        self.observation = observation

    def check_weights_non_negative(self) -> Optional[ValidationError]:
        """Check that the weights in the observation are non-negative.

        Returns
        -------
        ValidationError or None
            Returns a ValidationError if the check fails, otherwise None.
        """
        if (self.observation.weights < 0).any():
            return ValidationError(
                "Weights in the observation must be non-negative.",
                check=self.__class__.__name__,
                #! Placeholder for a meaningful context
                context={"observation.weights": self.observation.weights},
            )
        return None

    def check_weights_finite(self) -> Optional[ValidationError]:
        """Check that the weights in the observation are finite.

        Returns
        -------
        ValidationError or None
            Returns a ValidationError if the check fails, otherwise None.
        """
        if jnp.isinf(self.observation.weights).any():
            return ValidationError(
                "Weights in the observation must be finite.",
                check=self.__class__.__name__,
                #! Placeholder for a meaningful context
                context={"observation.weights": self.observation.weights},
            )
        return None
