import equinox as eqx
import jax.numpy as jnp

from .bbox import Box
from .frame import Frame
from .module import Module
from .renderer import (
    Renderer,
    NoRenderer,
    ConvolutionRenderer,
    ChannelRenderer,
    PreprocessMultiresRenderer,
    ResamplingMultiresRenderer,
    PostprocessMultiresRenderer
)


class Observation(Module):
    """Content and definition of an observation"""
    data: jnp.ndarray
    """Observed data"""
    weights: jnp.ndarray
    """Statistical weights (usually inverse variance) for :py:meth:`log_likelihood`"""
    frame: Frame = eqx.field(static=True)
    """Metadata to describe what view of the sky `data` amounts to"""
    # TODO: requires static, otherwise quickstart test aborts wiht "TypeError: unhashable type: 'slice'"
    renderer: (Renderer, eqx.nn.Sequential) = eqx.field(static=True)
    """Renderer to translate from the model frame the observation frame"""

    def __init__(self, data, weights, psf=None, wcs=None, channels=None, renderer=None):
        self.data = data
        self.weights = weights
        if channels is None:
            channels = range(data.shape[0])
        self.frame = Frame(Box(data.shape), psf, wcs, channels)
        if renderer is None:
            renderer = NoRenderer()
        self.renderer = renderer

    def render(self, model):
        """Render `model` in the frame of this observation

        Parameters
        ----------
        model: array
            The (pre-rendered) predicted data cube, typically from evaluateing :py:class:`~scarlet2.Scene`

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
            The (pre-rendered) predicted data cube, typically from evaluateing :py:class:`~scarlet2.Scene`

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
        D = jnp.prod(jnp.asarray(data.shape)) - jnp.sum(self.weights == 0)
        log_norm = D / 2 * jnp.log(2 * jnp.pi)
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

            if self.frame.psf != frame.psf:
                if frame.wcs != self.frame.wcs:
                    # 2) Pad model, model psf and obs psf and Fourier transform
                    renderers.append(PreprocessMultiresRenderer(frame, self.frame))

                    # 3)a) Resample at the obs resolution, deconvolve model PSF and
                    # convolve with obs PSF in Fourier space
                    renderers.append(ResamplingMultiresRenderer(frame, self.frame))

                    # 3)b) TODO: rotate and resample to obs orientation
                    # angle, h = interpolation.get_angles(self.wcs, frame.wcs)
                    # same_res = abs(h - 1) < np.finfo(float).eps
                    # same_rot = (np.abs(angle[1]) ** 2) < np.finfo(float).eps

                    # # convolve with obs PSF
                    # # TODO: if 2) is a resampling operation: model PSF needs to be resampled accordingly
                    # # Can be done by passing the renderer up to here to ConvolutionRenderer constructor below
                    # # Alternative: deconvolve from model_psf before 2) and convolve with full PSF in 3)
                    # # which is more modular but also more expensive unless all operations remain in Fourier space

                    # 4) Wrap the Fourier image and crop to obs frame
                    renderers.append(PostprocessMultiresRenderer(frame, self.frame))

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

    