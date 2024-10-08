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
    data: jnp.ndarray = eqx.field(static=True)
    weights: jnp.ndarray = eqx.field(static=True)
    frame: Frame = eqx.field(static=True)
    renderer: (Renderer, eqx.nn.Sequential) = eqx.field(static=True)

    def __init__(self, data, weights, psf=None, wcs=None, channels=None, renderer=None):
        # TODO: replace by DataStore class, and make that static
        self.data = data
        self.weights = weights
        if channels is None:
            channels = range(data.shape[0])
        self.frame = Frame(Box(data.shape), psf, wcs, channels)
        if renderer is None:
            renderer = NoRenderer()
        self.renderer = renderer

    @property
    def C(self):
        return self.frame.C
    
    def render(self, model):
        # render the model in the frame of the observation
        return self.renderer(model)

    def log_likelihood(self, model):
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

    