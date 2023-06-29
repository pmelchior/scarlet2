import jax.numpy as jnp

from .bbox import Box
from .frame import Frame
from .module import Module
from .renderer import Renderer, NoRenderer, ConvolutionRenderer


class Observation(Module):
    data: jnp.ndarray
    weights: jnp.ndarray
    frame: Frame
    renderer: Renderer

    def __init__(self, data, weights, psf=None, wcs=None, channels=None, renderer=None):
        self.data = jnp.asarray(data)
        self.weights = jnp.asarray(weights)
        if channels is None:
            channels = range(data.shape[0])
        self.frame = Frame(Box(data.shape), psf, wcs, channels)
        if renderer is None:
            renderer = NoRenderer()
        self.renderer = renderer

    def render(self, model):
        # render the model in the frame of the observation
        return self.renderer(model)

    def log_likelihood(self, model):
        # rendered model
        model_ = self.render(model)
        # normalization of the single-pixel likelihood:
        # 1 / [(2pi)^1/2 (sigma^2)^1/2]
        # with inverse variance weights: sigma^2 = 1/weight
        # full likelihood is sum over all (unmasked) pixels in data
        D = jnp.prod(jnp.asarray(self.data.shape)) - jnp.sum(self.weights == 0)
        log_norm = D / 2 * jnp.log(2 * jnp.pi)
        log_like = -jnp.sum(self.weights * (model_ - self.data) ** 2) / 2
        return log_like - log_norm

    def match(self, frame, renderer=None):
        # choose the renderer
        if renderer is None:
            if self.frame.psf is frame.psf:
                self.renderer = NoRenderer()
            else:
                assert self.frame.psf is not None and frame.psf is not None
                if self.frame.wcs is frame.wcs:
                    # same or None wcs: ConvolutionRenderer
                    renderer = ConvolutionRenderer(frame, self.frame)
                else:
                    raise NotImplementedError
                    # # if wcs shows changes in resolution or orientation:
                    # # use ResolutionRenderer
                    # assert self.frame.wcs is not None and model_frame.wcs is not None
                    # angle, h = interpolation.get_angles(self.wcs, model_frame.wc
                    # s)
                    # same_res = abs(h - 1) < np.finfo(float).eps
                    # same_rot = (np.abs(angle[1]) ** 2) < np.finfo(float).eps
                    # if same_res and same_rot:
                    #     renderer = ConvolutionRenderer(
                    #         self, model_frame, convolution_type="fft"
                    #     )
                    # else:
                    #     renderer = ResolutionRenderer(self, model_frame)
        else:
            assert isinstance(renderer, Renderer)
        object.__setattr__(self, 'renderer', renderer)
        return self
