from dataclasses import dataclass

import jax.numpy as jnp

from .bbox import Box
from .frame import Frame
from .renderer import Renderer, NoRenderer, ConvolutionRenderer


@dataclass
class Observation():
    data: jnp.ndarray  # = eqx.static_field()
    weights: jnp.ndarray  # = eqx.static_field()
    frame: Frame
    renderer: Renderer

    def __init__(self, data, weights, psf=None, wcs=None, channels=None, renderer=None):
        self.data = jnp.asarray(data)
        self.weights = jnp.asarray(weights)
        if channels is None:
            channels = range(data.shape[0])
        self.frame = Frame(Box(data.shape), psf, wcs, channels)
        self.renderer = renderer

    def render(self, model):
        # render the model in the frame of the observation
        # here: only convolution needed
        if self.renderer is None:
            return model
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
                assert isinstance(frame.psf.sigma(), float), "Model frame PSF needs to have single sigma value"
                if self.frame.wcs is frame.wcs:
                    # same or None wcs: ConvolutionRenderer
                    self.renderer = ConvolutionRenderer(frame, self.frame)
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
                    #     self.renderer = ConvolutionRenderer(
                    #         self, model_frame, convolution_type="fft"
                    #     )
                    # else:
                    #     self.renderer = ResolutionRenderer(self, model_frame)
        else:
            assert isinstance(renderer, Renderer)
            self.renderer = renderer

        return self
    
    # Matt addition, make __eq__ and __hash__ method for JIT compile to make obs hashable
    # def __eq__(self, other):
    #     # Equality Comparison between two objects
    #     return self.data == other.data and self.weights == other.weights and self.frame == other.frame and self.renderer == other.renderer 

    # def __hash__(self):
    #     # hash(custom_object)
    #     return hash((self.data))
