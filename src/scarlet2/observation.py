import equinox as eqx
import jax
import jax.numpy as jnp

from .bbox import Box, overlap_slices
from .frame import Frame
from .measure import Moments
from .module import Module
from .renderer import (
    AdjustToFrame,
    ChannelRenderer,
    ConvolutionRenderer,
    MultiresolutionRenderer,
    NoRenderer,
    Renderer,
)


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

    def eval_chi_square(self, model):
        """Evaluate the weighted mean (weighted by the inverse variance weights) of the squared residuals

        Parameters
        ----------
        model: array
            The (pre-rendered) predicted data cube, typically from evaluating :py:class:`~scarlet2.Scene`

        Returns
        -------
        """
        print("Chi^2", (self.weights * (self.render(model) - self.data) ** 2).mean())

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
        """
        residuals = self.render(scene()) - self.data

        for i, src in enumerate(scene.sources):
            bbox, _ = overlap_slices(self.frame.bbox, src.bbox, return_boxes=True)
            chi_in, chi_out = chi_square_in_box_and_border(residuals, self.weights, bbox, border_width)
            print(
                f"Source {i},",
                "Chi^2 source box:",
                chi_in,
                "Chi^2 box border:",
                chi_out,
                "Missing flux:",
                chi_out > 1.5 * chi_in,
            )

    def measure_residual_centroid(self, scene):
        """
        Compute moment of the residual image for each source and print the remaining flux,
        dipole centroid and dipole size

        Parameters
        ----------
        scene: :py:class:`~scarlet2.Scene`
            Scene containing the sources
        """
        residuals = self.render(scene()) - self.data

        moments = []

        for src in scene.sources:
            bbox, _ = overlap_slices(self.frame.bbox, src.bbox, return_boxes=True)
            source_res = jax.lax.dynamic_slice(residuals, bbox.start, bbox.shape)
            source_weights = jax.lax.dynamic_slice(self.weights, bbox.start, bbox.shape)
            m = Moments(source_res)
            moments.append(m)

        print("Remaining flux")
        for i, m in enumerate(moments):
            print(f"Source {i}, flux: {abs(m[0, 0].sum())}, std: {(1 / source_weights**0.5).sum()}")

        print()
        print("Residual dipole centroid per band")
        for i, m in enumerate(moments):
            print(f"Source {i}, centroid: {m.centroid}")

        print()
        print("Residual dipole size per band")
        for i, m in enumerate(moments):
            print(f"Source {i}, size: {m.size}")


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
