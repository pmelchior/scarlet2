"""Plotting functions"""
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from jax import jvp, grad, jit
from matplotlib.patches import Rectangle, Polygon

from . import measure
from .bbox import Box
from .renderer import ChannelRenderer


def channels_to_rgb(channels):
    """Get the linear mapping of multiple channels to RGB channels
    The mapping created here assumes the the channels are ordered in wavelength
    direction, starting with the shortest wavelength. The mapping seeks to produce
    a relatively even weights for across all channels. It does not consider e.g.
    signal-to-noise variations across channels or human perception.

    Parameters
    ----------
    channels: int in range(0,7)
        Number of channels

    Returns
    -------
    array
     (3, channels) to map onto RGB
    """
    assert channels in range(
        0, 8
    ), "No mapping has been implemented for more than {} channels".format(channels)

    channel_map = np.zeros((3, channels))
    if channels == 1:
        channel_map[0, 0] = channel_map[1, 0] = channel_map[2, 0] = 1
    elif channels == 2:
        channel_map[0, 1] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[1, 0] = 0.333
        channel_map[2, 0] = 0.667
        channel_map /= 0.667
    elif channels == 3:
        channel_map[0, 2] = 1
        channel_map[1, 1] = 1
        channel_map[2, 0] = 1
    elif channels == 4:
        channel_map[0, 3] = 1
        channel_map[0, 2] = 0.333
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.667
        channel_map[2, 1] = 0.333
        channel_map[2, 0] = 1
        channel_map /= 1.333
    elif channels == 5:
        channel_map[0, 4] = 1
        channel_map[0, 3] = 0.667
        channel_map[1, 3] = 0.333
        channel_map[1, 2] = 1
        channel_map[1, 1] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 1.667
    elif channels == 6:
        channel_map[0, 5] = 1
        channel_map[0, 4] = 0.667
        channel_map[0, 3] = 0.333
        channel_map[1, 4] = 0.333
        channel_map[1, 3] = 0.667
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[2, 2] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 2
    elif channels == 7:
        channel_map[:, 6] = 2/3.
        channel_map[0, 5] = 1
        channel_map[0, 4] = 0.667
        channel_map[0, 3] = 0.333
        channel_map[1, 4] = 0.333
        channel_map[1, 3] = 0.667
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[2, 2] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 2
    return channel_map


class Norm(ABC):
    """Base class to normalize the color values of RGB images"""
    def __init__(self):
        self._uint8Max = float(np.iinfo(np.uint8).max)

    def get_intensity(self, im):
        """Compute total intensity image"""
        return jnp.maximum(0, im).sum(axis=0)

    def clip(self, im, m, M):
        """Clip image between m and M"""
        return jnp.maximum(0, jnp.minimum(im - m, M - m))


    def convert_to_uint8(self, im):
        """Convert three-channel image to RGB image with uint8 dtype"""
        im_clipped = self.clip(im, 0, 1)
        uint_im = (im_clipped * self._uint8Max).astype("uint8")
        im_flipped = uint_im.transpose().swapaxes(0, 1)  # 3 x Ny x Nx -> Ny x Nx x 3
        return im_flipped


    def make_rgb_image(self, *im):
        """Compute RGB image from three-channel image"""
        # backwards compatible to astropy Mapping Call
        return self.convert_to_uint8(self.__call__(jnp.stack(im, axis=0)))


    @abstractmethod
    def __call__(self, im):
        """Compute normalized three-channel image"""
        pass


class LinearNorm(Norm):
    def __init__(self, minimum, maximum):
        """Linear norm, mapping the interval [`minimum`, `maximum`] to [0,1]
        
        Parameters
        ----------
        minimum: float
        maximum: float
        """
        self.m, self.M = minimum, maximum
        super().__init__()

    def __call__(self, im):
        return self.clip(im, self.m, self.M) / (self.M - self.m)


class LinearPercentileNorm(LinearNorm):
    def __init__(self, img, percentiles=[1, 99]):
        """Norm that is linear between the two elements of `percentiles` of `img`

        Parameters
        ----------
        img: array
            Image to normalize
        percentiles: list, optional
            Lower and upper percentile to consider. Pixel values below will be
            set to zero, above to saturated.
        """
        assert len(percentiles) == 2
        vmin, vmax = np.percentile(img, percentiles)
        super().__init__(minimum=vmin, maximum=vmax)


class AsinhNorm(Norm):
    def __init__(self, m, M, beta):
        """Norm that scales as arcsinh(I / beta) between `m` and `M`

        See Lupton+(2004) https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L

        Parameters
        ----------
        m: float
            Minimum value to consider
        M: float
            Maximum value to consider
        beta: float
            Turnover point of arcsinh. Below it norm behaves linear, above
            it norm approximates ln(2*I)
        """
        self.m, self.M, self.beta = m, M, beta
        self._rgb_max = 1
        super().__init__()

    def set_rgb_max(self, img, vibrance=0.15):
        """Set maximum value of normalized image

        Parameters
        ----------
        img: array
            Three-channel image
        vibrance: float
            Allowance to exceed normalization of three-channel image.
            Makes images more vibrant but causes slight color shifts towards white in the highlights.
        """
        rgb = self.__call__(img)
        self._rgb_max = rgb[np.isfinite(rgb)].max() / (1 + vibrance)

    def __call__(self, img):
        m = self.m
        M = self.M
        I = self.get_intensity(img)
        with np.errstate(
            invalid="ignore", divide="ignore"
        ):  # n.b. np.where can't and doesn't short-circuit
            # clip between m and M
            I_ = self.clip(I - m, 0, M - m)

            # arcsinh scaling from Lupton+(2004)
            f = np.arcsinh(I_ / self.beta)  # no need to normalize, done below
            rgb = img / (I / f)[None, :, :]

            # keep rgb between 0 and 1 (with an allowance of self.vibrance)
            rgb = rgb / self._rgb_max

            return rgb


class AsinhPercentileNorm(AsinhNorm):
    def __init__(self, img, percentiles=[45, 50, 99], vibrance=0.15):
        """Norm that scales as arcsinh(I / beta) between bottom and top percentile

        Uses the middle percentile to define the turnover `beta`. The defaults
        are chosen such that the median (percentile 50) tries to catch emission
        slightly above the sky level, while the minimum is aiming for the sky
        intensity itself.

        Parameters
        ----------
        img: array_like
            Image to normalize
        percentile: array_like, default=[45,50,99]
            Lower, middle, and upper percentile to consider. Pixel values below will be
            set to zero, above to one. Asinh turnover is given by middle percentile.
        vibrance: float
            Allowance to exceed normalization of three-channel image.
            Makes images more vibrant but causes slight color shifts in the highlights.
        """
        assert len(percentiles) == 3
        m, beta, M = np.percentile(img, percentiles)
        super().__init__(m, M, beta)
        super().set_rgb_max(img, vibrance=vibrance)


class AsinhAutomaticNorm(AsinhNorm):
    def __init__(
        self,
        observation,
        channel_map=None,
        minimum=0,
        upper_percentile=99.5,
        noise_level=1,
        vibrance=0.15,
    ):
        """Norm that scales as arcsinh(I / beta) with parameters chosen automatically

        The turnover `beta` is taken from the at `noise_level` * RMS, where RMS is the
        total variance of the observations. This norm should automatically create an
        image scaling that picks out low-surface brightness features and highlights.

        Parameters
        ----------
        observation: py:class:`~scarlet2.Observation`
            Observation object with weights
        channel_map: array
            Linear mapping from channels to RGB, dimensions (3, channels)
        noise_level: float
            Factor to be multiplied to the total noise RMS to define the turnover point
        upper_percentile: float
            Upper percentile: Pixel values above will be saturated.
        vibrance: float
            Allowance to exceed normalization of three-channel image.
            Makes images more vibrant but causes slight color shifts in the highlights.
        """
        if channel_map is None:
            channel_map = channels_to_rgb(observation.frame.C)

        im3 = img_to_3channel(observation.data, channel_map=channel_map)
        var3 = 1 / observation.weights
        var3 = np.where(np.isfinite(var3), var3, 0)
        var3 = img_to_3channel(1 / observation.weights, channel_map=channel_map)

        # total intensity and variance images
        I = self.get_intensity(im3)
        V = self.get_intensity(var3)

        # find upper clipping point
        (M,) = np.percentile(I.flatten(), [upper_percentile])
        m = minimum

        # find a good turnover point for arcsinh: ~noise level
        rms = np.median(np.sqrt(V))
        beta = rms * noise_level

        super().__init__(m, M, beta)
        super().set_rgb_max(im3, vibrance=vibrance)


def img_to_3channel(img, channel_map=None):
    """Convert multi-band image cube into 3 RGB channels

    Parameters
    ----------
    img: array
        This should be an array with dimensions (channels, height, width).
    channel_map: array
        Linear mapping from channels to RGB, dimensions (3, channels)

    Returns
    -------
    array
        Dimensions (3, height, width), type float
    """
    # expand single img into cube

    assert img.ndim in [2, 3]
    if len(img.shape) == 2:
        ny, nx = img.shape
        img_ = img.reshape(1, ny, nx)
    elif len(img.shape) == 3:
        img_ = img
    C = len(img_)

    # filterWeights: channel x band
    if channel_map is None:
        channel_map = channels_to_rgb(C)
    else:
        assert channel_map.shape == (3, len(img))

    # map channels onto RGB channels
    _, ny, nx = img_.shape
    rgb = jnp.dot(channel_map, img_.reshape(C, -1)).reshape(3, ny, nx)

    rgb = jnp.where(np.isfinite(rgb), rgb, 0)

    return rgb


def img_to_rgb(img, channel_map=None, fill_value=0, norm=None, mask=None):
    """Convert images to normalized RGB.

    If normalized values are outside of the range [0..255], they will be
    truncated such as to preserve the corresponding color.

    Parameters
    ----------
    img: array
        This should be an array with dimensions (channels, height, width).
    channel_map: array
        Linear mapping from channels to RGB, dimensions (3, channels)
    fill_value: float, optional
        Value to use for any masked pixels.
    norm: Norm, optional
        Norm to use for mapping in the allowed range [0..255]. If `norm=None`,
        `scarlet.display.LinearPercentileNorm` will be used.
    mask: array_like, optional
        A [0,1] binary mask to apply over the top of the image,
        where pixels with mask==1 are masked out.

    Returns
    -------
    array
        Dimensions (3, height, width), type float
    """
    im3 = img_to_3channel(img, channel_map=channel_map)
    if norm is None:
        norm = LinearPercentileNorm(im3)
    RGB = norm.make_rgb_image(*im3)
    if mask is not None:
        RGB = jnp.dstack([RGB, ~mask * 255])
    return RGB

panel_size = 4.0 

def observation(
        observation,
        norm=None,
        channel_map=None,
        sky_coords=None,
        show_psf=False,
        add_labels=True,
        split_channels=False,
        fig_kwargs=dict(),
        title_kwargs=dict(),
        label_kwargs={"color": "w", "ha": "center", "va": "center"},
):
    """Plot observation

    Show entire content of `observation`, optionally with list of sources given by `sky_coords` or a PSF image.

    Parameters
    ----------
    observation: :py:class:`~scarlet2.Observation`
    norm: Norm, optional
        Norm to scale the intensity of `observation` into RGB 0..256
    channel_map: array, optional
        Linear mapping from channels to RGB, dimensions (3, channels)
    sky_coords: list, optional
        2D coordinates (in pixel coordinates or sky coordinates).
        If in sky coordinates, the Frame of `observation` needs to have a valid WCS.
    show_psf: bool, optional
        Whether to plot a panel with the PSF model of `observation` centered in the middle
    add_labels: bool, optional
        Whether to plot a text label with the running number for each of the sources in `sky_coords`
    split_channels: bool, optional
        Whether to split the observation into separate channels
    fig_kwargs: dict, optional
        Additional arguments for `mpl.subplots`
    title_kwargs: dict, optional
        Additional arguments for `mpl.set_title`
    label_kwargs: dict, optional
        Additional arguments for `mpl.text`

    Returns
    -------
    mpl.Figure
    """
    if show_psf:
        assert observation.frame.psf is not None, "show_psf requires observation.frame.psf to be set"
        psf_model = observation.frame.psf()

    rows = len(observation.frame.channels) if split_channels else 1
    panels = 1 if show_psf is False else 2
    figsize = fig_kwargs.pop("figsize", None)
    if figsize is None:
        figsize = (panel_size * panels, panel_size * rows)
    fig, ax = plt.subplots(rows, panels, figsize=figsize, squeeze=False, **fig_kwargs)
    if not hasattr(ax, "__iter__"):
        ax = (ax,)

    extent = observation.frame.bbox.get_extent()

    for row in range(rows):
        if split_channels:
            data = observation.data[row]
            mask = observation.weights[row] == 0
            name = observation.frame.channels[row]
            if show_psf:
                psf = psf_model[row]
                # make PSF as bright as the brightest pixel of the observation
                psf *= data.max() / psf.max()
        else:
            data = observation.data
            # Mask any pixels with zero weight in all channels
            mask = np.sum(observation.weights, axis=0) == 0
            name = ""
            if show_psf:
                psf = psf_model
                # make PSF as bright as the brightest pixel of the observation
                psf *= observation.data.mean(axis=0).max() / psf_model.mean(axis=0).max()

        # if there are no masked pixels, do not use a mask
        if np.all(mask == 0):
            mask = None

        panel = 0
        ax[row, panel].imshow(
            img_to_rgb(data, norm=norm, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[row, panel].set_title(f"Observation {name}", **title_kwargs)

        if add_labels and sky_coords is not None:
            for k, center in enumerate(sky_coords):
                center_ = observation.frame.get_pixel(center)
                ax[row, panel].text(*center_[::-1], k, **label_kwargs)

        if show_psf:
            panel = 1
            psf_image = np.zeros(data.shape)
            # insert into middle of "blank" observation
            full_box = Box(psf_image.shape)
            shift = tuple(
                psf_image.shape[d] // 2 - psf.shape[d] // 2
                for d in range(full_box.D)
            )
            model_box = Box(psf.shape) + shift
            model_box.insert_into(psf_image, psf)
            # slices = scarlet.box.overlapped_slices
            ax[row, panel].imshow(img_to_rgb(psf_image, norm=norm), origin="lower")
            ax[row, panel].set_title("PSF", **title_kwargs)

    fig.tight_layout()
    return fig


# ------------------------------------------------------ #
# include a routine to calculate the hallucination score #
#  ----------------------------------------------------- #
def cut_square_box(arr, center, size):
    """
    Cut out a square box from a 2D array based on the center and size.

    Parameters:
    arr (numpy.ndarray): The input 2D array.
    center (tuple): The center of the box in the format (row_center, col_center).
    size (int): The size of the square box (side length).

    Returns:
    numpy.ndarray: The square box extracted from the input array.
    """

    # get the dimensions of the data 
    obsDim = arr.ndim 

    row_center, col_center = center
    #col_center, row_center = center
    half_size = size // 2

    # Calculate the indices for slicing
    start_row = row_center - half_size
    end_row = start_row + size
    start_col = col_center - half_size
    end_col = start_col + size

    # Ensure the indices are within the array bounds
    start_row = max(0, start_row)
    start_col = max(0, start_col)
    if obsDim==2:
        end_row = min(arr.shape[0], end_row)
        end_col = min(arr.shape[1], end_col)
    else:
        end_row = min(arr.shape[1], end_row)
        end_col = min(arr.shape[2], end_col)

    # Cut out the square box
    if obsDim==2:
        square_box = arr[start_row:end_row, start_col:end_col]
    else:
        square_box = arr[:, start_row:end_row, start_col:end_col]

    # pad array up if needed (ie box outside array bounds)
    pad = False
    if obsDim==2:
        if square_box.shape[0] < size or square_box.shape[1] < size:
            pad_low = size - square_box.shape[0]
            pad_high = size - square_box.shape[1]
            pad = True
    else:
        if square_box.shape[1] < size or square_box.shape[2] < size:
            pad_low = size - square_box.shape[1]
            pad_high = size - square_box.shape[2]
            pad = True

    # perform the padding
    if pad == True:
        # If the square box is not the correct size, pad it with zeros
        if pad_low < 0:
            pad_low = 0
        if pad_high < 0:
            pad_high = 0
        if obsDim <= 2:
            square_box = np.pad(square_box, ((pad_low, 0), (pad_high, 0)), 
                            mode="constant", constant_values=0)
        else:
                # Get the original array shape
            original_height, original_width, num_channels = square_box.shape

            # Create a new zero-padded array
            padded_rgb_array = np.zeros((original_height + 2 * pad_high,
                                 original_width + 2 * pad_low,
                                 num_channels), dtype=square_box.dtype)

            # Place the original RGB array in the center of the padded array
            padded_rgb_array[pad_high:pad_high + original_height,
                      pad_low:pad_low + original_width, :] = square_box

    return square_box

@jax.grad
def neural_grad(galaxy, src):
    parameters = src.get_parameters(return_info=True)
    prior = 2 * sum(info["prior"].log_prob(galaxy)
                        for name, (p, info) in parameters.items()
                        if info["prior"] is not None
                        )
    return prior

def log_like(morph, spectrum, data, weights):
    model = morph[None, :, :] * spectrum[:, None, None]
    D = jnp.prod(jnp.asarray(data.shape)) - jnp.sum(weights == 0)
    log_norm = D / 2 * jnp.log(2 * jnp.pi)
    log_like = -jnp.sum(weights * (model - data) ** 2) / 2
    return log_like - log_norm

# --------------------- #
# Hessian approximation #
# --------------------- #
# https://arxiv.org/pdf/2006.00719.pdf

# for regular functions f
def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]

# for score functions
def hvp_grad(grad_f, primals, tangents):
    return jvp(grad_f, primals, tangents)[1]

# diagonals of Hessian from HVPs
def hvp_rad(hvp, shape):
    max_iters = 100 # maximum number of iterations
    H = jnp.zeros(shape, dtype=jnp.float32)
    H_ = jnp.zeros(shape, dtype=jnp.float32)
    for i in range(max_iters):
        key = random.PRNGKey(i)
        z = random.rademacher(key, shape , dtype=jnp.float32)
        H += jnp.multiply(z, hvp(z))
        if i > 0:
            norm = jnp.linalg.norm(H/(i+1) - H_/i, ord=2)
            if norm < 1e-6 * jnp.linalg.norm(H/(i+1), ord=2): # gets reasonable results with 1e-2
                break
        H_ = H
    return H/(i+1) 

#TODO: fix the jit compilation errors here
def hallucination_score(scene, obs, src_num):
    src = scene.sources[src_num]
    center = np.array(src.morphology.bbox.center)[::-1]
    morph = src.morphology.data
    f = lambda morph: neural_grad(morph, src)
    jit_hvp_x2 = jit(lambda z: hvp_grad(f, (morph,), (z,)))
    hvp_nn = hvp_rad(jit_hvp_x2, morph.shape)
    hvp_nn = np.array(hvp_nn)

    model_scene = scene()
    morph = model_scene[src_num]  # FIXME: this must be wrong because that is a channel image, not a source image
    spectrum = jnp.array((1,))
    data = obs.data
    weights = obs.weights

    # jit the HVP for this loss and this morph model
    f = lambda morph: log_like(morph, spectrum, data, weights)
    jit_hvp_x = jit(lambda z: hvp(f, (morph,), (z,)))
    hvp_ll = hvp_rad(jit_hvp_x, morph.shape)
    
    box_size = hvp_nn.shape[1] 
    # Cut out the square box
    hvp_ll_cut = cut_square_box(hvp_ll, center, box_size)
    hallucination = -hvp_nn + hvp_ll_cut
    
    return -hallucination * src.morphology() , jnp.sum(-hallucination * src.morphology())

def confidence(scene, observation):
    sources = scene.sources
    n_sources = len(sources)
    metrics = np.zeros(n_sources)
    for k, src in enumerate(sources):
        hallucination, metric = hallucination_score(scene, observation, k)
        metrics[k] = metric
    return metrics


def sources(
        scene,
        observation=None,
        norm=None,
        channel_map=None,
        show_model=True,
        show_observed=False,
        show_rendered=False,
        show_spectrum=True,
        model_mask=None,
        add_markers=True,
        add_boxes=False,
        fig_kwargs=dict(),
        title_kwargs=dict(),
        marker_kwargs={"color": "w", "marker": "x", "mew": 1, "ms": 10},
        box_kwargs={"facecolor": "none", "edgecolor": "w", "lw": 0.5},
):
    """Plot all sources in `scene`

    Creates one figure, with each source in `scene` occupying one row. Depending on the chosen options, multiple panels
    per source will be created.

    Parameters
    ----------
    scene: :py:class:`~scarlet2.Scene`
    observation: :py:class:`~scarlet2.Observation`, optional
        The observation to render the sources for, or to show the data of.
        Only needed when `show_observed` or `show_rendered` is True.
    norm: Norm, optional
        Norm to scale the intensity of `observation` into RGB 0..256
    channel_map: array, optional
        Linear mapping from channels to RGB, dimensions (3, channels)
    show_model: bool, optional
        Whether to show the internal model of each source
    show_observed: bool, optional
        Whether to show the observations in the same region as the source
    show_rendered: bool, optional
        Whether to show the model of each source rendered into the frame of `observation`
    show_spectrum: bool, optional
        Whether to show the spectrum of each source
    add_markers: bool, optional
        Whether to plot a marker at the center for each source
        Requires the source to have a `center` attribute.
    add_boxes: bool, optional
        Whether to plot the bounding box of each source
    fig_kwargs: dict, optional
        Additional arguments for `mpl.subplots`
    title_kwargs: dict, optional
        Additional arguments for `mpl.set_title`
    marker_kwargs: dict, optional
        Additional arguments for `mpl.plot` of the source centers
    box_kwargs: dict, optional
        Additional arguments for `mpl.Polygon`

    Returns
    -------
    mpl.Figure
    """
    sources = scene.sources
    n_sources = len(sources)
    panels = sum((show_model, show_observed, show_rendered, show_spectrum))

    figsize = fig_kwargs.pop("figsize", None)
    if figsize is None:
        figsize = (panel_size * panels, panel_size * n_sources)

    fig, ax = plt.subplots(n_sources, panels, figsize=figsize, squeeze=False, **fig_kwargs)

    for k, src in enumerate(sources):

        center = np.array(src.center)[::-1]
        start, stop = src.bbox.start[-2:][::-1], src.bbox.stop[-2:][::-1]
        box_coords = (start, (start[0], stop[1]), stop, (stop[0], start[1]))

        # model in its bbox
        panel = 0
        model = src()
        if show_model:
            # Show the unrendered model in it's bbox
            extent = src.bbox.get_extent()
            ax[k][panel].imshow(
                img_to_rgb(model, norm=norm, channel_map=channel_map, mask=model_mask),
                extent=extent,
                origin="lower",
            )
            ax[k][panel].set_title("Model Source {}".format(k), **title_kwargs)
            if center is not None and add_markers:
                ax[k][panel].plot(*center, **marker_kwargs)
            panel += 1

        if show_rendered or show_observed:
            assert observation is not None, "show_rendered or show_observed requires observation"
            extent = observation.frame.bbox.get_extent()

        # model in observation frame
        if show_rendered:
            model = scene.evaluate_source(src)
            model_ = observation.render(model)

            ax[k][panel].imshow(
                img_to_rgb(model_, norm=norm, channel_map=channel_map, mask=model_mask),
                extent=extent,
                origin="lower",
            )
            ax[k][panel].set_title("Model Source {} Rendered".format(k), **title_kwargs)
            if add_markers:
                ax[k][panel].plot(*center, **marker_kwargs)
            if add_boxes:
                poly = Polygon(box_coords, closed=True, **box_kwargs)
                ax[k][panel].add_artist(poly)
            panel += 1

        if show_observed:
            # Center the observation on the source and display it
            ax[k][panel].imshow(
                img_to_rgb(observation.data, norm=norm, channel_map=channel_map),
                extent=extent,
                origin="lower",
            )
            ax[k][panel].set_title("Observation".format(k), **title_kwargs)
            if add_markers:
                ax[k][panel].plot(*center, **marker_kwargs)
            if add_boxes:
                poly = Polygon(box_coords, closed=True, **box_kwargs)
                ax[k][panel].add_artist(poly)
            panel += 1

        if show_spectrum:
            # needs to be evaluated in the source box to prevent truncation
            spectra = [measure.flux(src), ] + [measure.flux(component) for component in src.components]
            
            for spectrum in spectra:
                ax[k][panel].plot(spectrum)
            ax[k][panel].set_xticks(range(len(spectrum)))
            if scene.frame.channels is not None:
                ax[k][panel].set_xticklabels(scene.frame.channels)
            ax[k][panel].set_title("Spectrum", **title_kwargs)
            ax[k][panel].set_xlabel("Channel")
            ax[k][panel].set_ylabel("Intensity")

    fig.tight_layout()
    return fig


def scene(
        scene,
        observation=None,
        norm=None,
        channel_map=None,
        show_model=True,
        show_observed=False,
        show_rendered=False,
        show_residual=False,
        add_labels=True,
        add_boxes=False,
        split_channels=False,
        fig_kwargs=dict(),
        title_kwargs=dict(),
        label_kwargs={"color": "w", "ha": "center", "va": "center"},
        box_kwargs={"facecolor": "none", "edgecolor": "w", "lw": 0.5},
):
    """Plot all sources to recreate the scence.
    The functions provide a fast way of evaluating the quality of the entire model,
    i.e. the combination of all scenes that seek to fit the observation.

    Parameters
    ----------
    scene: :py:class:`~scarlet2.Scene`
    observation: :py:class:`~scarlet2.Observation`, optional
    norm: Norm
        Norm to scale the intensity of `observation` into RGB 0..256
    channel_map: array_like
        Linear mapping from channels to RGB, dimensions (3, channels)
    show_model: bool
        Whether the internal model is shown in the model frame
    show_observed: bool
        Whether the observation is shown
    show_rendered: bool
        Whether the model, rendered to match the observation, is shown
    show_residual: bool
        Whether the residuals between rendered model and observation is shown
    add_label: bool
        Whether each source is labeled with its numerical index in the source list
    add_boxes: bool
        Whether each source box is shown
    split_channels: bool
        Whether to split the observation into separate channels
    fig_kwargs: dict
        kwargs for plt.figure()
    title_kwargs: dict
        kwargs for plt.title()
    label_kwargs: dict
        kwargs for source labels
    box_kwargs: dict
        kwargs for source boxes

    Returns
    -------
    mpl.Figure
    """

    # for animations with multiple scenes
    if hasattr(scene, '__iter__'):
        scenes = scene
        scene = scenes[0]

    if show_observed or show_rendered or show_residual:
        assert (
                observation is not None
        ), "Provide matched observation to show observed frame"

    rows = len(observation.frame.channels) if split_channels else 1
    panels = sum((show_model, show_observed, show_rendered, show_residual))
    figsize = fig_kwargs.pop("figsize", None)
    if figsize is None:
        figsize = (panel_size * panels, panel_size * rows)
    fig, ax = plt.subplots(rows, panels, figsize=figsize, squeeze=False, **fig_kwargs)

    model = scene()
    if show_rendered or show_residual:
        model_rendered = observation.render(model)
    if show_model and observation is not None:
        c = ChannelRenderer(scene.frame, observation.frame)
        model = c(model)
    if show_observed or show_residual:
        data = observation.data
        mask = observation.weights == 0

    for row in range(rows):
        if split_channels:
            sel = row
            name = observation.frame.channels[row]
            channel_map = None
        else:
            sel = slice(None)
            name = ""

        panel = 0
        if show_model:
            extent = scene.frame.bbox.get_extent()
            # if scene.frame.wcs is not None:
            #     extent = scene.frame.get_pixel(
            #         scene.frame.get_sky_coord(np.array([[extent[0], extent[1]], [extent[2], extent[3]]]))
            #     ).flatten()

            model_img = ax[row, panel].imshow(
                img_to_rgb(model[sel], norm=norm, channel_map=channel_map),
                extent=extent,
                origin="lower",
            )
            ax[row, panel].set_title(f"Model {name}", **title_kwargs)
            panel += 1

        if show_rendered:
            rendered_img = ax[row, panel].imshow(
                img_to_rgb(model_rendered[sel], norm=norm, channel_map=channel_map),
                origin="lower",
            )
            ax[row, panel].set_title("Model Rendered", **title_kwargs)
            panel += 1

        if show_observed or show_rendered:
            if split_channels:
                mask_ = mask[sel]
            else:
                # Mask any pixels with zero weight in all channels
                mask_ = np.sum(mask, axis=0) > 0
            if np.all(mask_ == 0):
                mask_ = None

        if show_observed:
            observed_img = ax[row, panel].imshow(
                img_to_rgb(data[sel], norm=norm, channel_map=channel_map, mask=mask_),
                origin="lower",
            )
            ax[row, panel].set_title("Observation", **title_kwargs)
            panel += 1

        if show_residual:
            residual = data[sel] - model_rendered[sel]
            norm_ = LinearPercentileNorm(residual)
            residual_img = ax[row, panel].imshow(
                img_to_rgb(residual, norm=norm_, channel_map=channel_map, mask=mask_),
                origin="lower",
            )
            ax[row, panel].set_title("Data - Model", **title_kwargs)
            panel += 1

        for k, src in enumerate(scene.sources):
            start, stop = src.bbox.spatial.start[::-1], src.bbox.spatial.stop[::-1]

            if add_boxes:
                panel = 0
                if show_model:
                    extent = [start[0], stop[0], start[1], stop[1]]
                    rect = Rectangle(
                        (extent[0], extent[2]),
                        extent[1] - extent[0],
                        extent[3] - extent[2],
                        **box_kwargs
                    )
                    ax[row, panel].add_artist(rect)
                    panel = 1
                if observation is not None:
                    start = observation.frame.get_pixel(scene.frame.get_sky_coord(np.array(start))).flatten()
                    stop = observation.frame.get_pixel(scene.frame.get_sky_coord(np.array(stop))).flatten()
                    box_coords = (start, (start[0], stop[1]), stop, (stop[0], start[1]))
                    for panel in range(panel, panels):
                        poly = Polygon(box_coords, closed=True, **box_kwargs)
                        ax[row, panel].add_artist(poly)

            if add_labels:
                center = np.array(src.center)[::-1]
                panel = 0
                if show_model:
                    ax[row, panel].text(*center, k, **label_kwargs)
                    panel = 1
                if observation is not None:
                    center = observation.frame.get_pixel(scene.frame.get_sky_coord(center)).flatten()
                    for panel in range(panel, panels):
                        ax[row, panel].text(
                            *center, k, **label_kwargs
                        )

    fig.tight_layout()

    try:
        # animate multiple scenes
        n_frames = len(scenes)

        # update only images dependent on the current state of scene
        def update(i):
            updated = []
            scene = scenes[i]
            model = scene()
            if show_model:
                model_img.set_data(img_to_rgb(model, norm=norm, channel_map=channel_map))
                updated.append(model_img)

            if show_rendered or show_residual:
                model = observation.render(model)

            if show_rendered:
                rendered_img.set_data(img_to_rgb(model, norm=norm, channel_map=channel_map, mask=mask))
                updated.append(rendered_img)

            if show_residual:
                residual = observation.data - model
                norm_ = LinearPercentileNorm(residual)
                residual_img.set_data(img_to_rgb(residual, norm=norm_, channel_map=channel_map, mask=mask))
                updated.append(residual_img)
            return updated

        ani = animation.FuncAnimation(fig=fig, func=update, frames=n_frames, interval=30)
        return ani

    except NameError:
        return fig
