"""Helper methods to initialize sources"""

import operator
from functools import reduce

import jax.numpy as jnp

from . import Scenery
from . import measure
from .bbox import Box
from .morphology import GaussianMorphology
from .observation import Observation


# function to calculate values of perimeter pixels
def _get_edge_pixels(img, box):
    box_slices = box.slices
    box_start = box.start
    box_stop = box.stop
    edge = [
        img[:, box_slices[0], box_start[1]],
        img[:, box_slices[0], box_stop[1] - 1],
        img[:, box_start[0], box_slices[1]],
        img[:, box_stop[0] - 1, box_slices[1]],
    ]
    return jnp.concatenate(edge, axis=1)


def make_bbox(obs, center_pix, min_size=11, delta_size=3, min_snr=20, min_corr=0.99):
    """Make a bounding box for source at center

    This method finds small box around the center so that the edge flux has a minimum SNR,
    the color of the edge pixels remains highly correlated to the center pixel color,
    and the flux in all channels is lower than with the last acceptable box size.
    This should effectively isolate the source against the noise background and neighboring objects.

    Parameters
    ----------
    obs: :py:class:`~scarlet2.Observation`
    center_pix: tuple
        source enter, in pixel coordinates
    min_size: int
        smallest box size
    delta_size: int
        incremental increase of box size
    min_snr: float
        minimum SNR of edge pixels (aggregated over all observation channel) to allow increase of box size
    min_corr: float
        minimum correlation coefficient between center and edge color to allow increase of box size

    Returns
    -------
    ~scarlet2.Box
    """
    assert isinstance(obs, Observation)
    assert obs.weights is not None, "Observation weights are required"

    peak_spectrum = pixel_spectrum(obs, center_pix, correct_psf=True)
    last_spectrum = peak_spectrum.copy()
    box2d = Box((min_size, min_size))
    if not obs.frame.bbox.spatial.contains(center_pix):
        raise ValueError(f"Pixel coordinate expected, got {center_pix}")
    box2d.set_center(center_pix.astype(int))

    # increase box size until SNR is below threshold or spectrum changes significantly
    while max(box2d.shape) < max(obs.frame.bbox.spatial.shape):
        edge_pixels = _get_edge_pixels(obs.data, box2d)
        edge_spectrum = jnp.mean(edge_pixels, axis=-1)
        edge_spectrum /= jnp.sqrt(jnp.dot(edge_spectrum, edge_spectrum))

        weight_edge_pixels = _get_edge_pixels(obs.weights, box2d)
        snr_edge_pixels = edge_pixels * jnp.sqrt(weight_edge_pixels)
        valid_edge_pixel = weight_edge_pixels > 0
        mean_snr = jnp.sum(jnp.sum(snr_edge_pixels, axis=-1) / jnp.sum(valid_edge_pixel, axis=-1))

        if mean_snr < min_snr:
            break

        spec_corr = jnp.dot(edge_spectrum, peak_spectrum) / \
                    jnp.sqrt(jnp.dot(peak_spectrum, peak_spectrum)) / jnp.sqrt(jnp.dot(edge_spectrum, edge_spectrum))

        if spec_corr < min_corr or jnp.any(edge_spectrum > last_spectrum):
            if min(box2d.shape) > min_size:
                box2d = box2d.shrink(delta_size)
            break

        box2d = box2d.grow(delta_size)

    box = obs.frame.bbox[0] @ box2d
    return box


def compact_morphology(min_value=1e-6, max_value=1 - 1e-6):
    """Create image of the point source morphology model, i.e. the most compact source possible

    Parameters
    ----------
    min_value: float
        Minimum pixel value (needed for positively constrained morphologies)
    max_value: float
        minimum pixel value (useful to set to < 1 for unit interval constraints)


    Returns
    -------
    array
        2D array, normalized to the range [0,1]

    """
    try:
        frame = Scenery.scene.frame
    except AttributeError:
        print("Compact morphology can only be created within the context of a Scene")
        print("Use 'with Scene(frame) as scene: Source(...)'")
        raise
    if frame.psf is None:
        raise AttributeError("Compact morphology can only be create with a PSF in the model frame")

    morph = frame.psf.morphology()
    morph = jnp.minimum(jnp.maximum(morph, min_value), max_value)
    return morph


def standardized_moments(
        obs,
        center,
        footprint=None,
        bbox=None,
):
    """Create image of a Gaussian from the 2nd moments of the observation in the region of the bounding box

    The methods cuts out the pixel included in `bbox` or in `footprint`, measures their 2nd moments with respect to
    `center`, adjust the spatial coordinates and the PSF to match the model frame.

    Parameters
    ----------
    obs: :py:class:`~scarlet2.Observation`
    center: tuple
        central pixel of the source
    footprint: array, optional
        2D image with non-zero values for all pixels associated with this source (aka a segmentation map or footprint)
    bbox: :py:class:`~scarlet2.BBox`, optional
        box to cut out source from observation, in pixel coordinates

    Returns
    -------
    measure.Moments
        2nd moments, deconvolved, and in the coordinate frame of the model frame

    Raises
    ------
    AssertionError
        If neither `bbox` or `footprint` are set
    """
    assert isinstance(obs, Observation)
    # construct box from footprint
    if bbox is None:
        assert footprint is not None
        bbox = Box.from_data(footprint)
    # construct footprint as step function inside bbox
    if footprint is None:
        assert bbox is not None
        footprint = jnp.zeros(obs.frame.bbox.spatial.shape)
        footprint = footprint.at[bbox.spatial.slices].set(1)

    # cutout image and footprint
    cutout_img = obs.data[bbox.slices]
    cutout_fp = footprint[bbox.spatial.slices]
    center -= jnp.array(bbox.spatial.origin)

    try:
        frame = Scenery.scene.frame
    except AttributeError:
        print("Adaptive morphology can only be created within the context of a Scene")
        print("Use 'with Scene(frame) as scene: Source(...)'")
        raise
    if frame.psf is None:
        raise AttributeError("Adaptive morphology can only be created with a PSF in the model frame")

    # getting moment measures:
    # 1) get convolved moments
    g = measure.Moments(cutout_img, center=center, weight=cutout_fp[None, :, :], N=2)
    # 2) adjust moments for model frame
    g.transfer(obs.frame.wcs, frame.wcs)
    # 3) deconvolve from PSF (actually: difference kernel between obs PSF and model frame PSF)
    if hasattr(obs, "_dp"):
        p = obs._dp
    else:
        # moments of difference kernel between the model PSF and the observed PSF
        p = measure.Moments(obs.frame.psf(), N=2)
        p.transfer(obs.frame.wcs, frame.wcs)
        p0 = measure.Moments(frame.psf(), N=2)
        p.deconvolve(p0)
        # store in obs for repeated use
        object.__setattr__(obs, "_dp", p)
    # 3) deconvolve from difference kernel
    g.deconvolve(p)
    return g


def from_gaussian_moments(
        obs,
        center,
        min_size=11,
        delta_size=3,
        min_snr=20,
        min_corr=0.99,
        min_value=1e-6,
        max_value=1 - 1e-6,
):
    """Create a Gaussian-shaped morphology and associated spectrum from the observation(s).

    The method determines an suitable bounding box that contains the source given its `center`,
    computes the deconvolved moments up to order 2, constructs the spectrum from the 0th moment and a morphology image
    from the 2nd moments (assuming a Gaussian shape).

    If multiple observations are given, it takes the median of the moments in the same channel.

    Parameters
    ----------
    obs: :py:class:`~scarlet2.Observation` or list
        Observation from which the source is initialized.
    center: tuple
        central pixel of the source
    min_size: int
        smallest size (in pixels) for source bounding box
    delta_size: int
        increase in pixel size for the source bounding box
    min_snr: float
        minimum SNR of edge pixels (aggregated over all observation channel) to allow increase of box size
    min_corr: float
        minimum correlation coefficient between center and edge color to allow increase of box size
    min_value: float
        minimum pixel value (useful to set to > 0 for positivity constraints)
    max_value: float
        minimum pixel value (useful to set to < 1 for unit interval constraints)

    Returns
    -------
    (array,array)
         Spectrum and morphology arrays

    Warnings
    --------
    This method is stable only for isolated sources. In cases of significant blending, the size of the bounding box
    and the measured moments are likely biased high.

    See Also
    --------
    make_bbox: Defines bounding box that contains the source
    standardized_moments: Computes 2nd moments for source in bounding box
    """

    # TODO: implement with source footprints given for each observation
    # get moments from all channels in all observations
    if not isinstance(obs, (list, tuple)):
        observations = (obs,)
    else:
        observations = obs
    centers = [obs_.frame.get_pixel(center) for obs_ in observations]
    boxes = [make_bbox(obs_, center_, min_size=min_size, delta_size=delta_size, min_snr=min_snr, min_corr=min_corr) for
             obs_, center_ in zip(observations, centers)]
    moments = [standardized_moments(obs_, center_, bbox=bbox_) for obs_, center_, bbox_ in
               zip(observations, centers, boxes)]

    # flat lists of spectra, sorted as model frame channels
    spectra = jnp.concatenate([g[0, 0] for g in moments])
    channels = reduce(operator.add, [obs_.frame.channels for obs_ in observations])
    spectrum = _sort_spectra(spectra, channels)

    # average over all channels
    g = moments[0]
    for key in g.keys():
        g[key] = jnp.concatenate([g[key] for g in moments])  # combine all observations
        g[key] = jnp.median(g[key])  # this is not SNR weighted nor consistent aross different moments, but works(?)

    # create morphology and evaluate at center
    morph = GaussianMorphology.from_moments(g)
    morph = morph()
    spectrum /= morph.sum()
    morph = jnp.minimum(jnp.maximum(morph, min_value), max_value)
    return spectrum, morph


# initialise the spectrum
def pixel_spectrum(obs, pos, correct_psf=False):
    """Get the spectrum at a given position in the observation(s).

    Yields the spectrum of a single-pixel source with flux 1 in every channel,
    concatenated for all observations.

    Parameters
    ----------
    obs: `:py:class:`~scarlet2.Observation` or list
        Observation(s) to extract pixel SED from
    pos: tuple
        Position in the observation. Needs to be in sky coordinates if multiple observations have different locations
        or pixel scales.
    correct_psf: bool, optional
        Whether PSF shape variations in the observations should be corrected. If `True`, this method homogenizes the
        PSFs of the observations, which yields the correct spectrum for a flux=1 point source.

    Returns
    -------
    array or list
        If `obs` is a list, the method returns the associate list of spectra.
    """

    # for multiple observations, get spectrum from each observation and then combine channels in order of model frame
    if isinstance(obs, (list, tuple)):

        # flat lists of spectra and channels in order of observations
        spectra = jnp.concatenate(
            [pixel_spectrum(obs_, pos, correct_psf=correct_psf) for obs_ in obs])
        channels = reduce(operator.add, [obs_.frame.channels for obs_ in obs])
        spectrum = _sort_spectra(spectra, channels)

        return spectrum

    assert isinstance(obs, Observation)

    pixel = obs.frame.get_pixel(pos).astype(int)

    if not obs.frame.bbox.spatial.contains(pixel):
        raise ValueError(f"Pixel coordinate expected, got {pixel}")

    spectrum = obs.data[:, pixel[0], pixel[1]].copy()

    if correct_psf and obs.frame.psf is not None:

        try:
            frame = Scenery.scene.frame
        except AttributeError:
            print("Adaptive morphology can only be created within the context of a Scene")
            print("Use 'with Scene(frame) as scene: Source(...)'")
            raise
        if frame.psf is None:
            raise AttributeError("Adaptive morphology can only be create with a PSF in the model frame")

        # correct spectrum for PSF-induced change in peak pixel intensity
        psf_model = obs.frame.psf()
        psf_peak = psf_model.max(axis=(-2, -1))

        psf0_model = frame.psf()
        psf0_peak = psf0_model.max(axis=(-2, -1))

        spectrum /= psf_peak / psf0_peak

    if jnp.any(spectrum <= 0):
        # If the flux in all channels is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = f"Zero or negative spectrum {spectrum} at {pos}"
        if jnp.all(spectrum <= 0):
            print("Zero or negative spectrum in all channels: Setting spectrum to 1")
            spectrum = jnp.ones_like(spectrum)
        print(msg)

    return spectrum


def _sort_spectra(spectra, channels):
    try:
        frame = Scenery.scene.frame
    except AttributeError:
        print("Multi-observation initialization can only be created within the context of a Scene")
        print("Use 'with Scene(frame) as scene: ...")
        raise

    spectrum = []
    for channel in frame.channels:
        try:
            idx = channels.index(channel)
            spectrum.append(spectra[idx])
        except ValueError:
            msg = f"Channel '{channel}' not found in observations. Setting amplitude to 0."
            print(msg)
            spectrum.append(0)
    spectrum = jnp.array(spectrum)
    return spectrum
