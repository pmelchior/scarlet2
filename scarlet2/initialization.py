import jax.numpy as jnp

from . import Scenery
from . import measure
from .bbox import Box
from .morphology import GaussianMorphology


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


def adaptive_gaussian_morph(obs, center, min_size=11, delta_size=3, min_snr=20, min_corr=0.99):
    """
    Create image of a Gaussian from the centered 2nd moments of the observation.

    This method finds small box around the center so that the edge flux has a minimum SNR,
    the color of the edge pixels remains highly correlated to the center pixel color,
    and the flux in all channels is lower than with the last acceptable box size.
    This should effectively isolate the source against the noise background and neighboring objects.

    Parameters
    ----------
    obs: `~scarlet2.Observation`
    center: tuple
        source enter, in pixel or sky coordinates
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
    jnp.array morphology image, normalized to (0,1)
    """
    assert obs.weights is not None, "Observation weights are required"

    peak_spectrum = pixel_spectrum(obs, center, correct_psf=True)
    last_spectrum = peak_spectrum.copy()
    box2d = Box((min_size, min_size))
    center_pix = obs.frame.get_pixel(center)
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
    return gaussian_morphology(obs, box, center=center_pix)


def compact_morphology(min_value=1e-6):
    """
    Create image of the PointSourceMorphology model, i.e. the most compact source possible.

    Returns
    -------
    jnp.array morphology image, normalized to (0,1)

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
    morph = jnp.maximum(morph, min_value)
    return morph


def gaussian_morphology(
        obs,
        bbox,
        center=None,
        min_value=1e-6,
):
    """Create image of a Gaussian from the 2nd moments of the observation in the region of the bounding box.

    Parameters
    ----------
    obs : `~scarlet2.Observation`
    center : tuple
        central pixel of the source
    bbox: `~scarlet2.BBox`
        box to cut out source from observation, in pixel coordinates
    min_value: float
        minimum pixel value (useful to set to > 0 for positivity constraints)

    Returns
    -------
    jnp.array morphology image, normalized to (0,1)
    """
    # cut out image from observation
    cutout_img = obs.data[bbox.slices]
    if center is None:
        # define reference center, flatten image across channels
        # TODO: use spectrum-based detection coadd instead
        center = measure.centroid(cutout_img.sum(axis=0))
    else:
        center -= jnp.array(bbox.spatial.origin)

    try:
        frame = Scenery.scene.frame
    except AttributeError:
        print("Adaptive morphology can only be created within the context of a Scene")
        print("Use 'with Scene(frame) as scene: Source(...)'")
        raise
    if frame.psf is None:
        raise AttributeError("Adaptive morphology can only be create with a PSF in the model frame")

    # getting moment measures:
    # 1) get convolved moments
    g_ = measure.moments(cutout_img, center=center, N=2)
    if hasattr(obs, "_dp"):
        dp = obs._dp
    else:
        # compute the moments of the difference kernel between the model PSF and the observed PSF
        # TODO: this needs a resampling operation if the two frames have different resolutions
        # We need obs.frame.psf in the same pixels as model PSF
        p = measure.moments(obs.frame.psf(), N=2)
        p0 = measure.moments(frame.psf(), N=2)
        dp = measure.deconvolve(p, p0)
        # store in obs for repeated use
        object.__setattr__(obs, "_dp", dp)
    # 2) deconvolve in moment space
    g = measure.deconvolve(g_, dp)
    # 3) average over channels
    spectrum = g[0, 0].copy()
    for key in g.keys():
        g[key] = jnp.median(g[key])  # this is not SNR weighted, which might be better
    # 4) compute size and ellipticity for a Gaussian
    T = measure.size(g)
    ellipticity = measure.ellipticity(g)

    # create image of Gaussian with these 2nd moments
    if jnp.isfinite(center).all() and jnp.isfinite(T) and jnp.isfinite(ellipticity).all():
        center += jnp.array(bbox.spatial.origin)
        morph = GaussianMorphology(center, T, ellipticity, bbox=bbox)()
        spectrum /= morph.sum()
        morph = jnp.maximum(morph, min_value)
    else:
        raise ValueError(
            f"Gaussian morphology not possible with center={center}, size={T}, and ellipticity={ellipticity}!")
    return spectrum, morph


# initialise the spectrum
def pixel_spectrum(observations, pos, correct_psf=False):
    """Get the spectrum at a given position in the observation(s).

    Yields the spectrum of a single-pixel source with flux 1 in every channel,
    concatenated for all observations.

    If `correct_psf`, it homogenizes the PSFs of the observations, which yields the
    correct spectrum for a flux=1 point source.

    If `model` is set, it reads of the value of the model at `sky_coord` and yields the
    spectrum for that model.

    Parameters
    ----------
    observations: instance or list of `~scarlet.Observation`
        Observation to extract SED from.
    pos: tuple
        Position in the observation
    correct_psf: bool
        If PSF shape variations in the observations should be corrected.

    Returns
    -------
    spectrum: `~jnp.array` or list thereof if given list of observations
    """
    if not hasattr(observations, "__iter__"):
        single = True
        observations = (observations,)
    else:
        single = False

    spectra = []
    for obs in observations:
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

        spectra.append(spectrum)

    if single:
        return spectra[0]
    return spectra
