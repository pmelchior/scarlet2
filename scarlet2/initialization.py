# ---------------------------------------------------------------------------- #
# Initialization routine for Scarlet2. We define function allowing for the     #
# initialization of source morphologies and spectrums of indentified sources.  #
#                                                                              #
# For morphologies, we input the observational image with the locations of the #
# sources calculated. Then we fit a 2D Gaussian to the cutout and optimize for #
# the best fit with a maximum amplitute of 1.                                  #
#                                                                              #
# For spectrums, we input the observational image as with the morphology, we   #
# just take the pixel value of the center of the source for each band.         #
# ---------------------------------------------------------------------------- #

import numpy as np
import numpy.ma as ma
import jax.numpy as jnp
import jax
from .plot import cut_square_box
from .morphology import GaussianMorphology
from jaxopt import ScipyBoundedMinimize
import dm_pix
from functools import partial
from .measure import moments, binomial, deconvolve, snr


# function to calculate values of perimeter pixels
def perimeter_values(array_2d):
    """Return the values of the perimeter pixels of a 2D array."""
    rows, cols = len(array_2d), len(array_2d[0])
    perimeter_values = []
    # Top row
    perimeter_values.extend(array_2d[0])
    # Bottom row
    if rows > 1:
        perimeter_values.extend(array_2d[-1])
    # Left and right columns, excluding corners to avoid double counting
    if cols > 1:
        perimeter_values.extend(array_2d[i][0] for i in range(1, rows - 1))
        if (
            cols > 2
        ):  # Check if there are more than 2 columns to avoid duplicate corner values
            perimeter_values.extend(array_2d[i][-1] for i in range(1, rows - 1))

    return perimeter_values


# create a 2D Gaussian for simple feature free Initializations
def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y):
    """2D Gaussian function.
    Parameters
    ----------
    x : jnp.ndarray
    y : jnp.ndarray
    mu_x : float mean of the Gaussian in the x direction
    mu_y : float mean of the Gaussian in the y direction
    sigma_x : float standard deviation of the Gaussian in the x direction
    sigma_y : float standard deviation of the Gaussian in the y direction

    Returns
    -------
    jnp.ndarray 2D Gaussian profile
    """
    return jnp.exp(
        -((x - mu_x) ** 2 / (2 * sigma_x**2) + (y - mu_y) ** 2 / (2 * sigma_y**2))
    )


@partial(jax.jit, static_argnums=0)
def create_gaussian_array(boxsize, sigma_x=1, sigma_y=1, theta=0):
    """Create a 2D array with a Gaussian profile in the center.
    Parameters
    ----------
    boxsize : int size of the bounding box around source
    sigma_x : float standard deviation of the Gaussian in the x direction
    sigma_y : float standard deviation of the Gaussian in the y direction
    theta : float rotation angle of the 2D Gaussian

    Returns
    -------
    jnp.ndarray 2D array with a Gaussian profile rotated by theta
    """
    x = jnp.linspace(-boxsize / 2, boxsize / 2, boxsize)
    x, y = jnp.meshgrid(x, x)
    # Center of the Gaussian profile
    mu_x, mu_y = 0, 0
    # Generate the Gaussian profile
    gaussian_profile = gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y)
    gaussian_profile = (gaussian_profile - jnp.min(gaussian_profile)) / (
        jnp.max(gaussian_profile) - jnp.min(gaussian_profile)
    )
    gaussian_profile = jnp.array(gaussian_profile, dtype=jnp.float32)
    gaussian_profile = jnp.expand_dims(gaussian_profile, axis=-1)
    data = dm_pix.rotate(gaussian_profile, theta)
    return jnp.squeeze(data)


def fit_morph_params(data, center, bx):
    """Fit a 2D Gaussian to the source.

    Parameters
    ----------
    data : jnp.ndarray 2D cutout of the source in single band
    center : tuple (int, int) central pixel of the source
    bx : int (boxsize)

    Returns
    -------
    best_params : jnp.ndarray best fit parameters for the 2D Gaussian

    """
    data_cut = (data - jnp.min(data)) / (jnp.max(data) - jnp.min(data))
    central_gaussian = create_gaussian_array(bx, 1.75, 1.75, 0)
    fn = lambda x: jnp.mean(
        central_gaussian * (data_cut - create_gaussian_array(bx, x[0], x[1], x[2])) ** 2
    )
    x0 = jnp.array([1.5, 1.5, 0])
    lbfgsb = ScipyBoundedMinimize(fun=fn, method="l-bfgs-b", jit=True)
    lower_bounds = jnp.array([1, 1, jnp.pi / 2])
    upper_bounds = jnp.array([7, 7, jnp.pi])
    bounds = (lower_bounds, upper_bounds)
    best_params = lbfgsb.run(x0, bounds=bounds).params
    return best_params


# take the 3 moments of a @D gaussian and construct the covariance matrix to fit the shape
def init_simple_morph(
    observation, center, psf_sigma=0.5, noise_thresh=20, corr_thresh=0.8
):
    """
    Initialize the morphology of a source by fitting a 2D Gaussian to the cutout of the source.
    The boxsize will initially be fit as a compact point-source and then expanded until the snr
    is below a desired threshold.

    Parameters
    ----------
    observation: `~scarlet.Observation`
    center: tuple --> center of source
    psf_sigma: float --> initial psf used in model
    noise_thresh: float --> threshold for noise
    corr_thresh: float --> threshold for spectrum correlations

    Returns
    -------
    morph: `jnp.array` --> 2D Gaussian morphology
    """
    sigma = psf_sigma
    converged = False
    peak_spec = init_spectrum(observation, center, correct_psf=True)
    spectra_prev = peak_spec
    noise_rms = 1 / np.sqrt(ma.masked_equal(observation.weights, 0))
    ma.set_fill_value(noise_rms, np.inf)

    # loop over box size until SNR is below threshold
    while converged == False:
        morph = GaussianMorphology(center, sigma=sigma)
        box = morph.bbox.shape

        # now grab the perimeter values of the box
        perimeter_pixels = [None] * len(observation.data)
        perimeter_noise_rms = [None] * len(observation.data)

        # create the cutout and noise rms for each band
        for idx, band in enumerate(observation.data):
            cutout_obs = cut_square_box(band, center, box[0])
            cutout_noise = cut_square_box(noise_rms[idx], center, box[0])
            perimeter_pixels[idx] = jnp.array([perimeter_values(cutout_obs)])
            perimeter_noise_rms[idx] = jnp.array([perimeter_values(cutout_noise)])
        perimeter_pixels = jnp.squeeze(jnp.stack(perimeter_pixels, axis=0), axis=1)
        perimeter_noise_rms = jnp.squeeze(
            jnp.stack(perimeter_noise_rms, axis=0), axis=1
        )

        # simplified spectrum calculation for perimeter average around cutout
        models = (None,) * len(observation.data)
        spectra = []
        spectrum_avg = 0
        for obs, model in zip((observation,), models):
            psf_model = obs.frame.psf()
            psf_peak = psf_model.max(axis=(1, 2))
            for i in range(perimeter_pixels.shape[1]):
                spectrum = perimeter_pixels[:, i]
                spectrum /= psf_peak
                spectrum_avg += spectrum
            spectrum_avg /= perimeter_pixels.shape[1]
            spectra.append(spectrum_avg)
        spectra_box = spectra[0]

        # now check the SNR and spectrum correlations
        noise_avg = jnp.mean(perimeter_noise_rms, axis=1) * noise_thresh
        snr = spectra_box / noise_avg
        spec_corr = jnp.dot(spectra_box, peak_spec) / jnp.dot(
            jnp.sqrt(jnp.dot(spectra_box, spectra_box)),
            jnp.sqrt(jnp.dot(peak_spec, peak_spec)),
        )

        # check if SNR is below threshold or spectrum correlation is below threshold
        if snr.all() < 1 or spec_corr < corr_thresh:
            converged = True
        # check if flux on boundaries increased with larger size
        elif (spectra_box >= peak_spec).any() or (spectra_box >= spectra_prev).any():
            converged = True
        # increase box/morpgology size
        else:
            spectra_prev = spectra_box
            sigma += 1
        # for debugging
        if sigma > 5:
            break

    # normalise the morphology between 0 and 1
    morph = morph()
    morph = (morph - jnp.min(morph)) / (jnp.max(morph) - jnp.min(morph))
    return morph


def init_morphology(
    obs,
    center,
    psf_sigma=1,
    noise_thresh=100,
    corr_thresh=0.8,
    max_size=32,
    components=1,
):
    """Initialize the morphology of the sources.
    Parameters
    ----------
    obs : `~scarlet.observation.Observation` instance
    center : tuple (int, int) central pixel of the sources

    Returns
    -------
    morph : np.ndarray morphology of the source normalised to (0,1)
    """
    morph = init_simple_morph(obs, center, psf_sigma, noise_thresh, corr_thresh)
    # TODO: here create descending box size morphologies, and check for the SNR
    # use the moments from .measure
    if morph.shape[0] <= max_size:
        return morph
    # fit for a more complex morphology for bigger sources
    else:
        bx = morph.shape[0]
        cutout = cut_square_box(obs.data[0], center, bx)
        sx, sy, th = fit_morph_params(cutout, center, bx)
        morph = create_gaussian_array(bx, sx, sy, th)
        morph = np.array(morph, dtype=np.float32)  # convert to numpy to make mutable
        rows, cols = morph.shape
        central_row = rows // 2
        central_col = cols // 2
        morph = (morph - np.min(morph)) / (np.max(morph) - np.min(morph))
        if bx > 30 and components == 2:
            morph2 = create_gaussian_array(
                bx, 1.25, 1.25, 0
            )  # create a second component as a gaussian blob
            morph2 = (morph2 - np.min(morph2)) / (np.max(morph2) - np.min(morph2))
            return [morph, morph2]
        else:
            # if single component just return the fitted morhpology
            return morph


# initialise the spectrum
def init_spectrum(observations, center, correct_psf=None):
    """Get the spectrum at center of observation.

    Yields the spectrum of a single-pixel source with flux 1 in every channel,
    concatenated for all observations.

    If `correct_psf`, it homogenizes the PSFs of the observations, which yields the
    correct spectrum for a flux=1 point source.

    If `model` is set, it reads of the value of the model at `sky_coord` and yields the
    spectrum for that model.

    Parameters
    ----------
    center: tuple
        Position in the observation
    observations: instance or list of `~scarlet.Observation`
        Observation to extract SED from.
    correct_psf: bool
        If PSF shape variations in the observations should be corrected.

    Returns
    -------
    spectrum: `~numpy.array` or list thereof
    """
    if not hasattr(observations, "__iter__"):
        single = True
        observations = (observations,)
        models = (None,) * len(observations)
    else:
        models = (None,) * len(observations)
        single = False

    spectra = []
    for obs, model in zip(observations, models):
        spectrum = obs.data[:, center[0], center[1]].copy()

        if correct_psf and obs.frame.psf is not None:
            # correct spectrum for PSF-induced change in peak pixel intensity
            psf_model = obs.frame.psf()
            psf_peak = psf_model.max(axis=(1, 2))
            spectrum /= psf_peak
        elif model is not None:
            model_value = model[:, center[0], center[1]].copy()
            spectrum /= model_value

        spectra.append(spectrum)

        if jnp.any(spectrum <= 0):
            # If the flux in all channels is  <=0,
            # the new sed will be filled with NaN values,
            # which will cause the code to crash later
            msg = f"Zero or negative spectrum {spectrum} at {center}"
            if np.all(spectrum <= 0):
                print("Zero or negative spectrum at all sources")
                print("Setting spectrum bands to 1")
                spectrum = np.ones_like(spectrum)
            else:
                print(msg)

    if single:
        return spectra[0]
    return spectra



def init_simple_morph_beta(
    observation, center, psf_sigma=0.5, noise_thresh=20, corr_thresh=0.8
):
    """
    Initialize the morphology of a source by fitting a 2D Gaussian to the cutout of the source.
    The boxsize will initially be fit as a compact point-source and then expanded until the snr
    is below a desired threshold.

    Parameters
    ----------
    observation: `~scarlet.Observation`
    center: tuple --> center of source
    psf_sigma: float --> initial psf used in model
    noise_thresh: float --> threshold for noise
    corr_thresh: float --> threshold for spectrum correlations

    Returns
    -------
    morph: `jnp.array` --> 2D Gaussian morphology
    """
    sigma = psf_sigma
    converged = False
    peak_spec = init_spectrum(observation, center, correct_psf=True)
    spectra_prev = peak_spec
    noise_rms = 1 / np.sqrt(ma.masked_equal(observation.weights, 0))
    ma.set_fill_value(noise_rms, np.inf)
    boxsize = 5 # initial boxsize

    # loop over box size until SNR is below threshold
    while converged == False:
        morph = GaussianMorphology(center, sigma=sigma)
        box = (boxsize, boxsize)

        # now grab the perimeter values of the box
        perimeter_pixels = [None] * len(observation.data)
        perimeter_noise_rms = [None] * len(observation.data)

        # create the cutout and noise rms for each band
        for idx, band in enumerate(observation.data):
            cutout_obs = cut_square_box(band, center, box[0])
            cutout_noise = cut_square_box(noise_rms[idx], center, box[0])
            perimeter_pixels[idx] = jnp.array([perimeter_values(cutout_obs)])
            perimeter_noise_rms[idx] = jnp.array([perimeter_values(cutout_noise)])
        perimeter_pixels = jnp.squeeze(jnp.stack(perimeter_pixels, axis=0), axis=1)
        perimeter_noise_rms = jnp.squeeze(
            jnp.stack(perimeter_noise_rms, axis=0), axis=1
        )
        F_e = jnp.mean(perimeter_pixels, axis=0) # average flux of the perimeter pixels

        # simplified spectrum calculation for perimeter average around cutout
        models = (None,) * len(observation.data)
        spectra = []
        spectrum_avg = 0
        for obs, model in zip((observation,), models):
            psf_model = obs.frame.psf()
            psf_peak = psf_model.max(axis=(1, 2))
            for i in range(perimeter_pixels.shape[1]):
                spectrum = perimeter_pixels[:, i]
                spectrum /= psf_peak
                spectrum_avg += spectrum
            spectrum_avg /= perimeter_pixels.shape[1]
            spectra.append(spectrum_avg)
        spectra_box = spectra[0]

        # getting Gaussian moments 
        g = moments(cutout_obs, center, box[0])

        # create the Gaussian morphology
        # size T = \sqrt{Q_11 + Q_22}
        # ellipticity = [Q_11 - Q_22 + 2iQ_12] / [Q_11 + Q_12 + 2\sqrt{Q_11Q_22 - Q_12^2}]
        T = np.sqrt( g[(2, 0)] + g[(0, 2)] )
        ellipticity = ( g[(2, 0)] - g[(0, 2)] + 2j*g[(1, 1)] ) / ( g[(2, 0)] + g[(1, 1)] + 2*np.sqrt(g[(2, 0)]*g[(0, 2)] - g[(1, 1)]**2) )

        # now check the SNR and spectrum correlations
        noise_avg = jnp.mean(perimeter_noise_rms, axis=1) * noise_thresh
        snr = spectra_box / noise_avg
        spec_corr = jnp.dot(spectra_box, peak_spec) / jnp.dot(
            jnp.sqrt(jnp.dot(spectra_box, spectra_box)),
            jnp.sqrt(jnp.dot(peak_spec, peak_spec)),
        )

        # check if SNR is below threshold or spectrum correlation is below threshold
        if snr.all() < 1 or spec_corr < corr_thresh:
            converged = True
        # check if flux on boundaries increased with larger size
        elif (spectra_box >= peak_spec).any() or (spectra_box >= spectra_prev).any():
            converged = True
        # increase box/morpgology size
        else:
            spectra_prev = spectra_box
            sigma += 1
        # for debugging
        if sigma > 5:
            break

        # if end of loop reached increase the box size
        boxsize += 10

    # normalise the morphology between 0 and 1
    morph = morph()
    morph = (morph - jnp.min(morph)) / (jnp.max(morph) - jnp.min(morph))
    return morph
