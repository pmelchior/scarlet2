# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103
# ruff: noqa: D106

import warnings

import astropy.io.fits as fits
import astropy.units as u
import jax
import jax.numpy as jnp
import scarlet2
from astropy.wcs import WCS
from huggingface_hub import hf_hub_download
from scarlet2.frame import _rot_matrix, get_affine
from scarlet2.validation_utils import set_validation

warnings.filterwarnings("ignore")

# turn off automatic validation checks
set_validation(False)

# Load the HSC image data
# load test data from HSC and HST
filename = hf_hub_download(
    repo_id="astro-data-lab/scarlet-test-data",
    filename="test_resampling/Cut_HSC1.fits.gz",
    repo_type="dataset",
)
obs_hdu = fits.open(filename)
data_hsc = jnp.array(obs_hdu[0].data, jnp.float32)
wcs_hsc = WCS(obs_hdu[0].header)
channels_hsc = ["g", "r", "i", "z", "y"]
obs_hdu.close()

# Load the HSC PSF data
filename = hf_hub_download(
    repo_id="astro-data-lab/scarlet-test-data",
    filename="test_resampling/PSF_HSC.fits.gz",
    repo_type="dataset",
)
psf_hsc_data = jnp.array(fits.open(filename)[0].data, jnp.float32)
Np1, Np2 = psf_hsc_data[0].shape
psf_hsc_data = jnp.pad(psf_hsc_data, ((0, 0), (1, 0), (1, 0)))
psf_hsc_single_band_ = psf_hsc_data[:1]
psf_hsc = scarlet2.ArrayPSF(psf_hsc_data)
psf_hsc_single_band = scarlet2.ArrayPSF(psf_hsc_single_band_)

# Load the HST image data
filename = hf_hub_download(
    repo_id="astro-data-lab/scarlet-test-data",
    filename="test_resampling/Cut_HST1.fits.gz",
    repo_type="dataset",
)
obs_hdu = fits.open(filename)
data_hst = jnp.array(obs_hdu[0].data, jnp.float32)
wcs_hst = WCS(obs_hdu[0].header)
channels_hst = ["F814W"]
obs_hdu.close()

# Load the HST PSF data
filename = hf_hub_download(
    repo_id="astro-data-lab/scarlet-test-data",
    filename="test_resampling/PSF_HST.fits.gz",
    repo_type="dataset",
)
psf_hst = jnp.array(fits.open(filename)[0].data, jnp.float32)
psf_hst = psf_hst[None, :, :]
psf_hst = jnp.pad(psf_hst, ((0, 0), (1, 0), (1, 0)))
psf_hst_ = jnp.repeat(psf_hst, 5, 0)
psf_hst_single_band_ = psf_hst_[:1]
psf_hst = scarlet2.ArrayPSF(psf_hst_)
psf_hst_single_band = scarlet2.ArrayPSF(psf_hst_single_band_)

# Scale the HST data
n1, n2 = jnp.shape(data_hst)
data_hst = data_hst.reshape(1, n1, n2)
data_hst *= data_hsc.max() / data_hst.max()

r, N1, N2 = data_hsc.shape

# define two observation packages and match to frame
obs_hst = scarlet2.Observation(
    data_hst[:1, ...], wcs=wcs_hst, psf=psf_hst_single_band, channels=["channel"], weights=None
)

obs_hsc = scarlet2.Observation(
    data_hsc[:1, ...], wcs=wcs_hsc, psf=psf_hsc_single_band, channels=["channel"], weights=None
)

# Building a Frame from hst obs

hst_frame = scarlet2.Frame(
    bbox=scarlet2.Box(shape=data_hst.shape), channels=["channel"], psf=psf_hst, wcs=wcs_hst
)

# compare to galsim
# Perform the same operations with galsim
filename = hf_hub_download(
    repo_id="astro-data-lab/scarlet-test-data",
    filename="test_resampling/galsim_hst_to_hsc_resolution.npy",
    repo_type="dataset",
)
out_galsim = jnp.load(filename)


def test_hst_to_hsc_against_galsim():
    # Initializing renderers
    obs_hsc.match(hst_frame)

    # Deconvolution, Resampling and Reconvolution
    hst_resampled = obs_hsc.render(data_hst)

    assert jnp.allclose(out_galsim, hst_resampled[0], atol=2.3e-4)


# Rotate WCS
# Remember images coordinates are [y, x]
# Update CRPIX for the 90-degree rotation
wcs_hst_rot = wcs_hst.deepcopy()
crpix1_new = n1 - wcs_hst.wcs.crpix[1]
crpix2_new = wcs_hst.wcs.crpix[0]
wcs_hst_rot.wcs.crpix = [crpix1_new, crpix2_new]

# # Mock a rotation of 90 deg counter-clockwise of the HST WCS
M = get_affine(wcs_hst)
phi = 90 * u.deg
R = _rot_matrix(phi)
if jnp.linalg.det(M) < 0:  # improper rotation: flip y, rotate, flip back
    F = jnp.diag(jnp.array((1, -1)))
    R = F @ R @ F  # F = F^-1
# rotate first, then go the sky coordinates
wcs_hst_rot.wcs.pc = M @ R

# rotate data and psf counter-clockwise 90 deg
# need to change axes order because image rotation is defined from positive x-axis (axis=1)
rot90_image = lambda im: jnp.rot90(im, axes=(1, 0))
data_hst_rot = jax.vmap(rot90_image)(data_hst)
psf_hst_rot = jax.vmap(rot90_image)(psf_hst_)
psf_hst_rot = scarlet2.ArrayPSF(psf_hst_rot)

hst_frame_rot = scarlet2.Frame(
    bbox=scarlet2.Box(shape=data_hst_rot.shape), channels=["channel"], psf=psf_hst_rot, wcs=wcs_hst_rot
)


def test_hst_to_hsc_against_galsim_rotated_wcs():
    obs_hsc = scarlet2.Observation(
        data_hsc[:1, ...], wcs=wcs_hsc, psf=psf_hsc_single_band, channels=["channel"], weights=None
    )

    # Automatically find the difference between observation and model WCSs
    obs_hsc.match(hst_frame_rot)

    # Deconvolution, Resampling and Reconvolution
    hst_resampled = obs_hsc.render(data_hst_rot)

    assert jnp.allclose(out_galsim, hst_resampled[0], atol=2.3e-4)


if __name__ == "__main__":
    test_hst_to_hsc_against_galsim()
    test_hst_to_hsc_against_galsim_rotated_wcs()
