# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103
# ruff: noqa: D106

import warnings

import astropy.io.fits as fits
import jax
import jax.numpy as jnp
import scarlet2
from astropy.wcs import WCS
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

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
psf_hsc = scarlet2.ArrayPSF(psf_hsc_data)

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
psf_hst = scarlet2.ArrayPSF(psf_hst_)

# Scale the HST data
n1, n2 = jnp.shape(data_hst)
data_hst = data_hst.reshape(1, n1, n2)
data_hst *= data_hsc.max() / data_hst.max()

r, N1, N2 = data_hsc.shape

# define two observation packages and match to frame
obs_hst = scarlet2.Observation(
    data_hst[:1, ...], wcs=wcs_hst, psf=psf_hst, channels=["channel"], weights=None
)

obs_hsc = scarlet2.Observation(
    data_hsc[:1, ...], wcs=wcs_hsc, psf=psf_hsc, channels=["channel"], weights=None
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

    assert jnp.allclose(out_galsim, hst_resampled[0], atol=1.3e-4)


# Rotate WCS
# Remember images coordinates are [y, x]
# Update CRPIX for the 90-degree clockwise rotation
wcs_hst_rot = wcs_hst.deepcopy()
crpix1_new = n1 - wcs_hst.wcs.crpix[1]
crpix2_new = wcs_hst.wcs.crpix[0]

wcs_hst_rot.wcs.crpix = [crpix1_new, crpix2_new]

# # Mock a rotation of 90 deg clockwise of the HST WCS
phi = -90 / 180 * jnp.pi  # in rad
R = jnp.array([[jnp.cos(phi), jnp.sin(phi)], [-jnp.sin(phi), jnp.cos(phi)]])

data_hst_rot = jax.vmap(jnp.rot90)(data_hst)
psf_hst_rot = jax.vmap(jnp.rot90)(psf_hst_)
psf_hst_rot = scarlet2.ArrayPSF(psf_hst_rot)

wcs_hst_rot.wcs.pc = R @ wcs_hst.wcs.pc

hst_frame_rot = scarlet2.Frame(
    bbox=scarlet2.Box(shape=data_hst_rot.shape), channels=["channel"], psf=psf_hst_rot, wcs=wcs_hst_rot
)

assert wcs_hst_rot != wcs_hst


def test_hst_to_hsc_against_galsim_rotated_wcs():
    obs_hsc = scarlet2.Observation(
        data_hsc[:1, ...], wcs=wcs_hsc, psf=psf_hsc, channels=["channel"], weights=None
    )

    # Automatically find the difference between observation and model WCSs
    obs_hsc.match(hst_frame_rot)

    # Deconvolution, Resampling and Reconvolution
    hst_resampled = obs_hsc.render(data_hst_rot)

    assert jnp.allclose(out_galsim, hst_resampled[0], atol=1.3e-4)


def test_no_channel_axis_in_obs_psf():
    assert len(psf_hsc_data[0].shape) == 2
    obs_hsc = scarlet2.Observation(
        data_hsc[:1, ...],
        wcs=wcs_hsc,
        psf=scarlet2.ArrayPSF(psf_hsc_data[0]),
        channels=["channel"],
        weights=None,
    )

    # Initializing renderers
    obs_hsc.match(hst_frame)

    # Deconvolution, Resampling and Reconvolution
    hst_resampled = obs_hsc.render(data_hst)

    assert jnp.allclose(out_galsim, hst_resampled[0], atol=1.3e-4)


if __name__ == "__main__":
    test_hst_to_hsc_against_galsim()
    test_hst_to_hsc_against_galsim_rotated_wcs()
    test_no_channel_axis_in_obs_psf()
