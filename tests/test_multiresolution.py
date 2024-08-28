import os
import warnings
warnings.filterwarnings('ignore')

from numpy.testing import assert_allclose

import scarlet2
from utils import import_scarlet_test_data
import_scarlet_test_data()
from scarlet_test_data import data_path, tests_path

import astropy.io.fits as fits
from astropy.wcs import WCS

import jax
import jax.numpy as jnp

# Load the HSC image data
obs_hdu = fits.open(os.path.join(data_path, "test_resampling", "Cut_HSC1.fits"))
                    
data_hsc = jnp.array(obs_hdu[0].data.byteswap().newbyteorder(), jnp.float32)
wcs_hsc = WCS(obs_hdu[0].header)
channels_hsc = ['g','r','i','z','y']

# Load the HSC PSF data
psf_hsc_data  = fits.open(os.path.join(data_path, "test_resampling", "PSF_HSC.fits"))[0].data.astype('float32')
                    
Np1, Np2 = psf_hsc_data[0].shape
psf_hsc_data = jnp.pad(psf_hsc_data, ((0,0), (1,0), (1,0)))
psf_hsc = scarlet2.ArrayPSF(psf_hsc_data)

# Load the HST image data
hst_hdu = fits.open(os.path.join(data_path, "test_resampling", "Cut_HST1.fits"))

data_hst = hst_hdu[0].data
wcs_hst = WCS(hst_hdu[0].header)
channels_hst = ['F814W']

# Load the HST PSF data
psf_hst = fits.open(os.path.join(data_path, "test_resampling", "PSF_HST.fits"))[0].data
psf_hst = jnp.array(psf_hst[None, :, :], jnp.float32)
psf_hst = jnp.pad(psf_hst, ((0, 0), (1, 0), (1, 0)))
psf_hst_ = jnp.repeat(psf_hst, 5, 0)

psf_hst = scarlet2.ArrayPSF(psf_hst_)

# Scale the HST data
n1, n2 = jnp.shape(data_hst)
data_hst = data_hst.reshape(1, n1, n2).byteswap().newbyteorder()
data_hst *= data_hsc.max() / data_hst.max()

r, N1, N2 = data_hsc.shape

# define two observation packages and match to frame
obs_hst = scarlet2.Observation(data_hst[:1, ...],
                               wcs=wcs_hst,
                               psf=psf_hst,
                               channels=['channel'],
                               weights=None)

obs_hsc = scarlet2.Observation(data_hsc[:1,...],
                              wcs=wcs_hsc,
                              psf=psf_hsc,
                              channels=['channel'],
                              weights=None)

# Building a Frame from hst obs

hst_frame = scarlet2.Frame(
                bbox=scarlet2.Box(shape=data_hst.shape),
                channels=['channel'],
                psf=psf_hst,
                wcs=wcs_hst
)

def test_hst_to_hsc_against_galsim():

    # Initializing renderers
    obs_hsc.match(hst_frame)

    # Deconvolution, Resampling and Reconvolution
    hst_resampled = obs_hsc.render(data_hst)

    # Perform the same operations with galsim 
    out_galsim = jnp.load(os.path.join(tests_path, "galsim_hst_to_hsc_resolution.npy"))

    assert_allclose(out_galsim, hst_resampled[0], atol=1.3e-4)

def test_hst_to_hsc_against_galsim_rotated_wcs():
    # Remember images coordinates are [y, x]
    # Update CRPIX for the 90-degree clockwise rotation
    crpix1_new = n1 - wcs_hst.wcs.crpix[1]
    crpix2_new = wcs_hst.wcs.crpix[0]
    wcs_hst.wcs.crpix = [crpix1_new, crpix2_new]

    # # Mock a rotation of 90 deg clockwise of the HST WCS 
    phi = -90 / 180 * jnp.pi # in rad
    R = jnp.array([[jnp.cos(phi), jnp.sin(phi)],
                    [-jnp.sin(phi), jnp.cos(phi)]])

    data_hst_ = jax.vmap(jnp.rot90)(data_hst)
    psf_hst = jax.vmap(jnp.rot90)(psf_hst_)
    psf_hst = scarlet2.ArrayPSF(psf_hst)

    wcs_hst.wcs.pc = R @ wcs_hst.wcs.pc    

    hst_frame = scarlet2.Frame(
                bbox=scarlet2.Box(shape=data_hst_.shape),
                channels=['channel'],
                psf=psf_hst,
                wcs=wcs_hst
    )

    # Automatically find the difference between observation and model WCSs
    obs_hsc.match(hst_frame)
    
    # Deconvolution, Resampling and Reconvolution
    hst_resampled = obs_hsc.render(data_hst_)

    # Perform the same operations with galsim 
    out_galsim = jnp.load(os.path.join(tests_path, "galsim_hst_to_hsc_resolution.npy"))

    assert_allclose(out_galsim, hst_resampled[0], atol=1.3e-4)

if __name__=="__main__":
    test_hst_to_hsc_against_galsim()
    test_hst_to_hsc_against_galsim_rotated_wcs()