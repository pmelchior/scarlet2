import warnings
warnings.filterwarnings('ignore')

import numpy as np
from numpy.testing import assert_allclose
import scarlet2

import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS

import jax.numpy as jnp

# Load the HSC image data
obs_hdu = fits.open('../scarlet-test-data/data/test_resampling/Cut_HSC1.fits')
data_hsc = jnp.array(obs_hdu[0].data.byteswap().newbyteorder(), jnp.float32)
wcs_hsc = WCS(obs_hdu[0].header)
channels_hsc = ['g','r','i','z','y']

# Load the HSC PSF data
psf_hsc_data = fits.open('../scarlet-test-data/data/test_resampling/PSF_HSC.fits')[0].data.astype('float32')
Np1, Np2 = psf_hsc_data[0].shape
psf_hsc_data = jnp.pad(psf_hsc_data, ((0,0), (1,0), (1,0)))
psf_hsc = scarlet2.ArrayPSF(psf_hsc_data)

# Load the HST image data
hst_hdu = fits.open('../scarlet-test-data/data/test_resampling/Cut_HST1.fits')
data_hst = hst_hdu[0].data
wcs_hst = WCS(hst_hdu[0].header)
channels_hst = ['F814W']

# Load the HST PSF data
psf_hst = fits.open('../scarlet-test-data/data/test_resampling/PSF_HST.fits')[0].data
psf_hst = np.array(psf_hst[None,:,:], np.float32)
psf_hst = jnp.pad(psf_hst, ((0,0), (1,0), (1,0)))
psf_hst = np.repeat(psf_hst, 5, 0)

psf_hst = scarlet2.ArrayPSF(psf_hst)

# Scale the HST data
n1,n2 = np.shape(data_hst)
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
    out_galsim = np.load('../scarlet-test-data/galsim_hst_to_hsc_resolution.npy')

    assert_allclose(out_galsim, hst_resampled[0], atol=1.3e-4)