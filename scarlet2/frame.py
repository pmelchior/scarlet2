import astropy.wcs.wcs
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u

from .bbox import Box
from .psf import PSF

class Frame(eqx.Module):
    bbox: Box
    psf: PSF = None
    wcs: astropy.wcs.wcs = None
    channels: list

    def __init__(self, bbox, psf=None, wcs=None, channels=None):
        self.bbox = bbox
        self.psf = psf
        self.wcs = wcs
        if channels is None:
            channels = list(range(bbox.shape[0]))
        assert len(channels) == bbox.shape[0]

        self.channels = channels

    def __hash__(self):
        return hash(self.bbox)

    @property
    def C(self):
        return len(self.channels)

    @property
    def pixel_size(self):
        if self.wcs is not None:
            return get_pixel_size(get_affine(self.wcs)) * 60 * 60  # in arcsec
        else:
            return 1

    def get_pixel(self, sky_coord):
        """Get the sky coordinates from a world coordinate

        Parameters
        ----------
        sky_coord: SkyCoord, or list of Skycoords
            Coordinates on the sky

        Returns
        ---------
        pixel coordinates in the model frame
        """
        assert self.wcs is not None
        wcs_ = self.wcs.celestial # only use celestial portion
        
        pixel = jnp.asarray(sky_coord.to_pixel(wcs_), dtype="float32").T

        return pixel
    
    def get_sky_coord(self, pixels):
        """Get the sky coordinate from a pixel coordinate

        Parameters
        ----------
        pixel: array of pixel coordinates 
            Coordinates in the pixel space

        Returns
        ----------
        astropy.coordinates.SkyCoord
        """
        pixels = pixels.reshape(-1, 2)

        assert self.wcs is not None
        wcs = self.wcs.celestial # only use celestial portion
        sky_coord = SkyCoord.from_pixel(pixels[:,0], pixels[:,1], wcs)

        return sky_coord
    
    def convert_pixel_to(self, target, pixel=None):
        """Converts pixel coordinates from this frame to `target` Frame

            Parameters
            ----------
            target: `~scarlet2.Frame`
                target frame
            pixel: `array`, pixel coordinates in this frame
                If not set, convert all pixels in this frame

            Returns
            -------
            coord_target: `array`
                coordinates at the location of `coord` in the target frame
        """

        if pixel is None:
            y, x = jnp.indices(self.bbox.shape[-2:], dtype="float32")
            pixel = jnp.stack((y.flatten(), x.flatten()), axis=1)

        ra_dec = self.get_sky_coord(pixel)
        return target.get_pixel(ra_dec)
    
    def u_to_pixel(self, size):
        """Converts a size un astropy.units.Quantity to pixel size according 
           to this frame WCS

            Parameters
            ----------
            size: `astropy.units.Quantity`, must be PhysicalType("angle")

            Returns
            -------
            size in pixels
        """
        assert u.get_physical_type(size) == "angle"

        # first computer the pixel size
        pixel_size = get_pixel_size(
            get_affine(self.wcs.celestial) # only use celestial portion
        ) * 60 * 60 # in arcsec/pixel
        
        return size.to(u.arcsec).value / pixel_size

    @staticmethod
    def from_observations(
        observations, model_psf=None, model_wcs=None, obs_id=None, coverage="union"
    ):
        """Generates a suitable model frame for a set of observations.

        This method generates a frame from a set of observations by identifying the highest resolution
        and the smallest PSF and use them to construct a common frame for all observations.

        Parameters
        ----------
        observations: array of `scarlet.Observation` objects
            array that contains Observations to match onto a common frame
        model_psf: `scarlet.PSF`
            PSF of the model frame, to which all observations are to be deconvolved.
            If None, uses the smallest PSF across all observations and channels.
        model_wcs: `astropy.wcs.WCS`
            WCS for the model frame. If None, uses transformation of the observation
            with the smallest pixels.
        obs_id: int
            index of the reference observation
            If set to None, uses the observation with the smallest pixels.
        coverage: "union" or "intersection"
            Sets the frame to incorporate the pixels covered by any observation ('union')
            or by all observations ('intersection').
        """
        assert coverage in ["union", "intersection"]

        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # Array of pixel sizes for each observation
        pix_tab = []
        # Array of psf size for each psf of each observation
        fat_psf_size = None
        small_psf_size = None
        channels = []
        # Create frame channels and find smallest and largest psf
        for c, obs in enumerate(observations):

            # Concatenate all channels
            channels = channels + obs.frame.channels

            # concatenate all pixel sizes
            h_temp = get_pixel_size(get_affine(obs.frame.wcs))

            pix_tab.append(h_temp)
            # Looking for the sharpest and the fatest psf
            psf = obs.frame.psf.morphology
            for psf in psf:
                psf_size = get_psf_size(psf) * h_temp
                if (fat_psf_size is None) or (psf_size > fat_psf_size):
                    fat_psf_size = psf_size
                if (obs_id is None) or (c == obs_id):
                    if (model_psf is None) and (
                        (small_psf_size is None) or (psf_size < small_psf_size)
                    ):
                        small_psf_size = psf_size
                        model_psf_temp = ImagePSF(psf[np.newaxis, :, :])
                        psf_h = h_temp

        # Find a reference observation. Either provided by obs_id or as the observation with the smallest pixel
        if obs_id is None:
            obs_ref = observations[np.where(pix_tab == np.min(pix_tab))[0][0]]
        else:
            # Frame defined from obs_id
            obs_ref = observations[obs_id]

        # Reference wcs
        if model_wcs is None:
            model_wcs = obs_ref.frame.wcs

        # Scale of the smallest pixel
        h = get_pixel_size(get_affine(model_wcs))

        # TODO:
        # # If needed and psf is not provided: interpolate psf to smallest pixel
        # if model_psf is None:
        #     # If the reference PSF is not at the highest pixel resolution, make it!
        #     if psf_h > h:
        #         angle, h = interpolation.get_angles(model_wcs, obs.wcs)
        #         model_psf = PSF(
        #             interpolation.sinc_interp_inplace(model_psf_temp, psf_h, h, angle)
        #         )
        #     else:
        #         model_psf = model_psf_temp

        # Dummy frame for WCS computations
        model_shape = (len(channels), 0, 0)

        model_frame = Frame(
            Box(model_shape), channels=channels, psf=model_psf, wcs=model_wcs
        )

        # Determine overlap of all observations in pixel coordinates of the model frame
        for c, obs in enumerate(observations):

            if model_frame.wcs is obs.frame.wcs:
                this_box = obs_ref.frame.bbox[-2:]
            else:
                obs_coord = obs.frame.convert_pixel_to(model_frame)
                y_min = np.floor(np.min(obs_coord[:, 0])).astype("int")
                x_min = np.floor(np.min(obs_coord[:, 1])).astype("int")
                y_max = np.ceil(np.max(obs_coord[:, 0])).astype("int")
                x_max = np.ceil(np.max(obs_coord[:, 1])).astype("int")
                this_box = Box.from_bounds((y_min, y_max + 1), (x_min, x_max + 1))

            if c == 0:
                model_box = this_box
            else:
                if coverage == "union":
                    model_box |= this_box
                else:
                    model_box &= this_box

        # pad by the size of the widest psf to prevent leakage across the frame edge
        ny, nx = model_box.shape
        pad_size = fat_psf_size / h / 2
        offset = (np.round(pad_size).astype("int"), np.round(pad_size).astype("int"))
        model_box -= offset

        model_box_shape = tuple(s + 2 * o for s, o in zip(model_box.shape, offset))

        # move the reference pixel of the model wcs to the 0/0 pixel of the new shape
        model_wcs = model_wcs.deepcopy()
        model_wcs.wcs.crpix -= model_box.origin
        model_wcs.array_shape = model_box.shape

        # recreate the model frame with the correct shape
        # frame_shape = (len(channels), model_box_shape)
        frame_shape = np.concatenate([[len(channels)], np.array(model_box_shape)])

        model_frame = Frame(
            Box(frame_shape), channels=channels, psf=model_psf, wcs=model_wcs
        )

        # Match observations to this frame
        for obs in observations:
            obs.match(model_frame)

        return model_frame


def get_psf_size(psf):
    """Measures the size of a psf by computing the size of the area in 3 sigma around the center.

    This is an approximate method to estimate the size of the psf for setting the size of the frame,
    which does not require a precise measurement.

    Parameters
    ----------
        PSF: `scarlet.PSF` object
            PSF for whic to compute the size
    Returns
    -------
        sigma3: `float`
            radius of the area inside 3 sigma around the center in pixels
    """
    # Normalisation by maximum
    psf_frame = psf / np.max(psf)

    # Pixels in the FWHM set to one, others to 0:
    psf_frame[psf_frame > 0.5] = 1
    psf_frame[psf_frame <= 0.5] = 0

    # Area in the FWHM:
    area = np.sum(psf_frame)

    # Diameter of this area
    d = 2 * (area / np.pi) ** 0.5

    # 3-sigma:
    sigma3 = 3 * d / (2 * (2 * np.log(2)) ** 0.5)

    return sigma3

def get_affine(wcs):
    try:
        model_affine = wcs.wcs.pc
    except AttributeError:
        try:
            model_affine = wcs.cd
        except AttributeError:
            model_affine = wcs.wcs.cd
    return model_affine


def get_pixel_size(model_affine):
    """Extracts the pixel size from a wcs, and returns it in deg/pixel"""
    pix = np.sqrt(
        np.abs(model_affine[0, 0])
        * np.abs(model_affine[1, 1] - model_affine[0, 1] * model_affine[1, 0])
    )
    return pix
