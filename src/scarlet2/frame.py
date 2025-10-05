import astropy.units as u
import astropy.wcs
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord

from .bbox import Box
from .psf import PSF, GaussianPSF


class Frame(eqx.Module):
    """Definition of a view of the sky

    This class combines all elements to determine how a piece of the sky will appear.
    It includes metadata about the spatial and spectral coverage and resolution.
    """

    bbox: Box
    """Bounding box of the frame"""
    psf: PSF = None
    """PSF of the frame"""
    wcs: astropy.wcs.WCS = None
    """WCS information of the frame"""
    channels: list
    """Identifiers for the spectral elements"""

    def __init__(self, bbox, psf=None, wcs=None, channels=None):
        self.bbox = bbox
        self.psf = psf
        self.wcs = wcs
        if channels is None:
            channels = list(range(bbox.shape[0]))

        self.channels = channels

    def __hash__(self):
        return hash(self.bbox)

    @property
    def C(self) -> int:  # noqa: N802
        """Number of channels"""
        return len(self.channels)

    @property
    def pixel_size(self):
        """Get the size of the pixels

        Returns
        -------
        float, astropy.units.quantity.Quantity
            Pixel size in units of the WCS sky coordinates
        """
        if self.wcs is not None:
            return get_scale(self.wcs)
        else:
            return 1

    def get_pixel(self, pos):
        """Get the sky coordinates from a world coordinate

        Parameters
        ----------
        pos: jnp.ndarray or SkyCoord
            Coordinates on the sky

        Returns
        ---------
        pixel coordinates in the model frame
        """
        if isinstance(pos, SkyCoord):
            assert self.wcs is not None, "SkyCoord can only be converted with valid WCS"
            wcs_ = self.wcs.celestial  # only use celestial portion
            pixel = jnp.asarray(pos.to_pixel(wcs_), dtype="float32").T
            return pixel[..., ::-1]
        return pos

    def get_sky_coord(self, pos):
        """Get the sky coordinate from a pixel coordinate

        Parameters
        ----------
        pos: jnp.ndarray
            Coordinates in the pixel space

        Returns
        ----------
        astropy.coordinates.SkyCoord if WCS is set, otherwise pos
        """
        if self.wcs is not None:
            pixels = pos.reshape(-1, 2)
            wcs = self.wcs.celestial  # only use celestial portion
            sky_coord = SkyCoord.from_pixel(pixels[:, 1], pixels[:, 0], wcs)
            return sky_coord
        return pos

    def convert_pixel_to(self, target, pixel=None):
        """Converts pixel coordinates from this frame to `target` frame

        Parameters
        ----------
        target: :py:class:`~scarlet2.Frame`
            target frame
        pixel: array
            Pixel coordinates in this frame. If not set, convert all pixels in this frame

        Returns
        -------
        array
            coordinates at the location of `pixel` in the frame `target`
        """

        if pixel is None:
            y, x = jnp.indices(self.bbox.spatial.shape, dtype="float32")
            pixel = jnp.stack((y.flatten(), x.flatten()), axis=1)

        ra_dec = self.get_sky_coord(pixel)
        return target.get_pixel(ra_dec)

    def u_to_pixel(self, distance):
        """Converts celestial distance to pixel size according to this frame WCS

        Parameters
        ----------
        distance: :py:class:`astropy.units.Quantity`
            Physical size, must be `PhysicalType("angle")`

        Returns
        -------
        float
            size in pixels
        """
        if self.wcs is not None:
            pixel_size = get_pixel_size(self.wcs)
            return distance / pixel_size
        else:
            return distance

    def pixel_to_angle(self, size):
        """Converts pixel size to celestial distance according to this frame WCS

        Parameters
        ----------
        size: float
            The size in pixels

        Returns
        -------
        distance: :py:class:`astropy.units.Quantity`
        """

        pixel_size = get_pixel_size(self.wcs)
        distance = size * pixel_size
        return distance

    @staticmethod
    def from_observations(observations, model_psf=None, model_wcs=None, obs_id=None, coverage="union"):
        """Generates a suitable model frame for a set of observations.

        This method generates a frame from a set of observations by identifying the highest resolution
        and the smallest PSF and use them to construct a common frame for all observations.

        Parameters
        ----------
        observations: list
            list of :py:class:`~scarlet2.Observation` to determine a common frame
        model_psf: :py:class:`~scarlet2.PSF`, optional
            PSF to be adopted for the model frame. This is the effective resolution
            of the model, and all observations are to be deconvolved to this limit.
            If None, uses the smallest PSF across all observations and channels.
        model_wcs: :py:class:`astropy.wcs.WCS`
            WCS for the model frame. If None, uses WCS information of the
            observation with the smallest pixels.
        obs_id: int, optional
            index of the reference observation.
            If set to None, uses the observation with the smallest pixels.
        coverage: "union" or "intersection"
            Sets the frame to incorporate the pixels covered by any observation
            ('union') or by all observations ('intersection').
        """
        assert coverage in ["union", "intersection"]

        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # Array of pixel sizes for each observation
        pix_tab = []
        # Array of psf size for each psf of each observation
        small_psf_size = None
        channels = []
        # Create frame channels and find smallest and largest psf
        for c, obs in enumerate(observations):
            # Concatenate all channels
            channels = channels + obs.frame.channels

            # concatenate all pixel sizes
            h_temp = get_pixel_size(obs.frame.wcs)
            if isinstance(h_temp, u.Quantity):
                h_temp = h_temp.to(u.arcsec).value  # standardize pixel sizes, using simple scalars below
            pix_tab.append(h_temp)

            # Looking for the sharpest PSF
            psf = obs.frame.psf.morphology
            for psf_channel in psf:
                psf_size = get_psf_size(psf_channel) * h_temp
                if (
                    model_psf is None
                    and ((obs_id is None) or (c == obs_id))
                    and ((small_psf_size is None) or (psf_size < small_psf_size))
                ):
                    small_psf_size = psf_size

        # Find a reference observation. Either provided by obs_id or as the
        # observation with the smallest pixel
        if obs_id is None:
            p = jnp.array(pix_tab)
            obs_ref = observations[jnp.where(p == p.min())[0][0]]
        else:
            # Frame defined from obs_id
            obs_ref = observations[obs_id]

        # Reference wcs
        if model_wcs is None:
            model_wcs = obs_ref.frame.wcs

        # Scale of the model pixel
        h = get_pixel_size(model_wcs)

        # If needed and psf is not provided: interpolate psf to smallest pixel
        if model_psf is None:
            # create Gaussian PSF with a sigma smaller than the smallest observed PSF
            sigma = 0.7
            assert (
                small_psf_size / h > sigma
            ), f"Default model PSF width ({sigma} pixel) too large for best-seeing observation"
            model_psf = GaussianPSF(sigma=sigma)

        # Dummy frame for WCS computations
        model_shape = (len(channels), 0, 0)
        model_frame = Frame(Box(model_shape), channels=channels, psf=model_psf, wcs=model_wcs)

        # Determine overlap of all observations in pixel coordinates of the model frame
        for c, obs in enumerate(observations):
            obs_coord = obs.frame.convert_pixel_to(model_frame)
            # round coordinate to nearest integer (use python, not jnp)
            minmax_int = lambda x: (int(f) for f in jnp.round(jnp.sort(x)[jnp.array([0, -1])]))  # noqa:E731
            y_min, y_max = minmax_int(obs_coord[:, 0])
            x_min, x_max = minmax_int(obs_coord[:, 1])

            # +1 because Box.shape is a length, not a coordinate
            this_box = Box.from_bounds((y_min, y_max + 1), (x_min, x_max + 1))
            if c == 0:
                model_box = this_box
            else:
                if coverage == "union":
                    model_box |= this_box
                else:
                    model_box &= this_box

        frame_shape = (len(channels),) + model_box.shape
        frame_origin = (0,) + model_box.origin
        # TODO: update model_wcs to change NAXIS1/2 and CRPIX1/2, but don't change frame_origin!
        model_frame = Frame(
            Box(shape=frame_shape, origin=frame_origin), channels=channels, psf=model_psf, wcs=model_wcs
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
    psf: `scarlet.PSF`
        PSF for which to compute the size

    Returns
    -------
    sigma3: `float`
        radius of the area inside 3 sigma around the center in pixels
    """
    # Normalisation by maximum
    psf_frame = psf / jnp.max(psf)

    # Pixels in the FWHM set to one, others to 0:
    psf_frame = jnp.where(psf_frame > 0.5, 1.0, 0.0)

    # Area in the FWHM:
    area = jnp.sum(psf_frame)

    # Diameter of this area
    d = 2 * (area / jnp.pi) ** 0.5

    # 3-sigma:
    sigma3 = 3 * d / (2 * (2 * jnp.log(2)) ** 0.5)

    return sigma3


def get_affine(wcs):
    """Return the WCS transformation matrix"""
    if wcs is None:
        return jnp.diag(jnp.ones(2))
    wcs_ = wcs.celestial
    try:
        model_affine = wcs_.wcs.pc
    except AttributeError:
        try:
            model_affine = wcs_.cd
        except AttributeError:
            model_affine = wcs_.wcs.cd
    return model_affine[:2, :2]


# for WCS linear matrix calculations:
# rotation matrix for counter-clockwise rotation from positive x-axis
# uses (x,y) coordinates and phi in radian!!
def _rot_matrix(phi):
    sinphi, cosphi = jnp.sin(phi), jnp.cos(phi)
    return jnp.array([[cosphi, -sinphi], [sinphi, cosphi]])


# flip in y!!!
# uses (x,y) coordinates!
_flip_matrix = lambda flip: jnp.diag(jnp.array((1.0, flip)))

# 2x2 matrix determinant
_det = lambda m: m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]


def get_scale_angle_flip(trans):
    """Return, scale, angle, flip from the WCS transformation matrix

    Parameters
    ----------
    trans: (`astropy.wcs.WCS`, array)
        WCS or WCS transformation matrix

    Returns
    -------
    scale: `float`
    angle: `float`, in radian
    flip: -1 or 1
    """
    if isinstance(trans, (np.ndarray, jnp.ndarray)):  # noqa: SIM108
        M = trans  # noqa: N806
    else:
        M = get_affine(trans)  # noqa: N806

    det = _det(M)
    # this requires pixels to be square
    # if not, use scale = jnp.linalg.svd(M, compute_uv=False)
    # but be careful with rotations as anisotropic stretch and rotation do not commute
    scale = jnp.sqrt(jnp.abs(det)).item(0)

    # if rotation is improper: need to apply y-flip to M to get pure rotation matrix (and unique angle)
    improper = det < 0
    flip = -1 if improper else 1
    F = _flip_matrix(flip)  # noqa: N806, flip in y, is identity if flip = 1!!!
    M_ = F @ M  # noqa: N806, flip = inverse flip
    angle = jnp.arctan2(M_[1, 0], M_[0, 0])

    return scale, angle, flip


def get_pixel_size(wcs):
    """Extracts the pixel size from a wcs, and returns it in deg/pixel

    Parameters
    ----------
    wcs: `astropy.wcs.WCS`
        WCS structure or transformation matrix

    Returns
    -------
    pixel_size: `float`
    """
    scale, angle, flip = get_scale_angle_flip(wcs)
    return scale


def get_scale(wcs, separate=False):
    """
    Get WCS axis scales in deg/pixel

    Parameters
    ----------
    wcs: `astropy.wcs.WCS`
        WCS structure or transformation matrix
    separate: `bool`
          Compute separate axis scales

    Returns
    -------
    float
    """
    if separate:
        M = get_affine(wcs)  # noqa: N806
        c1 = (M[0, :] ** 2).sum() ** 0.5
        c2 = (M[1, :] ** 2).sum() ** 0.5
        return jnp.array([c1, c2])
    else:
        scale, angle, flip = get_scale_angle_flip(wcs)
        return scale


def get_angle(wcs):
    """
    Get WCS rotation angle

    The angle is computed counter-clockwise from the positive x-axis, in radians.

    Parameters
    ----------
    wcs: `astropy.wcs.WCS`
        WCS structure or transformation matrix

    Returns
    -------
    `astropy.units.quantity.Quantity`, unit = u.rad
    """
    scale, angle, flip = get_scale_angle_flip(wcs)
    return angle


def get_flip(wcs):
    """
    Return WCS sign convention

    A negative sign means that the rotation is improper and requires a flip.
    By convention, we define this to be a flip in the y-axis.

    Parameters
    ----------
    wcs: `astropy.wcs.WCS`
        WCS structure or transformation matrix

    Returns
    -------
    -1 or 1
    """
    scale, angle, flip = get_scale_angle_flip(wcs)
    return flip
