import astropy.units as u
import astropy.wcs
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord

from .bbox import Box
from .psf import PSF, ArrayPSF, GaussianPSF


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
        if isinstance(psf, jnp.ndarray):
            psf = ArrayPSF(psf)
        self.psf = psf
        if wcs is None:
            wcs = _wcs_default(bbox.spatial.shape)
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
        return get_scale(self.wcs)

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
            # WCS uses x/y convention
            wcs = self.wcs.celestial  # only use celestial portion
            x, y = pos.to_pixel(wcs)
            pos = jnp.stack([y, x], axis=-1)  # convert to y/x
        return pos

    def get_sky_coord(self, pos):
        """Get the sky coordinate from a pixel coordinate

        Parameters
        ----------
        pos: jnp.ndarray
            Coordinates in the pixel space

        Returns
        ----------
        astropy.coordinates.SkyCoord
        """
        wcs = self.wcs.celestial  # only use celestial portion
        if jnp.ndim(pos) > 1:
            pos = jnp.asarray(pos).reshape(-1, 2).T.reshape(2, *jnp.shape(pos)[:-1])
        sky_coord = SkyCoord.from_pixel(pos[1], pos[0], wcs)
        return sky_coord

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
    def from_observations(observations, model_psf=None, model_wcs=None, reference_id=None, coverage="union"):
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
        reference_id: int, optional
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
                    and ((reference_id is None) or (c == reference_id))
                    and ((small_psf_size is None) or (psf_size < small_psf_size))
                ):
                    small_psf_size = psf_size

        # Find a reference observation. Either provided by obs_id or as the
        # observation with the smallest pixel
        if reference_id is None:
            p = jnp.array(pix_tab)
            obs_ref = observations[jnp.where(p == p.min())[0][0]]
        else:
            # Frame defined from obs_id
            obs_ref = observations[reference_id]

        # Reference wcs
        if model_wcs is None:
            model_wcs = obs_ref.frame.wcs.deepcopy()

        # Scale of the model pixel
        h = get_pixel_size(model_wcs)

        # If needed and psf is not provided: interpolate psf to smallest pixel
        if model_psf is None:
            # create Gaussian PSF with a sigma smaller than the smallest observed PSF
            sigma = 0.7
            assert small_psf_size / h > sigma, (
                f"Default model PSF width ({sigma} pixel) too large for best-seeing observation"
            )
            model_psf = GaussianPSF(sigma=sigma)

        # Dummy frame for WCS computations
        model_shape = (len(channels), 0, 0)
        model_frame = Frame(Box(model_shape), channels=channels, psf=model_psf, wcs=model_wcs)

        # Determine overlap of all observations in pixel coordinates of the model frame
        for c, obs in enumerate(observations):
            obs_coord = obs.frame.convert_pixel_to(model_frame)
            y_min, y_max = _minmax_int(obs_coord[:, 0])
            x_min, x_max = _minmax_int(obs_coord[:, 1])

            # +1 because Box.shape is a length, not a coordinate
            this_box = Box.from_bounds((y_min, y_max + 1), (x_min, x_max + 1))
            if c == 0:
                model_box = this_box
            else:
                if coverage == "union":
                    model_box |= this_box
                else:
                    model_box &= this_box

        # update model_wcs to change NAXIS1/2 and CRPIX1/2, but don't change frame_origin!
        model_wcs._naxis = list(model_wcs._naxis)
        model_wcs._naxis[:2] = model_box.shape[::-1]  # x/y needed here
        model_wcs.wcs.crpix[:2] -= model_box.origin[::-1]  # x/y needed here

        # frame_origin = (0,) + model_box.origin
        frame_shape = (len(channels),) + model_box.shape
        model_frame = Frame(Box(shape=frame_shape), channels=channels, psf=model_psf, wcs=model_wcs)

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


def get_affine(wcs=None, linear=True):
    """Return the WCS transformation matrix

    The transformation to intermediate world coordinates is given by the equation
    $q = M\\cdot (p - r)$, where $p$ is the pixel coordinate, $r$ is `CRPIX`, and $M$ is the `CD` matrix.

    This method provides the augmented matrix of the affine transformation:
    $T = \begin{bmatrix} M & -M\\cdot r\\ 0 & 1\\end{bmatrix}$, for the extended vector $(p,1)$.

    See Greisen & Calabretta (2002) for details.

    Parameters
    ----------
    wcs: `astropy.wcs.WCS`
        WCS structure
    linear: `bool`
        Return only linear 2x2 matrix

    Returns
    -------
    array (3x3 or 2x2)
    """
    if wcs is None:
        return jnp.diag(jnp.ones(3))
    wcs_ = wcs.celestial
    try:
        m = wcs_.wcs.pc
    except AttributeError:
        try:
            m = wcs_.cd
        except AttributeError:
            m = wcs_.wcs.cd
    m = m[:2, :2]  # avoid using channel information that is not declared "spectral" in the WCS
    if linear:
        return m
    r = wcs_.wcs.crpix - 1  # CRPIX is 1-based!?!
    b = -m @ r
    t = jnp.zeros((3, 3))
    t = t.at[:2, :2].set(m).at[:2, 2].set(b).at[2, 2].set(1)
    return t


# for WCS linear matrix calculations:
# rotation matrix for counter-clockwise rotation from positive x-axis
# uses (x,y) coordinates and phi in radian!!
def _rot_matrix(phi, d=2):
    sinphi, cosphi = jnp.sin(phi), jnp.cos(phi)
    if d == 2:
        return jnp.array([[cosphi, -sinphi], [sinphi, cosphi]])
    else:
        return jnp.array([[cosphi, -sinphi, 0], [sinphi, cosphi, 0], [0, 0, 1]])


# flip in y!!!
# uses (x,y) coordinates!
_flip_matrix = flip_matrix = (
    lambda flip, d=2: jnp.diag(jnp.array((1, flip), dtype=float))
    if d == 2
    else jnp.diag(jnp.array((1, flip, 1), dtype=float))
)

# 2x2 matrix determinant
_det2 = lambda m: m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]

# round coordinate to nearest integer (use python, not jnp)
_minmax_int = lambda x: tuple(int(f) for f in jnp.round(jnp.sort(x)[jnp.array([0, -1])]))  # noqa:E731


# create trivial WCS for image with given shape
# scale = 1, pixel center in the middle of image
def _wcs_default(shape):
    shape_ = shape[-2:][::-1]  # x/y
    wcs = astropy.wcs.WCS(naxis=2)
    wcs._naxis = shape_
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = jnp.array((shape_[0] // 2, shape_[1] // 2)) + 1  # 1-based pixel coordinates
    return wcs


def get_scale_angle_flip_shift(trans):
    """Return, scale, angle, flip, translation from the WCS transformation matrix

    Parameters
    ----------
    trans: (`astropy.wcs.WCS`, array)
        WCS or WCS transformation matrix

    Returns
    -------
    scale: `float`
    angle: `float`, in radian
    flip: -1 or 1
    shift: `numpy.ndarray`

    See Also
    --------
    get_affine
    """
    if isinstance(trans, (np.ndarray, jnp.ndarray)):  # noqa: SIM108
        m = trans  # noqa: N806
    else:
        m = get_affine(trans)  # noqa: N806

    # get shift and then reduce to 2x2 for linear part
    if m.shape == (3, 3):
        shift = tuple(b.item() for b in m[:2, 2])[::-1]  # prevent tracing, y/x convention
        m = m[:2, :2]
    else:
        shift = (0, 0)

    det = _det2(m)
    # this requires pixels to be square
    # if not, use scale = jnp.linalg.svd(M, compute_uv=False)
    # but be careful with rotations as anisotropic stretch and rotation do not commute
    scale = jnp.sqrt(jnp.abs(det)).item(0)

    # if rotation is improper: need to apply y-flip to M to get pure rotation matrix (and unique angle)
    improper = det < 0
    flip = -1 if improper else 1
    f = _flip_matrix(flip)  # noqa: N806, flip in y, is identity if flip = 1!!!
    m_ = f @ m  # noqa: N806, flip = inverse flip
    angle = jnp.arctan2(m_[1, 0], m_[0, 0]).item()

    return scale, angle, flip, shift


def get_relative_jacobian_shift(model_frame, obs_frame):
    """Return the linear transformation matrix and shift between two frame WCSs

    Parameters
    ----------
    model_frame: `~scarlet2.Frame`
        The frame that defines the origin of the transformation
    obs_frame: `~scarlet2.Frame`
        The frame that defines the target of the transformation

    Returns
    -------
    jacobian: jnp.ndarray
        2x2 Jacobian matrix
    shift: tuple
        2D shift of the center of the frames

    """
    # Extract rotation angle, flip, scale between WCSs
    m_in = get_affine(model_frame.wcs)
    m_out = get_affine(obs_frame.wcs)
    jacobian = jnp.linalg.inv(m_out) @ m_in  # transformation from model pixel -> sky -> obs pixels

    # shift can be defined by the extended 3x3 Jacobian of the affine transformation matrix,
    # but it would ignore CRPIX/CRVAL difference between frmes
    # so we define it from the shift of the center of the two frames
    center_model = jnp.array(model_frame.bbox.spatial.center)
    center_model_in_obs = obs_frame.get_pixel(model_frame.get_sky_coord(center_model))
    center_obs = jnp.array(obs_frame.bbox.spatial.center)
    shift = center_obs - center_model_in_obs
    shift = tuple(c.item() for c in shift)  # avoid tracing
    return jacobian, shift


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
    scale, _, _, _ = get_scale_angle_flip_shift(wcs)
    return scale


def get_scale(wcs, separate=False):
    """Get WCS axis scales in deg/pixel

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
        scale, _ = get_scale_angle_flip_shift(wcs)
        return scale


def get_angle(wcs):
    """Get WCS rotation angle

    The angle is computed counter-clockwise from the positive x-axis, in radians.

    Parameters
    ----------
    wcs: `astropy.wcs.WCS`
        WCS structure or transformation matrix

    Returns
    -------
    `astropy.units.quantity.Quantity`, unit = u.rad
    """
    scale, angle, flip, shift = get_scale_angle_flip_shift(wcs)
    return angle


def get_flip(wcs):
    """Return WCS sign convention

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
    scale, angle, flip, shift = get_scale_angle_flip_shift(wcs)
    return flip


def get_shift(wcs):
    """Return WCS shift

    The WCS specify an affine transformation via the `CRPIX` keyword. This method
    returns the affine shift parameter in standard form.

    Parameters
    ----------
    wcs: `astropy.wcs.WCS`
        WCS structure or transformation matrix

    Returns
    -------
    array

    See Also
    --------
    get_affine
    """
    scale, angle, flip, shift = get_scale_angle_flip_shift(wcs)
    return shift
