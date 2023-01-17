from dataclasses import dataclass

import astropy.wcs.wcs
import jax.numpy as jnp

from .bbox import Box
from .psf import PSF


@dataclass
class Frame:
    bbox: Box
    psf: PSF = None
    wcs: astropy.wcs.wcs = None
    channels: (list, tuple) = None

    def __hash__(self):
        return hash(self.bbox)

    def get_pixel(self, sky_coord):
        """Get the pixel coordinate from a world coordinate

        Parameters
        ----------
        sky_coord: tuple, array
            Coordinates on the sky
        """
        sky = jnp.asarray(sky_coord, dtype=jnp.float64).reshape(-1, 2)

        if self._wcs is not None:
            wcs_ = self._wcs.celestial  # only use celestial portion
            pixel = jnp.array(wcs_.world_to_pixel_values(sky)).reshape(-1, 2)
            # y/x instead of x/y:
            pixel = jnp.flip(pixel, axis=-1)
        else:
            pixel = sky

        if pixel.size == 2:  # only one coordinate pair
            return pixel[0]
        return pixel

    def get_sky_coord(self, pixel):
        """Get the sky coordinate from a pixel coordinate

        Parameters
        ----------
        pixel: tuple, array
            Coordinates in the pixel space
        """
        pix = jnp.asarray(pixel, dtype=np.float64).reshape(-1, 2)

        if self._wcs is not None:
            wcs_ = self._wcs.celestial  # only use celestial portion
            # x/y instead of y/x:
            pix = jnp.flip(pix, axis=-1)
            sky = jnp.array(wcs_.pixel_to_world_values(pix))
        else:
            sky = pix

        if sky.size == 2:
            return sky[0]
        return sky
