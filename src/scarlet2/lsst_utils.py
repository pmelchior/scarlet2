import jax.numpy as jnp
import lsst.afw.geom as afwGeom
import lsst.geom as geom
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from lsst.afw.fits import MemFileManager
from lsst.afw.image import ExposureF
from lsst.geom import Point2D
from lsst.meas.algorithms import WarpedPsf
from lsst.pipe.tasks.registerImage import RegisterConfig, RegisterTask
from pyvo.dal.adhoc import DatalinkResults, SodaQuery

import scarlet2


def warp_img(ref_img, img_to_warp, ref_wcs, wcs_to_warp):
    """Warp and rotate an image onto the coordinate system of another image

    Parameters
    ----------
    ref_img: 'ExposureF'
        is the reference image for the re-projection
    img_to_warp: 'ExposureF'
        the image to rotate and warp onto the reference image's wcs
    ref_wcs: 'wcs object'
        the wcs of the reference image (i.e. ref_img.getWcs() )
    wcs_to_warp: 'wcs object'
        the wcs of the image to warp (i.e. img_to_warp.getWcs() )
    Returns
    -------
    warpedExp: 'ExposureF'
        a reprojected, rotated image that is aligned and matched to ref_image
    """
    config = RegisterConfig()
    task = RegisterTask(name="register", config=config)
    warpedExp = task.warpExposure(img_to_warp, wcs_to_warp, ref_wcs, ref_img.getBBox())

    return warpedExp


def read_cutout_mem(sq):
    """Read the cutout into memory

    Parameters
    ----------
    sq : 'dict'
        returned from SodaQuery.from_resource()

    Returns
    -------
    exposure : 'ExposureF'
        the cutout in exposureF format
    """

    cutout_bytes = sq.execute_stream().read()
    sq.raise_if_error()
    mem = MemFileManager(len(cutout_bytes))
    mem.setData(cutout_bytes, len(cutout_bytes))
    exposure = ExposureF(mem)

    return exposure


def make_image_cutout(tap_service, ra, dec, dataId, cutout_size=0.01, imtype=None, filename=None):
    """Wrapper function to generate a cutout using the cutout tool

    Parameters
    ----------
    tap_service : an instance of the TAP service
    ra, dec : 'float'
        the ra and dec of the cutout center
    dataId : 'dict'
        the dataId of the image to make a cutout from. The format
        must correspond to that provided for parameter 'imtype'
    cutout_size : 'float', optional
        edge length in degrees of the cutout
    imtype : 'string', optional
        string containing the type of LSST image to generate
        a cutout of (e.g. deepCoadd, calexp). If imtype=None,
        the function will assume a deepCoadd.
    filename : 'string', optional
        filename of the resulting cutout (which has fits format)

    Returns
    -------
    exposure : 'ExposureF'
        the cutout in exposureF format
    """

    spherePoint = geom.SpherePoint(ra * geom.degrees, dec * geom.degrees)

    if imtype == "calexp":
        query = (
            "SELECT access_format, access_url, dataproduct_subtype, "
            + "lsst_visit, lsst_detector, lsst_band "
            + "FROM dp02_dc2_catalogs.ObsCore WHERE dataproduct_type = 'image' "
            + "AND obs_collection = 'LSST.DP02' "
            + "AND dataproduct_subtype = 'lsst.calexp' "
            + "AND lsst_visit = "
            + str(dataId["visit"])
            + " "
            + "AND lsst_detector = "
            + str(dataId["detector"])
        )
        results = tap_service.search(query)

    else:
        # Find the tract and patch that contain this point
        tract = dataId["tract"]
        patch = dataId["patch"]

        # add optional default band if it is not contained in the dataId
        band = dataId["band"]

        query = (
            "SELECT access_format, access_url, dataproduct_subtype, "
            + "lsst_patch, lsst_tract, lsst_band "
            + "FROM dp02_dc2_catalogs.ObsCore WHERE dataproduct_type = 'image' "
            + "AND obs_collection = 'LSST.DP02' "
            + "AND dataproduct_subtype = 'lsst.deepCoadd_calexp' "
            + "AND lsst_tract = "
            + str(tract)
            + " "
            + "AND lsst_patch = "
            + str(patch)
            + " "
            + "AND lsst_band = "
            + "'"
            + str(band)
            + "' "
        )
        results = tap_service.search(query)

    # Get datalink
    dataLinkUrl = results[0].getdataurl()
    auth_session = tap_service._session
    dl = DatalinkResults.from_result_url(dataLinkUrl, session=auth_session)

    # from_resource: creates a instance from
    # a number of records and a Datalink Resource.
    sq = SodaQuery.from_resource(dl, dl.get_adhocservice_by_id("cutout-sync"), session=auth_session)

    sq.circle = (
        spherePoint.getRa().asDegrees() * u.deg,
        spherePoint.getDec().asDegrees() * u.deg,
        cutout_size * u.deg,
    )

    exposure = read_cutout_mem(sq)

    # cutout_bytes = sq.execute_stream().read()
    # mem = MemFileManager(len(cutout_bytes))
    # mem.setData(cutout_bytes, len(cutout_bytes))
    # exposure = ExposureF(mem)

    return exposure


def dia_source_to_observations(cutout_size_pix, dia_src, service, plot_images=False):
    """Convert a DIA source to a list of scarlet2 Observations

    Parameters
    ----------
    cutout_size_pix : 'int'
        the size of the cutout in pixels
    dia_src : `astropy.table.Table`
        the DIA source table containing the sources to make cutouts for
    service : `pyvo.dal.tap.TAPService`
        the TAP service to use for querying the cutouts
        i.e. the result from lsst.rsp.get_tap_service()
    tempdir : 'str'
        the temporary directory to write the cutouts to

    Returns
    -------
    observations : 'list of scarlet2.Observation'
        a list of scarlet2 Observations, one for each DIA source
    channels_sc2 : 'list of tuples'
        a list of tuples containing the band and channel number for each observation
    """
    # TODO: - add back in the plotting functionality?
    # - figure out how to get the WCS from the cutout without writing to disk
    cutout_size = cutout_size_pix * 0.2 / 3600.0
    ra = dia_src["ra"][0]
    dec = dia_src["decl"][0]

    observations = []
    channels_sc2 = []
    img_ref = None
    wcs_ref = None

    first_time = dia_src["midPointTai"][0]
    vmin = -200
    vmax = 300

    if plot_images:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    for i, src in enumerate(dia_src):
        ccdvisitID = src["ccdVisitId"]
        band = str(src["filterName"])
        visit = str(ccdvisitID)[:-3]
        detector = str(ccdvisitID)[-3:]
        visit = int(visit)
        detector = int(detector)
        dataId_calexp = {"visit": visit, "detector": detector}

        img = make_image_cutout(
            service, ra, dec, cutout_size=cutout_size * 50.0, imtype="calexp", dataId=dataId_calexp
        )
        if i == 0:
            img_ref = img
            # no warping is needed for the reference
            img_warped = img_ref
            offset = geom.Extent2D(geom.Point2I(0, 0) - img_ref.getXY0())
            shifted_wcs = img_ref.getWcs().copyAtShiftedPixelOrigin(offset)
            wcs_ref = WCS(shifted_wcs.getFitsMetadata())
        else:
            img_warped = warp_img(img_ref, img, img_ref.getWcs(), img.getWcs())
        im_arr = img_warped.image.array
        var_arr = img_warped.variance.array
        N1, N2 = im_arr.shape
        image_sc2 = im_arr.reshape(1, N1, N2)
        N1, N2 = var_arr.shape
        weight_sc2 = 1 / var_arr.reshape(1, N1, N2)
        point_tuple = (int(img_warped.image.array.shape[0] / 2), int(img_warped.image.array.shape[1] / 2))
        point_image = Point2D(point_tuple)
        xyTransform = afwGeom.makeWcsPairTransform(img.wcs, img_warped.wcs)
        psf_w = WarpedPsf(img.getPsf(), xyTransform)
        point_tuple = (int(img_warped.image.array.shape[0] / 2), int(img_warped.image.array.shape[1] / 2))
        point_image = Point2D(point_tuple)
        psf_warped = psf_w.computeImage(point_image).convertF()
        if np.sum(psf_warped.array) == 0:
            print("PSF model unavailable, skipping")
            continue
        N1, N2 = psf_warped.array.shape
        psf_sc2 = psf_warped.array.reshape(1, N1, N2)
        # this is required to get the WCS, we should figure out a better
        # way to do this without writing to disk
        obs = scarlet2.Observation(
            jnp.array(image_sc2).astype(float),
            weights=jnp.array(weight_sc2).astype(float),
            psf=scarlet2.ArrayPSF(jnp.array(psf_sc2).astype(float)),
            wcs=wcs_ref,
            channels=[(band, str(i))],
        )
        channels_sc2.append((band, str(i)))
        observations.append(obs)

        if plot_images:
            plt.imshow(im_arr, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.1,
                0.9,
                r"$\Delta$t=" + str(round(src["midPointTai"] - first_time, 2)),
                color="white",
                fontsize=12,
            )
            plt.show()
            plt.close()

    return observations, channels_sc2
