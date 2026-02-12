import jax
import jax.numpy as jnp

from . import Scenery
from .bbox import overlap_slices
from .frame import Frame
from .module import Module
from .validation_utils import print_validation_results


class Scene(Module):
    """Model of the celestial scene

    This class connects the main functionality of `scarlet2`: the fitting of an
    :py:class:`~scarlet2.Observation` (or several) by a :py:class:`~scarlet2.Source`
    model (or several). Model parameters can be optimized or samples with any method
    implemented in jax, but this class provides the :py:func:`fit` and
    :py:func:`sample` methods as built-in solutions.
    """

    frame: Frame
    """Portion of the sky represented by this model"""
    sources: list
    """List of :py:class:`~scarlet2.Source` comprised in this model"""

    def __init__(self, frame):
        """
        Parameters
        ----------
        frame: `Frame`
            Portion of the sky represented by this model

        Examples
        --------
        The class provides a context so that sources can be added to the same model frame:

        >>> with Scene(model_frame) as scene:
        >>>    Source(center, spectrum, morphology)

        This adds a single source to the list :py:attr:`~scarlet2.Scene.sources`
        of `scene`. The context provides a common definition of the model frame,
        so that, e.g., `center` can be given as :py:class:`astropy.coordinates.SkyCoord`
        and will automatically be converted to the pixel coordinate in the model frame.

        The constructed source does not go out of scope after the `with` context
        is closed, it is stored in the scene.

        See Also
        --------
        :py:class:`~scarlet2.Scenery`, :py:class:`~scarlet2.Source`
        """
        self.frame = frame
        self.sources = list()

    def __call__(self):
        """What to run when the scene is called"""
        model = jnp.zeros(self.frame.bbox.shape)
        for source in self.sources:
            model += self.evaluate_source(source)
        return model

    def evaluate_source(self, source):
        """Evaluate a single source in the frame of this scene.

        This method inserts the model of `source` into the proper location in `scene`.

        Parameters
        ----------
        source: :py:class:`~scarlet2.Source`
            The source to evaluate.

        Returns
        -------
        array
            Array of the dimension indicated by :py:attr:`shape`.
        """
        model_ = source()
        # cut out region from model, add single source model
        bbox, bbox_ = overlap_slices(self.frame.bbox, source.bbox, return_boxes=True)
        sub_model_ = jax.lax.dynamic_slice(model_, bbox_.start, bbox_.shape)

        # add model_ back in full model
        model = jnp.zeros(self.frame.bbox.shape)
        model = jax.lax.dynamic_update_slice(model, sub_model_, bbox.start)
        return model

    def __enter__(self):
        # context manager to register sources
        # purpose is to provide scene.frame to source inits that will need some of its information
        # also allows us to append the sources automatically to the scene
        Scenery.scene = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        Scenery.scene = None

        # (re)-import `VALIDATION_SWITCH` at runtime to avoid using a static/old value
        from .validation_utils import VALIDATION_SWITCH

        if VALIDATION_SWITCH:
            from .validation import check_scene

            validation_results = check_scene(self)
            print_validation_results("Source validation results", validation_results)
