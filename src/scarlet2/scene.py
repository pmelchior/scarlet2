import jax
import jax.numpy as jnp

from . import Scenery
from .bbox import overlap_slices
from .frame import Frame
from .module import Module
from .validation_utils import print_validation_results


class SourceList(list):
    """List subclass for :py:attr:`~scarlet2.Scene.sources`.

    Overrides ``__setitem__`` so that direct assignment (e.g.
    ``scene.sources[0] = Source(...)``) works correctly.  Because
    :py:class:`~scarlet2.Source.__init__` always appends the new source to the
    list, a plain ``sources[0] = Source(...)`` would leave the source registered
    twice.  This subclass detects that the source was just appended and moves
    it to the requested index instead.
    """

    def __setitem__(self, index, value):
        if self[-1] is value:
            self.pop()
        super().__setitem__(index, value)


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
        self.sources = SourceList()

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
        # this scene might have parameters defined, we need to reset its registry key
        object.__setattr__(self, "registry_key", "")

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

    def fit(
        self,
        observations,
        *args,
        schedule=None,
        max_iter=100,
        e_rel=1e-4,
        progress_bar=True,
        callback=None,
        **kwargs,
    ):
        """Fit model `parameters` of every source in the scene to match `observations`.

        Parameters
        ----------
        observations: :py:class:`~scarlet2.Observation` or list
            The observations to fit the model to.
        *args: list, optional
            Additional arguments passed. Only used for backwards (v0.3) compatibility.
        schedule: callable, optional
            A function that maps optimizer step count to value. See :py:class:`optax.Schedule` for details.
        max_iter: int, optional
            Maximum number of optimizer iterations
        e_rel: float, optional
            Upper limit for the relative change in the norm of any parameter to
            terminate the optimization early.
        progress_bar: bool, optional
            Whether to show a progress bar
        callback: callable, optional
            Function to be called on the current state of the optimized scene.
            Signature `callback(scene, convergence, loss) -> None`, where
            `convergence` is a tree of the same structure as `scene`, and `loss`
            is the current value of the log_posterior.
        **kwargs: dict, optional
            Additional keyword arguments passed to the `optax.scale_by_adam` optimizer.

        Returns
        -------
        Scene
            The scene model with updated parameters

        See Also
        --------
        :py:func:`~scarlet2.fit`
        """

        # making sure we can iterate
        if not isinstance(observations, (list, tuple)):
            observations = (observations,)
        # don't use this function with observation parameters
        if any(len(obs.parameters) for obs in observations):
            msg = "For Scene.fit(), observations must not have parameters. Use scarlet2.fit() instead."
            raise RuntimeError(msg)

        from .infer import fit

        scene_, _ = fit(
            self,
            observations,
            schedule=schedule,
            max_iter=max_iter,
            e_rel=e_rel,
            progress_bar=progress_bar,
            callback=callback,
            **kwargs,
        )
        return scene_

    def sample(
        self, observations, *args, seed=0, num_warmup=100, num_samples=200, progress_bar=True, **kwargs
    ):
        """Sample `parameters` of every source in the scene to get posteriors given `observations`.

        This method runs the HMC NUTS sampler from `numpyro` to get parameter
        posteriors. It uses the likelihood of `observations` as well as the `prior`
        attribute set for every :py:class:`~scarlet2.Parameter` in `parameters`.

        Parameters
        ----------
        observations: :py:class:`~scarlet2.Observation` or list
            The observations to fit the models to.
        *args: list, optional
            Additional arguments passed. Only used for backwards (v0.3) compatibility.
        seed: int, optional
            RNG seed for the sampler
        num_warmup: int, optional
            Number of samples during HMC warm-up
        num_samples: int, optional
            Number of samples to create from tuned HMC
        progress_bar: bool, optional
            Whether to show a progress bar
        **kwargs: dict, optional
            Additional keyword arguments passed to the `numpyro.infer.NUTS` sampler.

        Returns
        -------
        numpyro.infer.mcmc.MCMC

        See Also
        --------
        :py:func:`~scarlet2.sample`
        """
        from .infer import sample

        return sample(
            self,
            observations,
            seed=seed,
            num_warmup=num_warmup,
            num_samples=num_samples,
            progress_bar=progress_bar,
            **kwargs,
        )
