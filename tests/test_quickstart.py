import os

import jax.numpy as jnp
from numpyro.distributions import constraints

from scarlet2 import *
from utils import import_scarlet_test_data

import_scarlet_test_data()
from scarlet_test_data import data_path


def test_quickstart():
    file = jnp.load(os.path.join(data_path, "hsc_cosmos_35.npz"))
    data = jnp.asarray(file["images"])
    centers = [(src['y'], src['x']) for src in file["catalog"]]  # Note: y/x convention!
    weights = jnp.asarray(1 / file["variance"])
    psf = jnp.asarray(file["psfs"])

    frame_psf = GaussianPSF(0.7)
    model_frame = Frame(Box(data.shape), psf=frame_psf)
    obs = Observation(data, weights, psf=ArrayPSF(jnp.asarray(psf))).match(model_frame)

    from functools import partial
    spec_step = partial(relative_step, factor=0.05)

    with Scene(model_frame) as scene:

        for center in centers:
            center = jnp.array(center)
            spectrum = init.pixel_spectrum(obs, center)
            morph = init.adaptive_gaussian_morph(obs, center, min_corr=0.99) + 1e-6

            Source(
                center,
                ArraySpectrum(spectrum),
                ArrayMorphology(morph)
            )

    norm = plot.AsinhAutomaticNorm(obs)
    plot.scene(scene, obs, norm=norm, show_model=True, show_rendered=True, show_observed=True, show_residual=True);

    parameters = scene.make_parameters()
    for i in range(len(scene.sources)):
        parameters += Parameter(scene.sources[i].spectrum.data, name=f"spectrum:{i}", constraint=constraints.positive,
                                stepsize=spec_step)
        parameters += Parameter(scene.sources[i].morphology.data, name=f"morph:{i}", constraint=constraints.positive,
                                stepsize=0.1)

    maxiter = 200
    scene.set_spectra_to_match(obs, parameters)
    scene_ = scene.fit(obs, parameters, max_iter=maxiter, progress_bar=False)
    plot.scene(scene_, obs, norm=norm, show_model=True, show_rendered=True, show_observed=True, show_residual=True);


if __name__ == "__main__":
    test_quickstart()
