import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from numpyro.distributions import constraints

from scarlet2 import *


def test_quickstart():
    filename = hf_hub_download(repo_id="astro-data-lab/scarlet-test-data", filename="hsc_cosmos_35.npz",
                               repo_type="dataset")
    file = jnp.load(filename)
    data = jnp.asarray(file["images"])
    channels = [str(f) for f in file['filters']]
    centers = [(src['y'], src['x']) for src in file["catalog"]]  # Note: y/x convention!
    weights = jnp.asarray(1 / file["variance"])
    psf = jnp.asarray(file["psfs"])

    frame_psf = GaussianPSF(0.7)
    obs = Observation(data, weights, psf=ArrayPSF(jnp.asarray(psf)), channels=channels)
    model_frame = Frame.from_observations(obs)

    from functools import partial
    spec_step = partial(relative_step, factor=0.05)

    with Scene(model_frame) as scene:

        for center in centers:
            center = jnp.array(center)
            try:
                spectrum, morph = init.from_gaussian_moments(obs, center, min_corr=0.99)
            except ValueError:
                spectrum = init.pixel_spectrum(obs, center)
                morph = init.compact_morphology()

            Source(center, spectrum, morph)

    # fitting
    parameters = scene.make_parameters()
    for i in range(len(scene.sources)):
        parameters += Parameter(scene.sources[i].spectrum, name=f"spectrum:{i}", constraint=constraints.positive,
                                stepsize=spec_step)
        parameters += Parameter(scene.sources[i].morphology, name=f"morph:{i}", constraint=constraints.positive,
                                stepsize=0.1)

    maxiter = 100
    scene.set_spectra_to_match(obs, parameters)
    scene_ = scene.fit(obs, parameters, max_iter=maxiter, progress_bar=False)

    # sampling
    import numpyro.distributions as dist
    from numpyro.infer.initialization import init_to_sample
    parameters = scene_.make_parameters()
    p = scene_.sources[0].spectrum
    prior = dist.Normal(p, scale=1)
    parameters += Parameter(p, name=f"spectrum:0", prior=prior)
    mcmc = scene_.sample(obs, parameters, num_samples=200, dense_mass=True, init_strategy=init_to_sample,
                         progress_bar=False)

if __name__ == "__main__":
    test_quickstart()
