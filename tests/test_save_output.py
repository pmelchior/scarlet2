import os
import h5py
import numpy as np
import jax
import jax.numpy as jnp
from numpyro.distributions import constraints
import matplotlib.pyplot as plt

from scarlet2 import *
from utils import import_scarlet_test_data

import_scarlet_test_data()
from scarlet_test_data import data_path
from scarlet2.io import model_to_h5, model_from_h5


def test_save_output():
    file = jnp.load(os.path.join(data_path, "hsc_cosmos_35.npz"))
    data = jnp.asarray(file["images"])
    centers = [(src["y"], src["x"]) for src in file["catalog"]]  # Note: y/x convention!
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
            try:
                spectrum, morph = init.from_gaussian_moments(obs, center, min_corr=0.99)
            except ValueError:
                spectrum = init.pixel_spectrum(obs, center)
                morph = init.compact_morphology()

            Source(center, spectrum, morph)

    # fitting
    parameters = scene.make_parameters()
    for i in range(len(scene.sources)):
        parameters += Parameter(
            scene.sources[i].spectrum.data,
            name=f"spectrum:{i}",
            constraint=constraints.positive,
            stepsize=spec_step,
        )
        parameters += Parameter(
            scene.sources[i].morphology.data,
            name=f"morph:{i}",
            constraint=constraints.positive,
            stepsize=0.1,
        )

    maxiter = 200
    scene.set_spectra_to_match(obs, parameters)
    scene_ = scene.fit(obs, parameters, max_iter=maxiter, progress_bar=False)

    # save the output
    ID = 1
    filename = "demo_io"
    path = "stored_models"
    model_to_h5(filename, scene_, ID, path=path, overwrite=True)

    # demo that it works to add models to a single file
    ID = 2
    filename = "demo_io"
    path = "stored_models"
    model_to_h5(filename, scene_, ID, path=path, overwrite=True)

    # load files and show keys 
    with h5py.File(f"{path}/{filename}.h5", "r") as f:
        print(f.keys())

    # print the output
    print(f"Output saved to {path}/{filename}.h5")
    # print the storage size 
    print(f"Storage size: {os.path.getsize(f'{path}/{filename}.h5')/1e6:.4f} MB")
    # load the output and plot the sources
    scene_loaded = model_from_h5(filename, ID, path=path)
    print("Output loaded from h5 file")

    # compare scenes 
    saved = jax.tree_util.tree_leaves(scene_)
    loaded = jax.tree_util.tree_leaves(scene_loaded)
    status = True
    for leaf_saved, leaf_loaded in zip(saved, loaded):
        if (leaf_saved != leaf_loaded).all():
            status = False
        
    print(f"saved == loaded: {status}")
    assert status == True, "Loaded leaves not identical to original"

if __name__ == "__main__":
    test_save_output()
