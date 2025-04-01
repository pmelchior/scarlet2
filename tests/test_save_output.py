import os

import h5py
import jax
import jax.numpy as jnp

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

    # save the output
    ID = 1
    filename = "demo_io.h5"
    path = "stored_models"
    model_to_h5(scene, filename, ID, path=path, overwrite=True)

    # demo that it works to add models to a single file
    ID = 2
    model_to_h5(scene, filename, ID, path=path, overwrite=True)

    # load files and show keys
    full_path = os.path.join(path, filename)
    with h5py.File(full_path, "r") as f:
        print(f.keys())

    # print the output
    print(f"Output saved to {full_path}")
    # print the storage size 
    print(f"Storage size: {os.path.getsize(full_path) / 1e6:.4f} MB")
    # load the output and plot the sources
    scene_loaded = model_from_h5(filename, ID, path=path)
    print("Output loaded from h5 file")

    # compare scenes 
    saved = jax.tree_util.tree_leaves(scene)
    loaded = jax.tree_util.tree_leaves(scene_loaded)
    status = True
    for leaf_saved, leaf_loaded in zip(saved, loaded):
        if (leaf_saved != leaf_loaded).all():
            status = False
        
    print(f"saved == loaded: {status}")
    assert status == True, "Loaded leaves not identical to original"

if __name__ == "__main__":
    test_save_output()
