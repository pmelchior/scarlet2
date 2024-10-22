import h5py
import os
import numpy as np
import equinox as eqx
import jax.numpy as jnp
from scarlet2 import *


def model_to_h5(filename, scene, ID, path="", overwrite=True):
    """
    Save the model output to a single HDF5 file.
    NOTE: This is not a pure function hence cannot be
    utalised within a JAX JIT compilation.

    Inputs
    ------
    filename : str
    model : scarlet2.scene instance
    ID : int
    overwrite : bool, optional

    Returns
    -------
    None
    """
    # create directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # first serialize the model into a pytree
    model_group = "scene_id_" + str(ID)
    save_path = os.path.join(path, model_group)
    save_h5_path = os.path.join(path, filename)
    print(f"Saving model to {save_path}")
    eqx.tree_serialise_leaves(f"{save_path}.eqx", scene)

    f = h5py.File(f"{save_h5_path}.h5", "a")
    # create a group for the scene
    if model_group in f.keys():
        if overwrite:
            del f[model_group]
        else:
            raise ValueError(
                "ID already exists. Set overwrite=True to overwrite the ID."
            )

    group = f.create_group(model_group)
    group.create_dataset("model", data=f"{save_path}.eqx")
    group.create_dataset("bbox", data=scene.frame.bbox.shape)
    group.create_dataset("psf", data=scene.frame.psf.morphology.size, dtype="f4")
    group.create_dataset("channels", data=scene.frame.channels)

    # now save parameters for each source
    for i, src in enumerate(scene.sources):
        src_group = group.create_group(f"source_{i}")
        src_group.create_dataset("center", data=src.center)
        src_group.create_dataset("morph", data=src.morphology.data)
        src_group.create_dataset("spectrum", data=src.spectrum.data)
        src_group.create_dataset("bbox", data=src.bbox.shape)
        src_group.create_dataset("components", data=src.components)

    group.create_dataset("n_src", data=i + 1)

    f.close()

    return None


def model_from_h5(filename, ID, path="", scene=Scene):
    """
    Load the model output from a single HDF5 file.

    Inputs
    ------
    filename : str
    ID : int

    Returns
    -------
    scene : scarlet2.scene instance
    """

    # NOTE: I can reconstruct a blank scene with just the params

    filename = os.path.join(path, filename)
    f = h5py.File(f"{filename}.h5", "r")
    model_group = "scene_id_" + str(ID)
    model = f.get(model_group)
    if model_group not in f.keys():
        raise ValueError("ID not found in the file.")

    # now instantiate a blank scene
    psf = 0.7  # np.array(model.get('psf'))
    bbox = np.array(model.get("bbox"))
    channels = jnp.array(model.get("channels"))
    frame_psf = GaussianPSF(psf)
    model_frame = Frame(Box(bbox), psf=frame_psf)
    nsrc = np.array(model.get("n_src"))

    # loop through the sources
    with Scene(model_frame) as scene_:

        for i in range(nsrc):
            src_group = model.get(f"source_{i}")
            center = jnp.array(src_group.get("center"))
            spectrum = ArraySpectrum(jnp.array(src_group.get("spectrum")))
            morph = ArrayMorphology(jnp.array(src_group.get("morph")))
            Source(center, spectrum, morph)

    scene = eqx.tree_deserialise_leaves("stored_models/scene_id_1.eqx", scene_)
    f.close()

    return scene
