import h5py
import os
import numpy as np
import jax.numpy as jnp
from scarlet2 import *
import pickle
import jax


def model_to_h5(filename, scene, ID, path="", overwrite=False):
    """
    Save the model output to a single HDF5 file.
    NOTE: This is not a pure function hence cannot be
    utalized within a JAX JIT compilation.

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
    model_group = str(ID)
    save_h5_path = os.path.join(path, filename)

    f = h5py.File(f"{save_h5_path}.h5", "a")
    # create a group for the scene
    if model_group in f.keys():
        if overwrite:
            del f[model_group]
        else:
            raise ValueError(
                "ID already exists. Set overwrite=True to overwrite the ID."
            )

    # save the binary to HDF5
    group = f.create_group(model_group)
    model = pickle.dumps(scene)
    group.attrs["model"] = np.void(model)
    f.close()


def model_from_h5(filename, ID, path=""):
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

    filename = os.path.join(path, filename)
    f = h5py.File(f"{filename}.h5", "r")
    model_group = str(ID)
    if model_group not in f.keys():
        raise ValueError(f"ID {ID} not found in the file.")

    group = f.get(model_group)
    out = group.attrs["model"]
    binary_blob = out.tobytes()
    scene = pickle.loads(binary_blob)
    f.close()

    return scene
