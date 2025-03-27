"""Methods to save and load scenes"""

import os
import pickle

import h5py
import numpy as np


def model_to_h5(filename, scene, id, path="", overwrite=False):
    """Save the scene model to a HDF5 file

    Parameters
    ----------
    filename : str
        Name of the HDF5 file to create
    scene : :py:class:`~scarlet2.Scene`
        Scene to be stored
    id : int
        HDF5 group to store this `scene` under
    path: str, optional
        Explicit path for `filename`. If not set, uses local directory
    overwrite : bool, optional
        Whether to overwrite an existing file with the same path and filename

    Returns
    -------
    None

    Notes
    -----
    This is not a pure function hence cannot be utilized within a JAX JIT compilation.
    """
    # create directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # first serialize the model into a pytree
    model_group = str(id)
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


def model_from_h5(filename, id, path=""):
    """
    Load scene model from a HDF5 file

    Parameters
    ----------
    filename : str
        Name of the HDF5 file to load from
    id : int
        HDF5 group to identify the scene by
    path: str, optional
        Explicit path for `filename`. If not set, uses local directory

    Returns
    -------
    :py:class:`~scarlet2.Scene`
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
