import h5py
import jax.numpy as jnp
import os

def save_h5py(filename, scene, ID, overwrite=True):
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

    f = h5py.File(filename, "a")
    # create a group for the scene 
    model_group = "scene_id_" + str(ID)
    if model_group in f.keys():
        if overwrite:
            del f[model_group]
        else:
            raise ValueError("ID already exists. Set overwrite=True to overwrite the ID.")
    group = f.create_group(model_group)

    # now loop over the sources and save the data
    for idx, src in enumerate(scene.sources):
        src_name = "source_" + str(idx)
        # check if the source already exists
        if src_name in group.keys():
            if overwrite:
                del f[src_name]
            else:
                raise ValueError("Source already exists. Set overwrite=True to overwrite the source.")
        src_group = group.create_group(src_name)
        src_group.create_dataset("morph", data=src.morphology.data)
        src_group.create_dataset("spectrum", data=src.spectrum.data)

    f.close()
    return None

    
