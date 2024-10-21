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
    # now loop over the sources and save the data
    for idx, src in enumerate(scene.sources):
        src_name = "source_" + str(idx)
        # check if the source already exists
        if src_name in f.keys():
            if overwrite:
                del f[src_name]
            else:
                raise ValueError("Source already exists. Set overwrite=True to overwrite the source.")
        group = f.create_group(src_name)
        group.create_dataset("morph", data=src.morphology.data)

    f.close()
    return None

    
