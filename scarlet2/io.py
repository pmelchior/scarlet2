import h5py
import jax.numpy as jnp 

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
    num_sources = scene.num_sources

    f = h5py.File(filename, "a")

    # now loop over the sources and save the data
    for idx in range(num_sources):
        src_name = "source_" + str(idx)
        group = f.create_group(src_name)
        group.create_dataset("morph", data=scene.sources[idx].morphology.data)



    f.close()
    return None

    
