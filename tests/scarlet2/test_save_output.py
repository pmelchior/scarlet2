# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103
# ruff: noqa: D106

import os

import astropy.wcs as wcs
import h5py
import jax
import jax.tree_util as jtu
import pytest

from scarlet2.io import model_from_h5, model_to_h5
from scarlet2.validation_utils import set_validation


@pytest.fixture(autouse=True)
def setup_validation():
    """Automatically disable validation for all tests. This permits the creation
    of intentionally invalid Observation objects."""
    set_validation(False)


def test_save_output(scene):
    # save the output
    id = 1
    filename = "demo_io.h5"
    path = "stored_models"
    model_to_h5(scene, filename, id, path=path, overwrite=True)

    # demo that it works to add models to a single file
    id = 2
    model_to_h5(scene, filename, id, path=path, overwrite=True)

    # load files and show keys
    full_path = os.path.join(path, filename)
    with h5py.File(full_path, "r") as f:
        print(f.keys())

    # print the output
    print(f"Output saved to {full_path}")
    # print the storage size
    print(f"Storage size: {os.path.getsize(full_path) / 1e6:.4f} MB")
    # load the output and plot the sources
    scene_loaded = model_from_h5(filename, id, path=path)
    print("Output loaded from h5 file")

    # compare scenes
    saved = jtu.tree_leaves(scene)
    loaded = jtu.tree_leaves(scene_loaded)
    status = True
    for leaf_saved, leaf_loaded in zip(saved, loaded, strict=False):
        if isinstance(leaf_saved, wcs.WCS):  # wcs doesn't allow direct == comparison...
            if not leaf_saved.wcs.compare(leaf_loaded.wcs):
                status = False
        elif isinstance(leaf_saved, jax.Array):
            if (leaf_saved != leaf_loaded).any():
                status = False
        else:
            if leaf_saved != leaf_loaded:
                status = False

    print(f"saved == loaded: {status}")
    assert status, "Loaded leaves not identical to original"


if __name__ == "__main__":
    test_save_output()
