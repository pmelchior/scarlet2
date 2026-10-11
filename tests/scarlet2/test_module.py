# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.distributions import constraints

import scarlet2 as sc2
from scarlet2 import Parameter, Parameters
from scarlet2.module import ParameterValidator


@pytest.fixture
def scene():
    frame = sc2.Frame(sc2.Box((3, 31, 31)))
    with sc2.Scene(frame) as scene:
        for center in ([10.0, 10.0], [20.0, 18.0]):
            sc2.Source(jnp.array(center), jnp.ones(3), jnp.full((11, 11), 0.1))

    with Parameters(scene):
        for i, source in enumerate(scene.sources):
            Parameter(source.center, name=f"center:{i}", stepsize=0.1)
            Parameter(source.spectrum, name=f"spectrum:{i}", constraint=constraints.positive, stepsize=0.1)
            Parameter(source.morphology, name=f"morph:{i}", constraint=constraints.positive, stepsize=0.1)
    return scene


@pytest.fixture
def sersic(scene):
    shape = scene.sources[0].morphology.shape
    return sc2.SersicMorphology(n=jnp.array(2.0), size=jnp.array(2.0), shape=shape)


def sersic_parameters(morph):
    Parameter(morph.n, name="n:0", prior=dist.Uniform(0.5, 6))
    Parameter(morph.size, name="size:0", constraint=constraints.positive, stepsize=0.1)


def test_replace_subtree(scene, sersic):
    where = lambda sc: sc.sources[0].morphology  # noqa: E731
    scene_ = scene.replace_subtree(where, sersic, parameters=sersic_parameters)

    assert isinstance(scene_.sources[0].morphology, sc2.SersicMorphology)
    assert set(scene_.parameters) == {
        "center_0",
        "spectrum_0",
        "center_1",
        "spectrum_1",
        "morph_1",
        "n_0",
        "size_0",
    }

    # remaining parameters point to unchanged nodes, and reuse Parameter instances
    values, values_ = scene.get(), scene_.get()
    for name in ["center_0", "spectrum_0", "center_1", "spectrum_1", "morph_1"]:
        assert jnp.array_equal(values[name], values_[name])
        assert scene.parameters[name][1] is scene_.parameters[name][1]
    assert values_["n_0"] is scene_.sources[0].morphology.n
    assert values_["size_0"] is scene_.sources[0].morphology.size

    # original scene is untouched
    assert set(scene.parameters) == {f"{p}_{i}" for p in ["center", "spectrum", "morph"] for i in range(2)}
    assert scene().shape == scene_().shape

    # set works with the remapped indices
    scene_ = scene_.set({"size_0": jnp.array(3.0), "morph_1": jnp.zeros((11, 11))})
    assert scene_.sources[0].morphology.size == 3.0
    assert jnp.all(scene_.sources[1].morphology == 0)


def test_replace_subtree_without_new_parameters(scene, sersic):
    scene_ = scene.replace_subtree(lambda sc: sc.sources[0].morphology, sersic)
    assert "morph_0" not in scene_.parameters
    assert len(scene_.parameters) == len(scene.parameters) - 1


def test_replace_subtree_without_parameters(sersic):
    frame = sc2.Frame(sc2.Box((3, 31, 31)))
    with sc2.Scene(frame) as scene:
        sc2.Source(jnp.array([10.0, 10.0]), jnp.ones(3), jnp.full((11, 11), 0.1))
    scene_ = scene.replace_subtree(lambda sc: sc.sources[0].morphology, sersic)
    assert isinstance(scene_.sources[0].morphology, sc2.SersicMorphology)
    assert len(scene_.parameters) == 0


def test_remap_multiple_excludes(scene, sersic):
    where = lambda sc: (sc.sources[0].morphology, sc.sources[1].spectrum)  # noqa: E731
    scene_ = eqx.tree_at(where, scene, (sersic, jnp.full(3, 2.0)))
    with Parameters(scene_) as parameters:
        parameters.remap(scene.parameters, exclude=where)
    assert set(scene_.parameters) == {"center_0", "spectrum_0", "center_1", "morph_1"}


def test_remap_missing_node(scene, sersic):
    # replacing the morphology without excluding its parameter
    scene_ = eqx.tree_at(lambda sc: sc.sources[0].morphology, scene, sersic)
    with pytest.raises(RuntimeError, match="morph_0"), Parameters(scene_) as parameters:
        parameters.remap(scene.parameters)


@pytest.mark.parametrize("value, n_errors", [(2.0, 0), (-2.0, 1), (jnp.array(-2.0), 1)])
def test_constrained_scalar_parameter_feasibility(value, n_errors):
    parameter = Parameter.__new__(Parameter)
    parameter.constraint, parameter.prior, parameter.stepsize = constraints.positive, None, 0.1
    validator = ParameterValidator("size", parameter, value)
    assert len(validator.check_constrained_parameter_is_feasible()) == n_errors


def test_replace_subtree_with_python_floats(scene):
    shape = scene.sources[0].morphology.shape
    sersic = sc2.SersicMorphology(n=2.0, size=2.0, shape=shape)
    scene_ = scene.replace_subtree(lambda sc: sc.sources[0].morphology, sersic, sersic_parameters)
    assert scene_.get("size_0") == 2.0
