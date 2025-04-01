# _scarlet2_ Documentation

_scarlet2_ is an open-source python library for modeling astronomical sources from multi-band, multi-epoch, and
multi-instrument data.
It's implemented in [jax](http://jax.readthedocs.io/), layered on top of
the [equinox](https://docs.kidger.site/equinox/)
library. It can be deployed to GPUs and TPUs and supports optimization and sampling approaches.

## Installation

For performance reasons, you should first install `jax` with the suitable `jaxlib` for your platform. After that

```
pip install scarlet2
```

should do. If you want the latest development version, use

```
pip install git+https://github.com/pmelchior/scarlet2.git
```

This will allow you to evaluate source models and compute likelihoods of observed data, so you can run your own
optimizer/sampler. If you want a fully fledged library out of the box, you need to install `optax`, `numpyro`, and
`h5py` as well.tes

## Usage

```{toctree}
:maxdepth: 2

quickstart
tutorials
api
```

## Differences between _scarlet_ and _scarlet2_

[_scarlet_](https://pmelchior.github.io/scarlet/) was introduced by
[Melchior et al. (2018)](https://doi.org/10.1016/j.ascom.2018.07.001) to solve the deblending problem for the Rubin
Observatory. A stripped down version of it (developed by Fred Moolekamp) runs as part of the Rubin Observatory software
stack and is used for their data releases. We now call this version _scarlet1_.

_scarlet2_ follows very similar concepts. So, what's different?

### Model specification

_scarlet1_ is designed for a specific purpose: deblending for the Rubin Observatory. That has implications for the
quality and type of data it needs to work with. **_scarlet2_ is much more flexible to handle complex sources
configurations**, e.g. strong-lensing systems, supernova host galaxies, transient sources, etc.

This flexibility led us to carefully design a "language" to construct sources. It allows new source combinations and
is more explicit about the parameters and their initialization.

### Compute

Because some of the constraints in _scarlet1_ are expensive to evaluate, they are implemented in C++, which requires
the installation of a lot of additional code just to get it to run. **_scarlet2_ is implemented entirely in jax.**
A combination of `conda` and `pip` will get it installed. Unlike _scarlet1_, it will also run on GPUs and TPUs, and
performs just-in-time compilation of the model evaluation.

In addition, we can now interface with deep learning methods. In particular, we can employ neural networks as
data-driven priors, which helps break the degeneracies that arise when multiple components need to be
fit at the same time.

### Constraints

_scarlet1_ uses constrained optimization to help with fitting degeneracies, but that requires non-standard
(namely proximal) optimization because these constraints are not differentiable.
That can lead to problems with calibration, but, more importantly, it prevents the use of gradient-based
optimization or sampling. As a result, we could never calculate errors for _scarlet1_ models.
**_scarlet2_ uses only constraints that can be differentiated.** It supports any continuous optimization or sampling
method, including error estimates.

## Ideas, Questions or Problems?

If you have any of those, head over to our [github repo](https://github.com/pmelchior/scarlet2/) and create an issue.
