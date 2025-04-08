# _scarlet2_

_scarlet2_ is an open-source python library for modeling astronomical sources from multi-band, multi-epoch, and
multi-instrument data. It provides non-parametric and parametric models, can handle source overlap (aka blending), and
can integrate neural network priors. It's designed to be modular, flexible, and powerful.

_scarlet2_ is implemented in [jax](http://jax.readthedocs.io/), layered on top of
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
`h5py` as well.