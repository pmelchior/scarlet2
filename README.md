# _scarlet2_

[![PyPI](https://img.shields.io/pypi/v/scarlet2)](https://pypi.python.org/pypi/scarlet2/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat&logo=readthedocs)](https://scarlet2.readthedocs.io/)
[![status](https://joss.theoj.org/papers/25e966ab45b4411dbf05fa749ca70b82/status.svg)](https://joss.theoj.org/papers/25e966ab45b4411dbf05fa749ca70b82)

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

## Citation

If you use this software, please cite the associated JOSS paper:

```
@article{scarlet2, 
    author = {Melchior, Peter and Ward, Charlotte and Remy, Benjamin and Wiemann, Matt L. and Siegel, Jared}, 
    title = {scarlet2: Astronomical scene modeling in JAX}, 
    journal = {Journal of Open Source Software},
    year = {2026}, 
    volume = {11}, 
    number = {120}, 
    pages = {9646}, 
    doi = {10.21105/joss.09646}, 
    url = {https://doi.org/10.21105/joss.09646}, 
    publisher = {The Open Journal} 
}
```
