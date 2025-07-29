---
title: 'scarlet2: Astronomical scene modeling in jax'

tags:
  - Python
  - image analysis
  - astronomy
  - cosmology
  - deblending
  - transients

authors:
  - name: Peter Melchior
    orcid: 0000-0002-8873-5065
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Charlotte Ward
    orcid: 0000-0002-4557-6682
    affiliation: 1
  - name: Benjamin Remy
    orcid: 0000-0002-0978-5612
    affiliation: 1
  - name: Matt L. Sampson
    orcid: 0000-0001-5748-5393
    affiliation: 1
  - name: Jared Siegel
    orcid: 0000-0002-9337-0902
    affiliation: 1

affiliations:
  - name: Princeton University, United States
    index: 1
    ror: 00hx57361

date: 25 July 2025

bibliography: paper.bib
---

# Summary

Large astronomical imaging surveys now contain billions of celestial sources, from stars in our Galaxy to galaxies
towards the edge of the observable Universe. Because of improvements in resolution and sensitivity of current and future
observing instruments, the images reveal more information than ever before. But they are also more difficult to analyze
because galaxies exhibit complex morphologies, which cannot be described by traditional parametric models. And because
there are so many sources, they routinely overlap with each other, either due to physical interactions or due to their
close alignment along the line of sight. To extract all information of interest and avoid biases from incorrect modeling
assumptions, it is therefore necessary to simultaneously model full scenes comprising many sources instead of analyzing
each source separately, and each of the source models may itself need to be composed of multiple, morphological complex
components.

# Statement of need

`scarlet2` is a Python package for full-scene modeling in observational astronomy. It inherits modeling assumptions from
`scarlet` [@scarlet], namely that a scene comprises multiple sources, each source comprises multiple
components, and
each component is determined by a spectrum model and a morphology model, whose outer product represents the light
emission in a sky region as a hyperspectral data cube (wavelength $\times$ height $\times$ width). `scarlet2` retains
the object-oriented paradigm and many classes and functions from `scarlet`, but augments standard Python with the `jax`
library [@jax2018github] for automatic differentiation and just-in-time compilation.

`scarlet2` acts as a flexible, modular, and extendable modeling language for celestial sources that combines parametric
and non-parametric models to describe complex scenarios such as multi-source blending, strong-lensing systems,
supernovae and their host galaxies, etc. As a modeling language, `scarlet2` is agnostic about the optimization or
inference method the user wants to employ, but it provides methods to optimize the likelihood function or sample from
the posterior, which utilize the `optax` package [@deepmind2020jax] or the `numpyro` inference framework
[@pyro-2019; @phan-2019], respectively. The likelihood of multiple
observations (at different resolutions, wavelengths, or observing epochs) times can be combined for a joint model of
static and transient sources. To match the coordinates from different observations, `scarlet2` utilizes the `Astropy`
package [@astropy]. `scarlet2` can also interface with deep learning methods. Besides being natively portable
to GPUs,
parameters can be specified with neural networks as data-driven priors, which helps break the degeneracies that arise
when multiple components are fit simultaneously [@sampson-2024].

![Scene with seven detected sources in multi-band images from the Hyper Suprime-Cam Subaru Strategic Program.
Each source is modelled with a non-parametric spectrum and morphology (1st panel), the entire scene is then convolved
with the telescope's point spread function (2nd panel) and compared to the observations (3rd panel).
The residuals (4th panel) reveal the presence of undetected sources and source components (e.g. in the center of source
#1).](scarlet2_model.png)

To support the wide range of scientific studies that will be made with large sky surveys, `scarlet2` was designed with
flexibility and ease of use in mind. Several publications have developed and demonstrated new capabilities, including
modeling of interstellar dust embedded in distant galaxies
[@siegel-2025] and of transient sources such as active galactic nuclei [@ward-2025] and tidal disruption
events
[@yao-2025].
Future developments will integrate into cloud-based science platforms, provide support for users to make effective
modeling choices and to validate their inference results, and create a robust processing pipeline for joint pixel-level
analyses of surveys from the Vera C. Rubin Observatory, the Euclid mission, the Nancy Grace Roman Space Telescope, and
the La Silla Schmidt Southern Survey.

# Acknowledgements

We acknowledge contributions from
the [LINCC Frameworks Incubator Program](https://lsstdiscoveryalliance.org/programs/lincc-frameworks/incubators/), in
particular from software engineers Max West, Drew Oldag, and Sean McGuire, in adopting comprehensive software workflows
through the Python Project Template [@oldag-2024] and creating a user-focused recommendation and validation
suite.

# References