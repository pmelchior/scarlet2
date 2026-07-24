# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os

# -- Example data ------------------------------------------------------------
# The notebooks download their data with hf_hub_download. Each notebook runs in
# its own kernel, and every call hits the Hub with a HEAD request even when the
# file is already cached, which trips the anonymous per-IP rate limit (429).
# Fetch each file once here, then put the notebook kernels into offline mode so
# they read straight from the local cache without any further requests.
HF_REPO_ID = "astro-data-lab/scarlet-test-data"
HF_FILES = [
    "hsc_cosmos_35.npz",
    "lsbg.pkl",
    "multiresolution_tutorial/data.fits.gz",
    "transient_tutorial/data.fits.gz",
]


def _prefetch_example_data():
    from huggingface_hub import hf_hub_download

    for filename in HF_FILES:
        hf_hub_download(repo_id=HF_REPO_ID, filename=filename, repo_type="dataset")


try:
    _prefetch_example_data()
except Exception as e:  # noqa: BLE001
    # Let the notebooks try on their own rather than failing the whole build.
    print(f"WARNING: could not prefetch example data ({e}); notebooks will download individually")
else:
    os.environ["HF_HUB_OFFLINE"] = "1"

# -- Project information -----------------------------------------------------

project = "scarlet2"
copyright = "2025, Peter Melchior"
author = "Peter Melchior"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx_issues",
    "myst_nb",
]
master_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "jupyter_execute"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "scarlet2"
html_favicon = "_static/icon.png"
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/pmelchior/scarlet2",
    "repository_branch": "main",
    "logo": {
        "image_light": "_static/logo_light.svg",
        "image_dark": "_static/logo_dark.svg",
    },
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "show_toc_level": 3,
}
html_baseurl = "https://scarlet2.readthedocs.io/en/latest/"

autoclass_content = "both"
autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False
autodoc_type_aliases = {
    "eqx.Module": "equinox.Module",
    "jnp.ndarray": "jax.numpy.array",
}

intersphinx_mapping = {
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
}

issues_github_path = "pmelchior/scarlet2"

nb_execution_timeout = 60
nb_execution_excludepatterns = ["_build", "jupyter_execute"]

# MyST parser extensions (myst-nb uses the MyST Markdown parser for notebooks).
# "dollarmath" enables $...$ inline and $$...$$ block math like JupyterLab;
# "amsmath" enables LaTeX environments such as \begin{align}.
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True
