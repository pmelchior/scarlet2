# Contributing to scarlet2

Thank you for your interest in contributing to **scarlet2**. Whether you're fixing a bug, proposing a new
feature, improving documentation, or sharing a use case, your contributions are welcome.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Tests](#tests)
- [Documentation](#documentation)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)
- [Questions and Discussions](#questions-and-discussions)

---

## Code of Conduct

This project follows a standard open-source code of conduct. Please be respectful, constructive, and inclusive
in all interactions. Harassment or discriminatory behavior of any kind will not be tolerated.

---

## Ways to Contribute

- **Bug reports**: Found something broken? Open an [issue](https://github.com/pmelchior/scarlet2/issues).
- **Feature requests**: Have an idea for a new model, prior, or utility? Start
  a [Discussion](https://github.com/pmelchior/scarlet2/discussions) first to gauge interest.
- **Code contributions**: Fix a bug, implement a feature, or improve performance.
- **Documentation**: Improve docstrings, tutorials, or the ReadTheDocs pages.
- **Benchmarks**: Add or improve benchmarks for model performance.
- **Examples and notebooks**: Share use cases (strong lensing, transients, deblending, etc.) as notebooks in
  `docs/`.

---

## Getting Started

### Prerequisites

scarlet2 is built on [JAX](https://github.com/google/jax)
and [equinox](https://github.com/patrick-kidger/equinox). Before setting up the dev environment, install JAX
with the appropriate `jaxlib` for your platform (CPU, GPU, or TPU) following
the [official JAX installation instructions](https://github.com/google/jax#installation).

### Fork and Clone

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/scarlet2.git
   cd scarlet2
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/pmelchior/scarlet2.git
   ```

### Install in Development Mode

Install scarlet2 along with its optional dependencies:

```bash
pip install -e ".[dev]"
```

For the full feature set (optimization, sampling, and I/O), also install:

```bash
pip install optax numpyro h5py
```

---

## Development Workflow

1. **Sync with upstream** before starting work:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b my-feature-or-fix
   ```

3. Make your changes, write tests, and update documentation as needed.

4. **Run tests** to make sure nothing is broken (see [Tests](#tests)).

5. **Commit** with a clear, descriptive message:
   ```bash
   git commit -m "Fix: correct morphology normalization in Frame.render"
   ```

6. **Push** your branch and open a pull request against `main`.

---

## Coding Standards

- **Style**: Follow [PEP 8](https://peps.python.org/pep-0008/). You can use `ruff` or `flake8` for linting.
- **Type hints**: Add type annotations where they improve clarity.
- **Docstrings**: Use NumPy-style docstrings for all public functions and classes.
- **JAX idioms**: Prefer pure functions and avoid side effects where possible. Be mindful of JAX's constraints
  on in-place mutation and Python control flow — use `jax.lax` control flow primitives when values are traced.
- **equinox conventions**: New model components should subclass `eqx.Module` and expose parameters as fields
  consistent with the rest of the codebase.

---

## Tests

Tests live in `tests/scarlet2/` and use [pytest](https://pytest.org/).

Run the full test suite:

```bash
pytest tests/
```

Run a specific test file:

```bash
pytest tests/scarlet2/test_scene.py
```

When adding new functionality, please include:

- At least one unit test covering the happy path.
- A test for edge cases or expected failure modes where applicable.

If your change touches performance-sensitive code paths, consider adding or updating entries in `benchmarks/`.

---

## Documentation

Documentation is built with [Sphinx](https://www.sphinx-doc.org/) and hosted
on [ReadTheDocs](https://scarlet2.readthedocs.io). Source files are in `docs/`.

To build the docs locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

Then open `docs/_build/html/index.html` in your browser.

**Guidelines:**

- Every public class and function should have a docstring.
- Tutorials and example notebooks should be runnable end-to-end with a standard scarlet2 install.
- When adding a new model component, add a short worked example demonstrating its use.

---

## Submitting a Pull Request

Before opening a PR, please ensure:

- [ ] All existing tests pass (`pytest tests/`).
- [ ] New code is covered by tests.
- [ ] Docstrings are complete for any new public API.
- [ ] The PR description explains *what* changed and *why*.
- [ ] For non-trivial changes, there is a linked issue or discussion.

PRs are reviewed by the maintainers. Please be patient — this is an academic research project and review
timelines may vary. Constructive feedback will be provided; feel free to ask for clarification.

---

## Reporting Issues

When opening a bug report, please include:

1. A minimal reproducible example.
2. The version of scarlet2, JAX, and equinox you are using (`pip show scarlet2 jax equinox`).
3. Your platform (CPU / GPU / TPU, OS).
4. The full traceback if an exception is raised.

Feature requests and questions about design decisions are better suited
for [Discussions](https://github.com/pmelchior/scarlet2/discussions) than issues.

---

## Questions and Discussions

Use [GitHub Discussions](https://github.com/pmelchior/scarlet2/discussions) for:

- Questions about usage or the API.
- Ideas for new features or design proposals.
- Sharing results or use cases built with scarlet2.

For quick questions, feel free to reach out to the maintainers via issues or discussions — there is no mailing
list or Slack at this time.

---

## Attribution

If you use scarlet2 in published research, please cite:

> Melchior et al. (2018) for the original scarlet algorithm, and the scarlet2 documentation
> at https://scarlet2.readthedocs.io for the JAX reimplementation.

Thank you for helping make scarlet2 better!
