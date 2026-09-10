# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103
# ruff: noqa: N802
"""Math sanity checks for the pair-similarity regularizer.

Targets the pure-tensor helper ``_pair_similarity_from_stack`` so the tests
do not require a Scene. Checks:

  (a) disjoint sources                        -> R near zero (no morph overlap)
  (b) parasitic flux raises R relative to the clean configuration
  (c) the gradient w.r.t. parasitic amplitude is positive (loss minimization
      reduces parasitic flux)
  (d) SED contrast modulates R: identical SEDs maximize, orthogonal SEDs
      drive R to zero
  (e) bit-identical no-op at weight=0 short-circuit
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scarlet2 import PairSimilarity
from scarlet2.infer import _cosine_matrix, _pair_similarity_from_stack


def _gaussian(shape, center, sigma, amp=1.0):
    yy, xx = jnp.mgrid[: shape[0], : shape[1]]
    r2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    return amp * jnp.exp(-0.5 * r2 / sigma**2)


SHAPE = (64, 64)


def test_disjoint_sources_near_zero_2d():
    A = _gaussian(SHAPE, (16, 16), sigma=3.0, amp=10.0)
    B = _gaussian(SHAPE, (48, 48), sigma=3.0, amp=10.0)
    R = float(_pair_similarity_from_stack(jnp.stack([A, B]), PairSimilarity(weight=1.0)))
    assert R < 1e-3


def test_identical_morphologies_with_identical_seds_gives_one():
    # Single pair, M_A = M_B, single-band -> sigma = 1, rho = 1, sum/2 = 1.
    A = _gaussian(SHAPE, (32, 32), sigma=4.0, amp=10.0)
    R = float(_pair_similarity_from_stack(jnp.stack([A, A]), PairSimilarity(weight=1.0)))
    assert pytest.approx(1.0, rel=1e-6) == R


def test_parasitic_flux_increases_R_2d():
    # Big smooth A under a small bright B. Adding a B-shaped bump on A
    # should raise the morphology cosine and thus R.
    A_clean = _gaussian(SHAPE, (32, 32), sigma=10.0, amp=2.0)
    B = _gaussian(SHAPE, (32, 32), sigma=1.5, amp=20.0)
    A_parasitic = A_clean + _gaussian(SHAPE, (32, 32), sigma=1.5, amp=8.0)

    R_clean = float(_pair_similarity_from_stack(jnp.stack([A_clean, B]), PairSimilarity(weight=1.0)))
    R_para = float(_pair_similarity_from_stack(jnp.stack([A_parasitic, B]), PairSimilarity(weight=1.0)))
    assert R_para > R_clean


def test_gradient_pushes_parasitic_amplitude_down_2d():
    A_clean = _gaussian(SHAPE, (32, 32), sigma=10.0, amp=2.0)
    B = _gaussian(SHAPE, (32, 32), sigma=1.5, amp=20.0)

    def R_of_amp(amp):
        A_var = A_clean + _gaussian(SHAPE, (32, 32), sigma=1.5, amp=amp)
        return _pair_similarity_from_stack(jnp.stack([A_var, B]), PairSimilarity(weight=1.0))

    grad_at_zero = float(jax.grad(R_of_amp)(0.0))
    grad_at_eight = float(jax.grad(R_of_amp)(8.0))
    # Loss minimization moves -grad: positive grad => parasitic amp shrinks.
    assert grad_at_zero > 0
    assert grad_at_eight > 0


# -----------------------------------------------------------------------------
# Multi-band: SED-cosine factor
# -----------------------------------------------------------------------------


def test_orthogonal_seds_zero_R_under_full_overlap():
    # Two sources with identical morphology but disjoint SEDs (one bright in
    # band 0, the other in band 1). sigma_AB = 0 -> R = 0.
    morph = _gaussian(SHAPE, (32, 32), sigma=4.0, amp=1.0)
    A = jnp.stack([morph, jnp.zeros_like(morph)], axis=0)
    B = jnp.stack([jnp.zeros_like(morph), morph], axis=0)
    R = float(_pair_similarity_from_stack(jnp.stack([A, B]), PairSimilarity(weight=1.0)))
    assert pytest.approx(0.0, abs=1e-6) == R


def test_identical_seds_recover_2d_morphology_cosine():
    # With identical SEDs, sigma_AB = 1, so R reduces to morphology cosine on
    # the band-summed maps.
    morph_A = _gaussian(SHAPE, (32, 32), sigma=10.0, amp=2.0)
    morph_B = _gaussian(SHAPE, (32, 32), sigma=1.5, amp=20.0)
    A = jnp.stack([0.5 * morph_A, 0.5 * morph_A], axis=0)
    B = jnp.stack([0.5 * morph_B, 0.5 * morph_B], axis=0)
    stack3d = jnp.stack([A, B], axis=0)
    stack2d = jnp.stack([morph_A, morph_B], axis=0)

    R_3d = float(_pair_similarity_from_stack(stack3d, PairSimilarity(weight=1.0)))
    R_2d = float(_pair_similarity_from_stack(stack2d, PairSimilarity(weight=1.0)))
    assert R_3d == pytest.approx(R_2d, rel=1e-5)


def test_sed_contrast_monotonically_reduces_R():
    # Fixed morphology overlap, vary SED of B from identical to f_A through
    # near-orthogonal. R must decrease monotonically.
    morph_A = _gaussian(SHAPE, (32, 32 - 2), sigma=8.0, amp=2.0)
    morph_B = _gaussian(SHAPE, (32, 32 + 2), sigma=2.0, amp=20.0)

    f_A_vec = jnp.array([1.0, 1.0, 1.0])
    Rs = []
    for b in [1.0, 0.7, 0.4, 0.2, 0.05]:
        f_B_vec = jnp.array([1.0, b, b**2])
        # Build (K, C, H, W) by tensoring the SED with the morphology.
        A = f_A_vec[:, None, None] * morph_A[None, :, :]
        B = f_B_vec[:, None, None] * morph_B[None, :, :]
        Rs.append(float(_pair_similarity_from_stack(jnp.stack([A, B]), PairSimilarity(weight=1.0))))
    # Strictly decreasing.
    for r_high, r_low in zip(Rs, Rs[1:], strict=False):
        assert r_high > r_low


# -----------------------------------------------------------------------------
# Three sources: pair sum is over unordered off-diagonal pairs.
# -----------------------------------------------------------------------------


def test_three_sources_pair_sum():
    # Three identical morphologies with identical SEDs: every pair has
    # sigma * rho = 1, three unordered pairs -> R = 3.
    A = _gaussian(SHAPE, (32, 32), sigma=4.0, amp=1.0)
    R = float(_pair_similarity_from_stack(jnp.stack([A, A, A]), PairSimilarity(weight=1.0)))
    assert pytest.approx(3.0, rel=1e-6) == R


# -----------------------------------------------------------------------------
# Weight short-circuit
# -----------------------------------------------------------------------------


def test_weight_zero_short_circuits_to_zero_in_penalty():
    # The Scene-facing ``_pair_similarity_penalty`` returns 0.0 when
    # weight=0, regardless of inputs. We exercise the helper directly here
    # since it does not depend on a Scene.
    from scarlet2.infer import _pair_similarity_penalty

    class _FakeScene:
        sources = []

        def evaluate_source(self, s):
            raise AssertionError("must not be called when weight=0")

    R = _pair_similarity_penalty(_FakeScene(), PairSimilarity(weight=0.0))
    assert float(R) == 0.0


# -----------------------------------------------------------------------------
# _cosine_matrix sanity
# -----------------------------------------------------------------------------


def test_cosine_matrix_diagonal_is_one():
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.normal(size=(5, 7)))
    C = _cosine_matrix(X, eps=1e-12)
    diag = np.asarray(jnp.diagonal(C))
    np.testing.assert_allclose(diag, 1.0, rtol=1e-5)
