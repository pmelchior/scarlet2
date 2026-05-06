"""Math sanity checks for the responsibility regularizer.

Targets the pure-tensor helper ``_responsibility_from_stack`` so the tests
do not require a Scene. Three regimes are checked:

  (a) disjoint sources                     -> R near zero
  (b) honest equal-amplitude overlap       -> R ~ 2 * log 2
  (c) 'parasitic' flux from a smooth source under a compact source
      -> R strictly larger than the clean configuration

We also check that gradients with respect to the parasitic amplitude are
positive, i.e. point in the direction that reduces parasitic flux.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scarlet2 import Responsibility
from scarlet2.infer import _responsibility_from_stack


def _gaussian(shape, center, sigma, amp=1.0):
    yy, xx = jnp.mgrid[: shape[0], : shape[1]]
    r2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    return amp * jnp.exp(-0.5 * r2 / sigma**2)


SHAPE = (64, 64)


def test_disjoint_sources_near_zero():
    A = _gaussian(SHAPE, (16, 16), sigma=3.0, amp=10.0)
    B = _gaussian(SHAPE, (48, 48), sigma=3.0, amp=10.0)
    R = float(_responsibility_from_stack(jnp.stack([A, B]), Responsibility(weight=1.0)))
    assert R < 1e-3


def test_equal_full_overlap_matches_baseline():
    # Two equal-amplitude blobs at the same location: gamma_k = 1/2 everywhere
    # in the support, so each source contributes -sum w_k log(1/2) = log 2.
    A = _gaussian(SHAPE, (32, 32), sigma=4.0, amp=10.0)
    B = _gaussian(SHAPE, (32, 32), sigma=4.0, amp=10.0)
    R = float(_responsibility_from_stack(jnp.stack([A, B]), Responsibility(weight=1.0)))
    assert R == pytest.approx(2 * np.log(2), abs=0.05)


def test_parasitic_flux_increases_R():
    # Big smooth A under a small bright B. Adding a parasitic bump on A right
    # under B should increase R relative to the clean configuration.
    A_clean = _gaussian(SHAPE, (32, 32), sigma=10.0, amp=2.0)
    B = _gaussian(SHAPE, (32, 32), sigma=1.5, amp=20.0)
    A_parasitic = A_clean + _gaussian(SHAPE, (32, 32), sigma=1.5, amp=8.0)

    R_clean = float(_responsibility_from_stack(jnp.stack([A_clean, B]), Responsibility(weight=1.0)))
    R_parasitic = float(_responsibility_from_stack(jnp.stack([A_parasitic, B]), Responsibility(weight=1.0)))
    assert R_parasitic > R_clean


def test_gradient_pushes_parasitic_amplitude_down():
    A_clean = _gaussian(SHAPE, (32, 32), sigma=10.0, amp=2.0)
    B = _gaussian(SHAPE, (32, 32), sigma=1.5, amp=20.0)

    def R_of_amp(amp):
        A_var = A_clean + _gaussian(SHAPE, (32, 32), sigma=1.5, amp=amp)
        return _responsibility_from_stack(jnp.stack([A_var, B]), Responsibility(weight=1.0))

    grad_at_zero = float(jax.grad(R_of_amp)(0.0))
    grad_at_eight = float(jax.grad(R_of_amp)(8.0))
    # Loss minimization moves -grad, so positive grad means parasitic amp shrinks.
    assert grad_at_zero > 0
    assert grad_at_eight > 0


# -----------------------------------------------------------------------------
# Multi-band: band_summed vs joint mode on (K, C, H, W) stacks.
# -----------------------------------------------------------------------------


def test_band_summed_collapses_channel_axis_to_2d_math():
    # The band_summed branch must be equivalent to first summing over the
    # channel axis and then running the 2D joint math on the summed result.
    morph_A = _gaussian(SHAPE, (32, 32), sigma=10.0, amp=2.0)
    morph_B = _gaussian(SHAPE, (32, 32), sigma=1.5, amp=20.0)
    A = jnp.stack([0.7 * morph_A, 0.3 * morph_A], axis=0)  # (C, H, W)
    B = jnp.stack([0.4 * morph_B, 0.6 * morph_B], axis=0)
    stack3d = jnp.stack([A, B], axis=0)  # (K, C, H, W)
    stack2d = jnp.stack([A.sum(axis=0), B.sum(axis=0)], axis=0)  # (K, H, W)

    R_3d = float(_responsibility_from_stack(stack3d, Responsibility(weight=1.0, mode="band_summed")))
    R_2d = float(_responsibility_from_stack(stack2d, Responsibility(weight=1.0, mode="joint")))
    assert R_3d == pytest.approx(R_2d, rel=1e-6)


def test_band_summed_and_joint_disagree_on_color_separated_overlap():
    # Two sources at the same spatial location with disjoint SEDs:
    # A bright in band 0, B bright in band 1.
    # joint mode: per (c, x), one source dominates -> R is small.
    # band_summed: spatially overlapping with comparable totals -> R ~ 2 log 2.
    morph = _gaussian(SHAPE, (32, 32), sigma=4.0, amp=1.0)
    A = jnp.stack([1.0 * morph, 0.05 * morph], axis=0)
    B = jnp.stack([0.05 * morph, 1.0 * morph], axis=0)
    stack = jnp.stack([A, B], axis=0)

    R_summed = float(_responsibility_from_stack(stack, Responsibility(weight=1.0, mode="band_summed")))
    R_joint = float(_responsibility_from_stack(stack, Responsibility(weight=1.0, mode="joint")))

    assert R_summed > R_joint
    assert R_summed == pytest.approx(2 * np.log(2), abs=0.1)


def test_3d_disjoint_sources_near_zero_in_both_modes():
    morph_A = _gaussian(SHAPE, (16, 16), sigma=3.0, amp=10.0)
    morph_B = _gaussian(SHAPE, (48, 48), sigma=3.0, amp=10.0)
    A = jnp.stack([morph_A, 0.5 * morph_A], axis=0)
    B = jnp.stack([0.5 * morph_B, morph_B], axis=0)
    stack = jnp.stack([A, B], axis=0)

    assert float(_responsibility_from_stack(stack, Responsibility(weight=1.0, mode="band_summed"))) < 1e-3
    assert float(_responsibility_from_stack(stack, Responsibility(weight=1.0, mode="joint"))) < 1e-3


def test_per_band_falls_back_to_joint_on_2d_input():
    # No channel axis to separate by: per_band and joint must agree.
    A = _gaussian(SHAPE, (32, 32), sigma=4.0, amp=10.0)
    B = _gaussian(SHAPE, (32, 32), sigma=4.0, amp=10.0)
    stack = jnp.stack([A, B])  # (K, H, W)

    R_per_band = float(_responsibility_from_stack(stack, Responsibility(weight=1.0, mode="per_band")))
    R_joint = float(_responsibility_from_stack(stack, Responsibility(weight=1.0, mode="joint")))
    assert R_per_band == pytest.approx(R_joint, rel=1e-6)


def test_per_band_diverges_from_joint_on_color_separated_overlap():
    # Same setup as the band_summed/joint divergence test: A and B fully
    # overlap spatially with disjoint SEDs. Per-band normalization weights
    # the per-channel cross-entropy by the per-channel flux fractions, which
    # is not equivalent to either band_summed or joint.
    morph = _gaussian(SHAPE, (32, 32), sigma=4.0, amp=1.0)
    A = jnp.stack([1.0 * morph, 0.05 * morph], axis=0)
    B = jnp.stack([0.05 * morph, 1.0 * morph], axis=0)
    stack = jnp.stack([A, B], axis=0)

    R_per_band = float(_responsibility_from_stack(stack, Responsibility(weight=1.0, mode="per_band")))
    R_joint = float(_responsibility_from_stack(stack, Responsibility(weight=1.0, mode="joint")))
    R_band_summed = float(_responsibility_from_stack(stack, Responsibility(weight=1.0, mode="band_summed")))
    assert R_per_band != pytest.approx(R_joint, rel=1e-3)
    assert R_per_band != pytest.approx(R_band_summed, rel=1e-3)


def test_invalid_mode_raises_on_construction():
    with pytest.raises(ValueError, match="mode"):
        Responsibility(weight=1.0, mode="bogus")


def test_per_band_detects_single_channel_parasitic_flux():
    # Big smooth A in both channels. Compact B exists only in channel 0.
    # Add a parasitic bump on A's channel 0 right under B; per_band's R must
    # increase relative to clean.
    morph_A = _gaussian(SHAPE, (32, 32), sigma=10.0, amp=2.0)
    morph_B = _gaussian(SHAPE, (32, 32), sigma=1.5, amp=20.0)
    bump = _gaussian(SHAPE, (32, 32), sigma=1.5, amp=8.0)

    A_clean = jnp.stack([morph_A, morph_A], axis=0)
    A_parasitic = jnp.stack([morph_A + bump, morph_A], axis=0)
    B = jnp.stack([morph_B, jnp.zeros_like(morph_B)], axis=0)

    R_clean = float(_responsibility_from_stack(jnp.stack([A_clean, B]), Responsibility(weight=1.0, mode="per_band")))
    R_parasitic = float(_responsibility_from_stack(jnp.stack([A_parasitic, B]), Responsibility(weight=1.0, mode="per_band")))
    assert R_parasitic > R_clean
