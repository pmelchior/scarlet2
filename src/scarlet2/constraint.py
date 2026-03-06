"""Numpyro constraints for scarlet2 parameters"""

import jax.numpy as jnp
from numpyro.distributions import constraints
from numpyro.distributions.constraints import ParameterFreeConstraint
from numpyro.distributions.transforms import ParameterFreeTransform, biject_to


class _UnitDisk(ParameterFreeConstraint):
    """Constraint for 2-element vectors lying in the open unit disk.

    A vector ``(e1, e2)`` satisfies this constraint iff ``e1**2 + e2**2 < 1``,
    i.e. the vector lies strictly inside the unit circle.

    The primary use case is the ellipticity parameter of
    :py:class:`~scarlet2.morphology.ProfileMorphology`, where values on or outside
    the unit circle correspond to unphysical (degenerate) ellipticities.

    See Also
    --------
    unit_disk : singleton instance of this class
    _UnitDiskTransform : the associated bijective transform
    """

    event_dim = 1

    def __call__(self, x):
        """Check whether `x` lies inside the open unit disk.

        Parameters
        ----------
        x: array
            Array with last dimension of size 2.

        Returns
        -------
        array
            Boolean array with the last dimension reduced.
        """
        return jnp.sum(x**2, axis=-1) < 1

    def feasible_like(self, prototype):
        """Return a feasible point with the same shape as `prototype`.

        Parameters
        ----------
        prototype: array
            Array whose shape is used as the template.

        Returns
        -------
        array
            Zero array of the same shape (the origin is always feasible).
        """
        return jnp.zeros_like(prototype)


class _UnitDiskTransform(ParameterFreeTransform):
    """Bijective transform from :math:`\\mathbb{R}^2` to the open unit disk.

    Uses the homeomorphism

    .. math::

        f(\\mathbf{x}) = \\frac{\\mathbf{x}}{1 + \\|\\mathbf{x}\\|_2}

    which maps any vector in :math:`\\mathbb{R}^2` to the open unit disk.
    Its inverse is :math:`f^{-1}(\\mathbf{y}) = \\mathbf{y} / (1 - \\|\\mathbf{y}\\|_2)`.

    The log absolute Jacobian determinant is :math:`-3 \\log(1 + \\|\\mathbf{x}\\|_2)`.

    Notes
    -----
    The Jacobian matrix of :math:`f` is

    .. math::

        J_{ij} = \\frac{\\delta_{ij}}{1+r} - \\frac{x_i x_j}{r(1+r)^2},
        \\quad r = \\|\\mathbf{x}\\|_2.

    Applying the matrix determinant lemma gives :math:`\\det J = (1+r)^{-3}`.

    A numerically safe norm :math:`\\sqrt{\\|\\mathbf{x}\\|^2 + \\varepsilon}` is used
    throughout to avoid NaN gradients at the origin, where
    :math:`\\partial\\|\\mathbf{x}\\|/\\partial\\mathbf{x} = \\mathbf{x}/\\|\\mathbf{x}\\|`
    would otherwise evaluate to :math:`0/0`.

    See Also
    --------
    unit_disk : the associated constraint singleton
    """

    domain = constraints.real_vector
    sign = 1
    _eps = 1e-12

    @staticmethod
    def _safe_norm(x):
        return jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + _UnitDiskTransform._eps)

    def __call__(self, x):
        """Apply the forward transform.

        Parameters
        ----------
        x: array
            Unconstrained 2-element vector.

        Returns
        -------
        array
            2-element vector inside the open unit disk.
        """
        norm = self._safe_norm(x)
        return x / (1.0 + norm)

    def _inverse(self, y):
        """Apply the inverse transform.

        Parameters
        ----------
        y: array
            2-element vector inside the open unit disk.

        Returns
        -------
        array
            Unconstrained 2-element vector.
        """
        norm = self._safe_norm(y)
        return y / (1.0 - norm)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        """Log absolute Jacobian determinant of the forward transform.

        Parameters
        ----------
        x: array
            Input in unconstrained space.
        y: array
            Output in the unit disk (unused; kept for API compatibility).
        intermediates: optional
            Unused.

        Returns
        -------
        array
            Scalar log absolute determinant, :math:`-3 \\log(1 + \\|\\mathbf{x}\\|_2)`.
        """
        norm = self._safe_norm(x)[..., 0]
        return -3.0 * jnp.log1p(norm)


unit_disk = _UnitDisk()
"""Singleton constraint for 2-element ellipticity vectors in the open unit disk.

A vector ``(e1, e2)`` satisfies this constraint iff ``e1**2 + e2**2 < 1``.
Use as the ``constraint`` argument of :py:class:`~scarlet2.module.Parameter`.

Examples
--------
>>> import scarlet2.constraint as sc2c
>>> import jax.numpy as jnp
>>> sc2c.unit_disk.check(jnp.array([0.3, 0.4]))  # 0.09 + 0.16 = 0.25 < 1
Array(True, dtype=bool)
>>> sc2c.unit_disk.check(jnp.array([0.8, 0.8]))  # 0.64 + 0.64 = 1.28 > 1
Array(False, dtype=bool)

See Also
--------
_UnitDisk, _UnitDiskTransform
"""

_UnitDiskTransform.codomain = unit_disk


@biject_to.register(_UnitDisk)
def _transform_to_unit_disk(constraint):
    return _UnitDiskTransform()
