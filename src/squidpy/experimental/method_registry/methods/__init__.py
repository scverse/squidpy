"""Estimator implementations, one subpackage per method family.

Importing this package imports each family subpackage, whose modules register
their estimators into the family registries in
:mod:`squidpy.experimental.method_registry.registry`. Each subpackage stays
cheap to import -- heavy/optional dependencies (e.g. JAX) are pulled in lazily,
only when an estimator actually runs.
"""

from __future__ import annotations

from squidpy.experimental.method_registry.methods import align_landmarks, align_samples  # noqa: F401
