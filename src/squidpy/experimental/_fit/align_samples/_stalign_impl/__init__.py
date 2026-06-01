"""Ported STalign JAX LDDMM solver (provenance: scverse/squidpy#1150).

Pure numerics only -- no :class:`~squidpy.experimental._fit.Estimator`. The
estimator adapter lives one level up in
:mod:`squidpy.experimental._fit.align_samples._stalign`. JAX is imported at
module load here, so this subpackage must only be imported lazily (the adapter
does so inside ``fit``, after checking requirements).
"""

from __future__ import annotations
