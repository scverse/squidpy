"""Slim container for a fitted stain reference.

Holds either a 3x3 stain matrix (Macenko/Vahadane, ships in PR 3) or a
pair of Ruderman Lab channel statistics (Reinhard, ships in PR 2). The
dataclass is intentionally minimal in this PR; cohort fields, persistence,
and provenance metadata land alongside their first consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import spatialdata as sd
    import xarray as xr
    from numpy.typing import DTypeLike

    from squidpy.experimental.im._stain._normalize import MethodParams

StainMethod = Literal["macenko", "vahadane", "reinhard"]
_DECOMPOSITION_METHODS: frozenset[str] = frozenset({"macenko", "vahadane"})
_VALID_METHODS: frozenset[str] = _DECOMPOSITION_METHODS | {"reinhard"}


def _coerce_finite(arr: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {out.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values.")
    return out


@dataclass(frozen=True)
class StainReference:
    """Container for a fitted stain reference.

    Parameters
    ----------
    method
        Fitting method: ``"macenko"``, ``"vahadane"``, or ``"reinhard"``.
    stain_matrix
        Shape ``(3, 3)`` unit-norm matrix in canonical order
        ``(H, E, complement)``. Required for decomposition methods.
    mu
        Shape ``(3,)`` Ruderman Lab channel means. Reinhard only.
    sigma
        Shape ``(3,)`` Ruderman Lab channel standard deviations. Reinhard
        only.
    white_point
        Shape ``(3,)`` per-channel white-point estimate. Required for
        decomposition methods (apply consumes it). Forbidden for Reinhard
        because Reinhard's color transfer operates in Ruderman Lab and
        does not model absorbance. There is no universal default; pass an
        estimate from your data (see ``estimate_white_point``).
    max_concentrations
        Shape ``(2,)`` reference per-stain (H, E) 99th-percentile concentrations
        - a fitted characterization of the reference's staining strength.
        Decomposition only, and diagnostic: the colour-basis ``apply`` transfers
        stain colour, not amount, so it does not consume this. Optional; forbidden
        for Reinhard.
    """

    method: StainMethod
    stain_matrix: np.ndarray | None = None
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None
    white_point: np.ndarray | None = None
    max_concentrations: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(f"Unknown method {self.method!r}; expected one of {sorted(_VALID_METHODS)}.")

        if self.method in _DECOMPOSITION_METHODS:
            if self.stain_matrix is None:
                raise ValueError(f"method={self.method!r} requires stain_matrix.")
            if self.mu is not None or self.sigma is not None:
                raise ValueError(f"method={self.method!r} forbids mu/sigma; pass them only for Reinhard.")
            if self.white_point is None:
                raise ValueError(f"method={self.method!r} requires white_point.")
            object.__setattr__(
                self,
                "stain_matrix",
                _coerce_finite(self.stain_matrix, shape=(3, 3), name="stain_matrix"),
            )
            bg = _coerce_finite(self.white_point, shape=(3,), name="white_point")
            if np.any(bg <= 0):
                raise ValueError("white_point must be strictly positive.")
            object.__setattr__(self, "white_point", bg)
            if self.max_concentrations is not None:
                maxc = _coerce_finite(self.max_concentrations, shape=(2,), name="max_concentrations")
                if np.any(maxc <= 0):
                    raise ValueError("max_concentrations must be strictly positive.")
                object.__setattr__(self, "max_concentrations", maxc)
        else:
            if self.mu is None or self.sigma is None:
                raise ValueError("method='reinhard' requires both mu and sigma.")
            if self.stain_matrix is not None:
                raise ValueError("method='reinhard' forbids stain_matrix.")
            if self.white_point is not None:
                raise ValueError(
                    "method='reinhard' forbids white_point; Reinhard's color "
                    "transfer is in Ruderman Lab and does not use a white point."
                )
            if self.max_concentrations is not None:
                raise ValueError("method='reinhard' forbids max_concentrations.")
            mu = _coerce_finite(self.mu, shape=(3,), name="mu")
            sigma = _coerce_finite(self.sigma, shape=(3,), name="sigma")
            if np.any(sigma <= 0):
                raise ValueError("sigma must be strictly positive.")
            object.__setattr__(self, "mu", mu)
            object.__setattr__(self, "sigma", sigma)

    def transform(
        self,
        sdata: sd.SpatialData,
        image_key: str,
        *,
        scale: str | Literal["auto"] = "auto",
        method_params: MethodParams = None,
        image_key_added: str | None = None,
        inplace: bool = True,
        output_dtype: DTypeLike | None = None,
        tissue_mask_key: str | None = None,
        preserve_background: bool = True,
    ) -> xr.DataArray | None:
        """Normalize an image in ``sdata`` to this reference.

        Parameters
        ----------
        sdata
            SpatialData object containing the source image.
        image_key
            Key of the RGB image in ``sdata.images`` to normalize.
        scale
            Scale level to normalize. ``"auto"`` (default) uses the finest level
            so the result is not downsampled; source statistics are reduced
            lazily so memory stays bounded.
        method_params
            Params matching this reference's ``method`` (instance, mapping, or ``None``).
        image_key_added
            Key for the written image when ``inplace=True``. If ``None`` (default),
            ``f"{image_key}_normalized"`` is used. Ignored when ``inplace=False``.
        inplace
            If ``True`` (default), write the normalized image to
            ``sdata.images[image_key_added]`` (rebuilding the pyramid for multiscale
            sources, preserving transforms) and return ``None``; raises if the key
            already exists. If ``False``, leave ``sdata`` untouched and return the
            lazy normalized :class:`~xarray.DataArray`.
        output_dtype
            Dtype of the result. If ``None`` (default), the source image's dtype is
            used. The reconstruction is clipped to that dtype's valid range and
            rounded (for integer dtypes) at the write boundary.
        tissue_mask_key
            Key of a tissue-label element in ``sdata.labels`` restricting the
            *source* statistics to tissue pixels. As for
            :func:`fit_stain_reference`, a tissue mask is required (defaults to
            ``f"{image_key}_tissue"``; raises if missing).
        preserve_background
            If ``True`` (default), non-tissue (background) pixels are passed through
            unchanged from the source image, so the normalization recolours only
            tissue. The colour map is a global linear transform that would otherwise
            tint background/white pixels. Set ``False`` for full-frame normalization.

        Returns
        -------
        ``None`` if ``inplace=True`` (the image is written), otherwise the lazy
        normalized :class:`xarray.DataArray`.
        """
        from squidpy.experimental.im._stain._normalize import _normalize_stains

        return _normalize_stains(
            sdata,
            image_key,
            self,
            scale=scale,
            method_params=method_params,
            image_key_added=image_key_added,
            inplace=inplace,
            output_dtype=output_dtype,
            tissue_mask_key=tissue_mask_key,
            preserve_background=preserve_background,
        )

    def decompose(
        self,
        sdata: sd.SpatialData,
        image_key: str,
        *,
        scale: str | Literal["auto"] = "auto",
        image_key_added: str | None = None,
        inplace: bool = True,
        output_dtype: DTypeLike = np.float16,
        include_residual: bool = True,
    ) -> dict[str, xr.DataArray] | None:
        """Decompose an image in ``sdata`` into separate per-stain concentration maps.

        Requires a decomposition reference (``method="macenko"`` or ``"vahadane"``):
        its stain matrix and white point are projected onto the image as-is, so this
        reference is the provenance record of how the maps were produced.

        Parameters
        ----------
        sdata, image_key
            The SpatialData object and the RGB image key to decompose.
        scale
            Scale level to decompose. ``"auto"`` (default) uses the finest level.
        image_key_added
            Key *prefix* for the written images when ``inplace=True``. If ``None``
            (default), ``image_key`` is used, so each stain is written as its own
            single-channel image ``sdata.images[f"{image_key}_{stain}"]`` (e.g.
            ``f"{image_key}_hematoxylin"``). Ignored when ``inplace=False``.
        inplace
            If ``True`` (default), write each stain as a separate single-channel
            image under the ``image_key_added`` prefix and return ``None``; the
            write is atomic (all target keys are validated free before any is
            written). If ``False``, leave ``sdata`` untouched and return the maps
            as a dict.
        output_dtype
            Dtype of the concentration maps. Defaults to ``float16`` (half the
            storage; ~3 significant figures, adequate for concentrations); pass
            ``float32`` for strict quantification.
        include_residual
            If ``True`` (default), also produce the ``"residual"`` map. The residual
            is the absorbance along the complement direction - a diagnostic of
            decomposition quality (extra chromogen, artifacts, or a poor fit), not a
            biological stain. Set ``False`` to keep only ``hematoxylin``/``eosin``.

        Returns
        -------
        ``None`` if ``inplace=True`` (the maps are written as separate images),
        otherwise a ``dict`` mapping each stain name to its ``(y, x)`` concentration
        :class:`~xarray.DataArray` (``"hematoxylin"``, ``"eosin"``, and
        ``"residual"`` unless dropped).
        """
        from squidpy.experimental.im._stain._normalize import _decompose_stains

        return _decompose_stains(
            sdata,
            image_key,
            self,
            scale=scale,
            image_key_added=image_key_added,
            inplace=inplace,
            output_dtype=output_dtype,
            include_residual=include_residual,
        )
