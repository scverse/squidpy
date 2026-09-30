"""The fitted stain reference and the operations that apply it.

Holds either a 3x3 stain matrix (Macenko/Vahadane) or a pair of Ruderman Lab channel
statistics (Reinhard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from squidpy._params import resolve_params
from squidpy._validators import normalize_choice
from squidpy.experimental.im._stain._conversion import cast_to_image_dtype
from squidpy.experimental.im._stain._white_point import validate_rgb_range

if TYPE_CHECKING:
    import spatialdata as sd
    import xarray as xr
    from numpy.typing import DTypeLike

    from squidpy.experimental.im._stain._normalize import MethodParams

StainMethod = Literal["macenko", "vahadane", "reinhard"]
_DECOMPOSITION_METHODS: frozenset[str] = frozenset({"macenko", "vahadane"})
_VALID_METHODS: frozenset[str] = _DECOMPOSITION_METHODS | {"reinhard"}
_CONCENTRATION_CHANNELS = ["hematoxylin", "eosin", "residual"]


def _coerce_finite(arr: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {out.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values.")
    return out


@dataclass(frozen=True)
class StainFit:
    """A fitted stain reference.

    Returned by :func:`~squidpy.experimental.im.fit_stain_reference`. Use :meth:`transform`
    to normalize an image to it, or :meth:`decompose` to project one onto its stain matrix.

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
        object.__setattr__(self, "method", normalize_choice(self.method, sorted(_VALID_METHODS), name="method"))

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
        *,
        image_key: str,
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
            Params matching this fit's ``method`` (mapping or ``None``).
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
        from squidpy.experimental.im._stain._decomposition import apply_decomposition
        from squidpy.experimental.im._stain._normalize import (
            _METHOD_PARAMS,
            _resolve_image,
            _resolve_output_tissue_mask,
            _resolve_tissue_bool_mask,
            _write_image,
        )
        from squidpy.experimental.im._stain._reinhard import apply_reinhard

        da = _resolve_image(sdata, image_key=image_key, scale=scale, prefer="finest")
        target_key = image_key_added if image_key_added is not None else f"{image_key}_normalized"
        if inplace and target_key in sdata.images:
            raise ValueError(f"image_key_added={target_key!r} already exists in sdata.images.")
        params = resolve_params(method_params, _METHOD_PARAMS[self.method])
        # Source statistics (Reinhard mu/sigma or the decomposition source matrix)
        # are reduced on a coarse level with a tissue mask; the lazy transform is
        # then applied to the full-resolution `da`.
        fit_rgb = _resolve_image(sdata, image_key=image_key, scale=scale, prefer="coarsest")
        # reject mis-typed source (e.g. 0-255 float) before the dtype-clipped reconstruction
        validate_rgb_range(fit_rgb)
        tissue_mask = _resolve_tissue_bool_mask(
            sdata, image_key=image_key, fit_da=fit_rgb, tissue_mask_key=tissue_mask_key
        )
        out_dtype = da.dtype if output_dtype is None else np.dtype(output_dtype)  # clip range + final cast
        if self.method == "reinhard":
            normalized = apply_reinhard(da, self, params, fit_rgb=fit_rgb, tissue_mask=tissue_mask, out_dtype=out_dtype)
        else:
            normalized = apply_decomposition(
                da, self, params, fit_rgb=fit_rgb, tissue_mask=tissue_mask, out_dtype=out_dtype
            )

        if preserve_background:
            # Keep non-tissue pixels byte-identical to the source: the global colour
            # map would otherwise recolour background/white pixels (HistomicsTK's
            # `mask_out`). Stays lazy - the mask aligns to `da` without materialising.
            keep = _resolve_output_tissue_mask(
                sdata, image_key=image_key, target_da=da, tissue_mask_key=tissue_mask_key
            )
            normalized = normalized.where(keep, da)

        # Deferred cast at the write boundary: the reconstruction was kept in float
        # (clipped to `out_dtype`'s range); round + cast here so the stored image is
        # the requested dtype and integer background stays byte-identical.
        normalized = cast_to_image_dtype(normalized, out_dtype)

        # The output is a 3-channel RGB image; tag it r/g/b so RGB-aware viewers
        # (spatialdata-plot) use one hue-preserving scale, not per-channel auto-contrast.
        normalized = normalized.assign_coords(c=["r", "g", "b"])

        if not inplace:
            return normalized
        _write_image(sdata, source_node=sdata.images[image_key], image_key_added=target_key, data_array=normalized)
        return None

    def decompose(
        self,
        sdata: sd.SpatialData,
        *,
        image_key: str,
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
        from squidpy.experimental.im._stain._decomposition import decompose_to_concentrations
        from squidpy.experimental.im._stain._normalize import _resolve_image, _write_image

        da = _resolve_image(sdata, image_key=image_key, scale=scale, prefer="finest")
        if self.method not in _DECOMPOSITION_METHODS or self.stain_matrix is None:
            raise ValueError("decompose requires a macenko/vahadane reference with a stain matrix.")
        stain_matrix, bg = self.stain_matrix, self.white_point

        names = ["hematoxylin", "eosin"] + (["residual"] if include_residual else [])
        prefix = image_key_added if image_key_added is not None else image_key
        target_keys = [f"{prefix}_{name}" for name in names]
        if inplace:  # validate all keys free up front, so a partial write can't leave a half-decomposed sdata
            clashes = [k for k in target_keys if k in sdata.images]
            if clashes:
                raise ValueError(f"decompose would overwrite existing image(s): {clashes}.")

        concentrations = decompose_to_concentrations(da, stain_matrix, bg).assign_coords(c=_CONCENTRATION_CHANNELS)
        concentrations = concentrations.astype(np.dtype(output_dtype))

        if not inplace:
            return {name: concentrations.sel(c=name) for name in names}

        source = sdata.images[image_key]
        for name, key in zip(names, target_keys, strict=True):
            # keep the c dim (length 1) so Image2DModel.parse accepts it
            _write_image(
                sdata, source_node=source, image_key_added=key, data_array=concentrations.sel(c=[name]), c_coords=[name]
            )
        return None
