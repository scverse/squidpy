"""GPU adapter functions for squidpy.gr functions.

These stubs provide explicit parameter mapping between squidpy and rapids_singlecell,
ensuring compatibility and clear documentation of supported parameters.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from squidpy._constants._pkg_constants import Key

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray


@dataclass
class GpuParamSpec:
    """Specification for a parameter's GPU compatibility."""

    default: Any
    message: str | None = None
    validator: Callable[[Any], str | None] | None = None


def _attr_validator(value: Any) -> str | None:
    """Special validator for attr param - only warn if not 'X'."""
    if value == "X":
        return None
    return f"attr={value!r} is not supported on GPU, using attr='X'. Set device='cpu' to use other attributes."


# Common CPU-only param specs (reusable)
_PARALLELIZE: dict[str, GpuParamSpec] = {
    "n_jobs": GpuParamSpec(None),
    "backend": GpuParamSpec("loky"),
    "show_progress_bar": GpuParamSpec(True),
}
_SEED: dict[str, GpuParamSpec] = {"seed": GpuParamSpec(None)}

# Registry: {func_name: {"cpu_only": {...}, "gpu_only": {...}}}
GPU_PARAM_REGISTRY: dict[str, dict[str, dict[str, GpuParamSpec]]] = {
    "spatial_autocorr": {
        "cpu_only": {
            "attr": GpuParamSpec("X", validator=_attr_validator),
            **_SEED,
            **_PARALLELIZE,
        },
        "gpu_only": {
            "use_sparse": GpuParamSpec(True),
        },
    },
    "co_occurrence": {
        "cpu_only": {
            "n_splits": GpuParamSpec(None),
            **_PARALLELIZE,
        },
        "gpu_only": {},
    },
    "ligrec": {
        "cpu_only": {
            "clusters": GpuParamSpec(None),
            "numba_parallel": GpuParamSpec(None),
            "transmitter_params": GpuParamSpec(None),
            "receiver_params": GpuParamSpec(None),
            "interactions_params": GpuParamSpec(None),
            "alpha": GpuParamSpec(0.05),
            **_SEED,
            **_PARALLELIZE,
        },
        "gpu_only": {},
    },
}


@dataclass
class CheckResult:
    """Result of parameter compatibility check."""

    ignored: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    gpu_defaults: dict[str, Any] = field(default_factory=dict)


def check_gpu_params(func_name: str, **cpu_only_values: Any) -> CheckResult:
    """Check CPU-only params against registry, warn about non-defaults, return GPU defaults.

    Parameters
    ----------
    func_name
        Name of the function in GPU_PARAM_REGISTRY.
    **cpu_only_values
        CPU-only parameter values to check.
    """
    result = CheckResult()
    registry = GPU_PARAM_REGISTRY.get(func_name, {"cpu_only": {}, "gpu_only": {}})

    # Check CPU-only params
    for name, spec in registry["cpu_only"].items():
        if name not in cpu_only_values:
            continue
        value = cpu_only_values[name]

        # Use custom validator if provided, else default behavior
        if spec.validator:
            msg = spec.validator(value)
        elif value != spec.default:
            msg = spec.message or f"{name}={value!r} is ignored on GPU."
        else:
            msg = None

        if msg:
            msg = msg.format(name=name, value=value)
            result.ignored[name] = value
            result.warnings.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)

    # Collect GPU-only param defaults
    for name, spec in registry["gpu_only"].items():
        result.gpu_defaults[name] = spec.default

    return result


def spatial_autocorr_gpu(
    adata: AnnData,
    connectivity_key: str = Key.obsp.spatial_conn(),
    genes: str | int | Sequence[str] | Sequence[int] | None = None,
    mode: Literal["moran", "geary"] = "moran",
    transformation: bool = True,
    n_perms: int | None = None,
    two_tailed: bool = False,
    corr_method: str | None = "fdr_bh",
    layer: str | None = None,
    use_raw: bool = False,
    copy: bool = False,
    # CPU-only params
    attr: Literal["obs", "X", "obsm"] = "X",
    seed: int | None = None,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
) -> Any:
    """GPU adapter for spatial_autocorr via rapids_singlecell."""
    from rapids_singlecell.squidpy_gpu import spatial_autocorr as _spatial_autocorr_gpu

    check = check_gpu_params(
        "spatial_autocorr",
        attr=attr,
        seed=seed,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )

    return _spatial_autocorr_gpu(
        adata=adata,
        connectivity_key=connectivity_key,
        genes=genes,
        mode=mode,
        transformation=transformation,
        n_perms=n_perms,
        two_tailed=two_tailed,
        corr_method=corr_method,
        layer=layer,
        use_raw=use_raw,
        copy=copy,
        **check.gpu_defaults,
    )


def co_occurrence_gpu(
    adata: AnnData,
    cluster_key: str,
    spatial_key: str = Key.obsm.spatial,
    interval: int | NDArray[Any] = 50,
    copy: bool = False,
    # CPU-only params
    n_splits: int | None = None,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
) -> Any:
    """GPU adapter for co_occurrence via rapids_singlecell."""
    from rapids_singlecell.squidpy_gpu import co_occurrence as _co_occurrence_gpu

    check_gpu_params(
        "co_occurrence",
        n_splits=n_splits,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )

    return _co_occurrence_gpu(
        adata=adata,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
        interval=interval,
        copy=copy,
    )


def ligrec_gpu(
    adata: AnnData,
    cluster_key: str,
    interactions: Any = None,
    complex_policy: Literal["min", "all"] = "min",
    threshold: float = 0.01,
    corr_method: str | None = None,
    corr_axis: Literal["interactions", "clusters"] = "clusters",
    use_raw: bool = True,
    copy: bool = False,
    key_added: str | None = None,
    gene_symbols: str | None = None,
    n_perms: int = 1000,
    # CPU-only params
    clusters: Any = None,
    seed: int | None = None,
    numba_parallel: bool | None = None,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
    transmitter_params: dict[str, Any] | None = None,
    receiver_params: dict[str, Any] | None = None,
    interactions_params: dict[str, Any] | None = None,
    alpha: float = 0.05,
) -> Any:
    """GPU adapter for ligrec via rapids_singlecell."""
    from rapids_singlecell.squidpy_gpu import ligrec as _ligrec_gpu

    check_gpu_params(
        "ligrec",
        clusters=clusters,
        seed=seed,
        numba_parallel=numba_parallel,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
        transmitter_params=transmitter_params,
        receiver_params=receiver_params,
        interactions_params=interactions_params,
        alpha=alpha,
    )

    return _ligrec_gpu(
        adata=adata,
        cluster_key=cluster_key,
        interactions=interactions,
        complex_policy=complex_policy,
        threshold=threshold,
        corr_method=corr_method,
        corr_axis=corr_axis,
        use_raw=use_raw,
        copy=copy,
        key_added=key_added,
        gene_symbols=gene_symbols,
        n_perms=n_perms,
    )
