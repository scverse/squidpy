from __future__ import annotations

import itertools
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
from asv_runner.benchmarks.mark import skip_for_params
from scipy.sparse import csc_matrix, csr_matrix

import squidpy as sq

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from collections.abc import Set as AbstractSet
    from typing import Literal, Protocol, TypeVar

    from anndata import AnnData

    C = TypeVar("C", bound=Callable)

    class ParamSkipper(Protocol):
        def __call__(self, **skipped: AbstractSet) -> Callable[[C], C]: ...

    Dataset = Literal["imc"]
    KeyX = Literal[None, "off-axis"]


@cache
def _imc() -> AnnData:
    adata = sq.datasets.imc()
    assert isinstance(adata.X, np.ndarray)
    assert not np.isfortran(adata.X)

    return adata


def imc() -> AnnData:
    return _imc().copy()


def to_off_axis(x: np.ndarray | csr_matrix | csc_matrix) -> np.ndarray | csc_matrix:
    if isinstance(x, csr_matrix):
        return x.tocsc()
    if isinstance(x, np.ndarray):
        assert not np.isfortran(x)
        return x.copy(order="F")
    msg = f"Unexpected type {type(x)}"
    raise TypeError(msg)


def _get_dataset_raw(dataset: Dataset) -> tuple[AnnData, str | None]:
    match dataset:
        case "imc":
            adata, cluster_key = imc(), "cell type"
        case _:
            msg = f"Unknown dataset {dataset}"
            raise AssertionError(msg)

    adata.layers["off-axis"] = to_off_axis(adata.X)

    return adata, cluster_key


def get_dataset(dataset: Dataset, *, layer: KeyX = None) -> tuple[AnnData, str | None]:
    adata, batch_key = _get_dataset_raw(dataset)
    if layer is not None:
        adata.X = adata.layers.pop(layer)
    return adata, batch_key


def param_skipper(param_names: Sequence[str], params: tuple[Sequence[object], ...]) -> ParamSkipper:
    """Create a decorator that will skip all combinations that contain any of the given parameters.

    Examples
    --------
    >>> param_names = ["letters", "numbers"]
    >>> params = [["a", "b"], [3, 4, 5]]
    >>> skip_when = param_skipper(param_names, params)

    >>> @skip_when(letters={"a"}, numbers={3})
    ... def func(a, b):
    ...     print(a, b)
    >>> run_as_asv_benchmark(func)
    b 4
    b 5

    """

    def skip(**skipped: AbstractSet) -> Callable[[C], C]:
        skipped_combs = [
            tuple(record.values())
            for record in (dict(zip(param_names, vals, strict=True)) for vals in itertools.product(*params))
            if any(v in skipped.get(n, set()) for n, v in record.items())
        ]
        # print(skipped_combs, file=sys.stderr)
        return skip_for_params(skipped_combs)

    return skip
