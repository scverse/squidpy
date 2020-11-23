"""Graph utilities."""
from abc import ABC, ABCMeta
from enum import Enum, EnumMeta
from typing import Any, Tuple, Union, TypeVar, Callable, Optional, Sequence
from functools import wraps
from threading import Thread
from multiprocessing import Manager, cpu_count

import joblib as jl

import numpy as np
from scipy.sparse import issparse, spmatrix, csc_matrix


class Signal(Enum):
    """Signalling values when informing parallelizer."""

    NONE = 0
    UPDATE = 1
    FINISH = 2
    UPDATE_FINISH = 3


Queue = TypeVar("Queue")


def parallelize(
    callback: Callable[[Any], Any],
    collection: Sequence[Any],
    n_jobs: int = 1,
    n_split: Optional[int] = None,
    unit: str = "",
    use_ixs: bool = False,
    backend: str = "loky",
    extractor: Optional[Callable[[Any], Any]] = None,
    show_progress_bar: bool = True,
    use_runner: bool = False,
) -> Any:
    """
    Parallelize function call over a collection of elements.

    Parameters
    ----------
    callback
        Function to parallelize. Can either accept a whole chunk (``use_runner=False``) or just a single
        element (``use_runner=True``).
    collection
        Sequence of items which to chunkify.
    n_jobs
        Number of parallel jobs.
    n_split
        Split ``collection`` into ``n_split`` chunks.
        If <= 0, ``collection`` is assumed to be already chunkified.
    unit
        Unit of the progress bar.
    use_ixs
        Whether to pass indices to the callback.
    backend
        Which backend to use for multiprocessing. See :class:`joblib.Parallel` for valid options.
    extractor
        Function to apply to the result after all jobs have finished.
    show_progress_bar
        Whether to show a progress bar.
    use_runner
        Whether the ``callback`` handles only 1 item from the ``collection`` or a chunk.
        The latter grants more control, e.g. using :func:`numba.prange` instead of normal iteration.

    Returns
    -------
        The result depending on ``callable``, ``extractor``.
    """
    if show_progress_bar:
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm import tqdm_notebook as tqdm
    else:
        tqdm = None

    def runner(iterable, *args, queue: Optional[Queue] = None, **kwargs):
        result = []

        for it in iterable:
            res = callback(it, *args, **kwargs)
            if res is not None:
                result.append(result)
            if queue is not None:
                queue.put(Signal.UPDATE)

        if queue is not None:
            queue.put(Signal.FINISH)

        return result

    def update(pbar, queue, n_total):
        n_finished = 0
        while n_finished < n_total:
            try:
                res = queue.get()
            except EOFError as e:
                if not n_finished != n_total:
                    raise RuntimeError(f"Finished only `{n_finished}` out of `{n_total}` tasks.") from e
                break

            assert isinstance(res, Signal)

            if res in (Signal.FINISH, Signal.UPDATE_FINISH):
                n_finished += 1
            if pbar is not None and res in (Signal.UPDATE, Signal.UPDATE_FINISH):
                pbar.update()

        if pbar is not None:
            pbar.close()

    def wrapper(*args, **kwargs):
        if pass_queue and show_progress_bar:
            pbar = None if tqdm is None else tqdm(total=col_len, unit=unit)
            queue = Manager().Queue()
            thread = Thread(target=update, args=(pbar, queue, len(collections)))
            thread.start()
        else:
            pbar, queue, thread = None, None, None

        res = jl.Parallel(n_jobs=n_jobs, backend=backend)(
            jl.delayed(runner if use_runner else callback)(
                *((i, cs) if use_ixs else (cs,)),
                *args,
                **kwargs,
                queue=queue,
            )
            for i, cs in enumerate(collections)
        )

        if thread is not None:
            thread.join()

        return res if extractor is None else extractor(res)

    if n_jobs == 0:
        raise ValueError("Number of jobs cannot be `0`.")
    if n_jobs < 0:
        n_jobs = cpu_count() + 1 + n_jobs

    if n_split is None:
        n_split = n_jobs

    if n_split <= 0:
        col_len = sum(map(len, collection))
        collections = collection
    else:
        col_len = len(collection)
        collections = list(filter(len, np.array_split(collection, n_split)))

    if use_runner:
        use_ixs = False
    pass_queue = not hasattr(callback, "py_func")  # we'd be inside a numba function

    return wrapper


def _get_n_cores(n_cores: Optional[int]) -> int:
    """
    Make number of cores a positive integer.

    This is useful for especially logging.

    Parameters
    ----------
    n_cores
        Number of cores to use.

    Returns
    -------
    int
        Positive integer corresponding to how many cores to use.
    """
    if n_cores == 0:
        raise ValueError("Number of cores cannot be `0`.")
    if n_cores is None:
        return 1
    if n_cores < 0:
        return cpu_count() + 1 + n_cores

    return n_cores


def _pretty_raise_enum(cls, fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except ValueError as e:
            _cls, value, *_ = args
            e.args = (cls._format(value),)
            raise e

    if not issubclass(cls, ErrorFormatterABC):
        raise TypeError(f"Class `{cls}` must be subtype of `ErrorFormatterABC`.")
    elif not len(cls.__members__):
        # empty enum, for class hierarchy
        return fun

    return wrapper


class ABCEnumMeta(EnumMeta, ABCMeta):  # noqa: D101
    def __call__(cls, *args, **kw):  # noqa: D102
        if getattr(cls, "__error_format__", None) is None:
            raise TypeError(f"Can't instantiate class `{cls.__name__}` " f"without `__error_format__` class attribute.")
        return super().__call__(*args, **kw)

    def __new__(cls, clsname, superclasses, attributedict):  # noqa: D102
        res = super().__new__(cls, clsname, superclasses, attributedict)
        res.__new__ = _pretty_raise_enum(res, res.__new__)
        return res


class ErrorFormatterABC(ABC):  # noqa: D101
    __error_format__ = "Invalid option `{!r}` for `{}`. Valid options are: `{}`."

    @classmethod
    def _format(cls, value):
        return cls.__error_format__.format(value, cls.__name__, [m.value for m in cls.__members__.values()])


class PrettyEnum(Enum):
    """Enum with a pretty __str__ and __repr__."""

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.value)


class ModeEnum(ErrorFormatterABC, PrettyEnum, metaclass=ABCEnumMeta):  # noqa: D101
    pass


def _check_tuple_needles(
    needles: Sequence[Tuple[Any, Any]],
    haystack: Sequence[Any],
    msg: str,
    reraise: bool = True,
) -> Sequence[Tuple[Any, Any]]:
    filtered = []

    for needle in needles:
        if isinstance(needle, str) or not isinstance(needle, Sequence):
            raise TypeError(f"Expected a `Sequence`, found `{type(needle).__name__}`")
        if len(needle) != 2:
            raise ValueError(f"Expected a `tuple` of length 2, found `{len(needle)}`")
        a, b = needle

        if a not in haystack:
            if reraise:
                raise ValueError(msg.format(a))
            else:
                continue
        if b not in haystack:
            if reraise:
                raise ValueError(msg.format(a))
            else:
                continue

        filtered.append((a, b))

    return filtered


def _create_sparse_df(data: Union[np.ndarray, spmatrix], index=None, columns=None, fill_value: float = 0):
    """
    Create a new DataFrame from a scipy sparse matrix or numpy array.

    This is the original :mod:`pandas` implementation with 2 differences:

        - allow creation also from :class:`numpy.ndarray`
        - expose ``fill_values``

    .. versionadded:: 0.25.0

    Parameters
    ----------
    data : scipy.sparse.spmatrix or numpy.ndarray
        Must be convertible to csc format.
    index, columns : Index, optional
        Row and column labels to use for the resulting DataFrame.
        Defaults to a RangeIndex.

    Returns
    -------
    DataFrame
        Each column of the DataFrame is stored as a
        :class:`arrays.SparseArray`.

    Examples
    --------
    >>> import scipy.sparse
    >>> mat = scipy.sparse.eye(3)
    >>> pd.DataFrame.sparse.from_spmatrix(mat)
         0    1    2
    0  1.0  0.0  0.0
    1  0.0  1.0  0.0
    2  0.0  0.0  1.0
    """
    from pandas import DataFrame
    from pandas._libs.sparse import IntIndex
    from pandas.core.arrays.sparse.accessor import (
        SparseArray,
        SparseDtype,
        SparseFrameAccessor,
    )

    if not issparse(data):
        data = csc_matrix(data)
        sort_indices = False
    else:
        data = data.tocsc()
        sort_indices = True

    data = data.tocsc()
    index, columns = SparseFrameAccessor._prep_index(data, index, columns)
    n_rows, n_columns = data.shape
    # We need to make sure indices are sorted, as we create
    # IntIndex with no input validation (i.e. check_integrity=False ).
    # Indices may already be sorted in scipy in which case this adds
    # a small overhead.
    if sort_indices:
        data.sort_indices()

    indices = data.indices
    indptr = data.indptr
    array_data = data.data
    dtype = SparseDtype(array_data.dtype, fill_value=fill_value)
    arrays = []

    for i in range(n_columns):
        sl = slice(indptr[i], indptr[i + 1])
        idx = IntIndex(n_rows, indices[sl], check_integrity=False)
        arr = SparseArray._simple_new(array_data[sl], idx, dtype)
        arrays.append(arr)

    return DataFrame._from_arrays(arrays, columns=columns, index=index, verify_integrity=False)
