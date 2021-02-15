"""Spatial tools general utility functions."""
from __future__ import annotations

import joblib as jl

from enum import Enum
from queue import Queue
from typing import (
    Any,
    Set,
    List,
    Tuple,
    Callable,
    Hashable,
    Iterable,
    Optional,
    Sequence,
    Generator,
    TYPE_CHECKING,
)
from threading import Thread
from contextlib import contextmanager
from multiprocessing import Manager, cpu_count

import numpy as np

__all__ = ["singledispatchmethod", "Signal", "SigQueue"]


try:
    from functools import singledispatchmethod
except ImportError:
    from functools import singledispatch, update_wrapper

    def singledispatchmethod(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[no-redef]
        """Backport of `singledispatchmethod` for < Python 3.8."""
        dispatcher = singledispatch(func)

        def wrapper(*args: Any, **kw: Any) -> Any:
            return dispatcher.dispatch(args[1].__class__)(*args, **kw)

        wrapper.register = dispatcher.register  # type: ignore[attr-defined]
        update_wrapper(wrapper, func)

        return wrapper


class SigQueue(Queue["Signal"] if TYPE_CHECKING else Queue):  # type: ignore[misc]
    """Signalling queue."""


def _unique_order_preserving(iterable: Iterable[Hashable]) -> Tuple[List[Hashable], Set[Hashable]]:
    """Remove items from an iterable while preserving the order."""
    seen: Set[Hashable] = set()
    seen_add = seen.add
    return [i for i in iterable if not (i in seen or seen_add(i))], seen


class Signal(Enum):
    """Signalling values when informing parallelizer."""

    NONE = 0
    UPDATE = 1
    FINISH = 2
    UPDATE_FINISH = 3


def parallelize(
    callback: Callable[..., Any],
    collection: Sequence[Any],
    n_jobs: int = 1,
    n_split: Optional[int] = None,
    unit: str = "",
    use_ixs: bool = False,
    backend: str = "loky",
    extractor: Optional[Callable[[Sequence[Any]], Any]] = None,
    show_progress_bar: bool = True,
    use_runner: bool = False,
    **_: Any,
) -> Any:
    """
    Parallelize function call over a collection of elements.

    Parameters
    ----------
    callback
        Function to parallelize. Can either accept a whole chunk (``use_runner=False``) or just a single
        element (``use_runner=True``).
    collection
        Sequence of items to split into chunks.
    n_jobs
        Number of parallel jobs.
    n_split
        Split ``collection`` into ``n_split`` chunks.
        If <= 0, ``collection`` is assumed to be already split into chunks.
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

    def runner(iterable: Iterable[Any], *args: Any, queue: Optional["SigQueue"] = None, **kwargs: Any) -> List[Any]:
        result: List[Any] = []

        for it in iterable:
            res = callback(it, *args, **kwargs)
            if res is not None:
                result.append(result)
            if queue is not None:
                queue.put(Signal.UPDATE)

        if queue is not None:
            queue.put(Signal.FINISH)

        return result

    def update(pbar: "tqdm.std.tqdm", queue: "SigQueue", n_total: int) -> None:
        n_finished = 0
        while n_finished < n_total:
            try:
                res = queue.get()
            except EOFError as e:
                if not n_finished != n_total:
                    raise RuntimeError(f"Finished only `{n_finished}` out of `{n_total}` tasks.") from e
                break

            assert isinstance(res, Signal), f"Invalid type `{type(res).__name__}`."

            if res in (Signal.FINISH, Signal.UPDATE_FINISH):
                n_finished += 1
            if pbar is not None and res in (Signal.UPDATE, Signal.UPDATE_FINISH):
                pbar.update()

        if pbar is not None:
            pbar.close()

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if pass_queue and show_progress_bar:
            pbar = None if tqdm is None else tqdm(total=col_len, unit=unit)
            queue = Manager().Queue()
            thread = Thread(target=update, args=(pbar, queue, len(collections)))
            thread.start()
        else:
            pbar, queue, thread = None, None, None  # type: ignore[assignment]

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
        step = int(np.ceil(len(collection) / n_split))
        collections = list(filter(len, (collection[i * step : (i + 1) * step] for i in range(col_len))))

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


@contextmanager
def verbosity(level: int) -> Generator[None, None, None]:
    """
    Temporarily set the verbosity level of :mod:`scanpy`.

    Parameters
    ----------
    level
        The new verbosity leve.

    Returns
    -------
    Nothing.
    """
    import scanpy as sc

    verbosity = sc.settings.verbosity
    sc.settings.verbosity = level
    try:
        yield
    finally:
        sc.settings.verbosity = verbosity
