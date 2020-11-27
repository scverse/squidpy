from textwrap import dedent

from docrep import DocstringProcessor


def inject_docs(**kwargs):  # noqa: D103
    # taken from scanpy
    def decorator(obj):
        obj.__doc__ = dedent(obj.__doc__).format(**kwargs)
        return obj

    def decorator2(obj):
        obj.__doc__ = dedent(kwargs["__doc__"])
        return obj

    if isinstance(kwargs.get("__doc__", None), str) and len(kwargs) == 1:
        return decorator2

    return decorator


_adata = """\
adata
    Annotated data object."""
_img_container = """\
img
    High-resolution image."""
_copy = """\
copy
    If `True`, return the result, otherwise save it to the ``adata`` object."""
_numba_parallel = """\
numba_parallel
    Whether to use :class:`numba.prange` or not. If `None`, it's determined automatically.
    For small datasets or small number of interactions, it's recommended to set this to `False`.
"""
_seed = """\
seed
    Random seed for reproducibility.
"""


d = DocstringProcessor(
    adata=_adata, img_container=_img_container, copy=_copy, numba_parallel=_numba_parallel, seed=_seed
)
