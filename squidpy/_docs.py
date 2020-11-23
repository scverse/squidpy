from textwrap import dedent

from docrep import DocstringProcessor

d = DocstringProcessor()


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
