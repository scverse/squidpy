import os
from typing import Union
from pathlib import Path

from scanpy import logging as logg
from scanpy import settings

from matplotlib.figure import Figure


def save_fig(fig: Figure, path: Union[str, os.PathLike], make_dir: bool = True, ext: str = "png", **kwargs) -> None:
    """
    Save a figure.

    Parameters
    ----------
    fig
        Figure to save.
    path
        Path where to save the figure. If path is relative, save it under :attr:`scanpy.settings.figdir`.
    make_dir
        Whether to try making the directory if it does not exist.
    ext
        Extension to use if none is provided.
    kwargs
        Keyword arguments for :meth:`matplotlib.figure.Figure.savefig`.

    Returns
    -------
    None
        Just saves the plot.
    """
    path = Path(path)

    if os.path.splitext(path)[1] == "":
        path = f"{path}.{ext}"

    if not path.is_absolute():
        path = Path(settings.figdir) / path

    if make_dir:
        try:
            os.makedirs(Path.parent, exist_ok=True)
        except OSError as e:
            logg.debug(f"Unable to create directory `{Path.parent}`. Reason: `{e}`.")

    logg.debug(f"Saving figure to `{path!r}`")

    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("transparent", True)

    fig.savefig(path, **kwargs)
