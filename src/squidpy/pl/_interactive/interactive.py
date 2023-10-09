from __future__ import annotations

from typing import (
    Any,
    Union,  # noqa: F401
)

import matplotlib.pyplot as plt
from anndata import AnnData
from scanpy import logging as logg

from squidpy._docs import d
from squidpy._utils import NDArrayA, deprecated
from squidpy.im import ImageContainer  # type: ignore[attr-defined]
from squidpy.pl._utils import save_fig

try:
    from squidpy.pl._interactive._controller import ImageController
except ImportError as e:
    _error: str | None = str(e)
else:
    _error = None


__all__ = ["Interactive"]


@d.dedent
class Interactive:
    """
    Interactive viewer for spatial data.

    Parameters
    ----------
    %(img_container)s
    %(_interactive.parameters)s
    """

    @deprecated(
        reason="The squidpy napari plugin is deprecated, please use https://github.com/scverse/napari-spatialdata",
    )
    def __init__(self, img: ImageContainer, adata: AnnData, **kwargs: Any):
        if _error is not None:
            raise ImportError(f"Unable to import the interactive viewer. Reason `{_error}`.")

        self._controller = ImageController(adata, img, **kwargs)

    def show(self, restore: bool = False) -> Interactive:
        """
        Launch the :class:`napari.Viewer`.

        Parameters
        ----------
        restore
            Whether to reinitialize the GUI after it has been destroyed.

        Returns
        -------
        Nothing, just launches the viewer.
        """
        self._controller.show(restore=restore)
        return self

    @d.dedent
    def screenshot(
        self,
        return_result: bool = False,
        dpi: float | None = 180,
        save: str | None = None,
        canvas_only: bool = True,
        **kwargs: Any,
    ) -> NDArrayA | None:
        """
        Plot a screenshot of the viewer's canvas.

        Parameters
        ----------
        return_result
            If `True`, return the image as an :class:`numpy.uint8`.
        dpi
            Dots per inch.
        save
            Whether to save the plot.
        canvas_only
            Whether to show only the canvas or also the widgets.
        kwargs
            Keyword arguments for :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        Nothing, if ``return_result = False``, otherwise the image array.
        """
        try:
            arr = self._controller.screenshot(path=None, canvas_only=canvas_only)
        except RuntimeError as e:
            logg.error(f"Unable to take a screenshot. Reason: {e}")
            return None

        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=dpi)
        fig.tight_layout()

        ax.imshow(arr, **kwargs)
        plt.axis("off")

        if save is not None:
            save_fig(fig, save)

        return arr if return_result else None

    def close(self) -> None:
        """Close the viewer."""
        self._controller.close()

    @property
    def adata(self) -> AnnData:
        """Annotated data object."""
        return self._controller.model.adata

    def __repr__(self) -> str:
        return f"Interactive view of {repr(self._controller.model.container)}"

    def __str__(self) -> str:
        return repr(self)
