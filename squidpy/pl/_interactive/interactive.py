from typing import Any, Optional

from scanpy import logging as logg
from anndata import AnnData

import numpy as np

import matplotlib.pyplot as plt

from squidpy.im import ImageContainer  # type: ignore[attr-defined]
from squidpy._docs import d
from squidpy.pl._utils import save_fig

try:
    from squidpy.pl._interactive._controller import ImageController
except ImportError as e:
    _error: Optional[str] = str(e)
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

    def __init__(self, img: ImageContainer, adata: AnnData, **kwargs: Any):
        if _error is not None:
            raise ImportError(f"Unable to import the interactive viewer. Reason `{_error}`.")

        self._controller = ImageController(adata, img, **kwargs)

    def show(self, restore: bool = False) -> "Interactive":
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
        self, return_result: bool = False, dpi: Optional[float] = 180, save: Optional[str] = None, **kwargs: Any
    ) -> Optional[np.ndarray]:
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
        kwargs
            Keyword arguments for :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        Nothing, if ``return_result = False``, otherwise the image array.
        """
        try:
            arr = self._controller.screenshot(path=None)
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
