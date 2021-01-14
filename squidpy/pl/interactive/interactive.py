from typing import Any, Optional

from scanpy import logging as logg
from anndata import AnnData

import numpy as np

import matplotlib.pyplot as plt

from squidpy.im import ImageContainer  # type: ignore[attr-defined]
from squidpy._docs import d
from squidpy.pl._utils import save_fig
from squidpy.constants._pkg_constants import Key
from squidpy.pl.interactive._controller import ImageController


@d.dedent
class Interactive:
    """
    Interactive viewer for spatial data.

    Parameters
    ----------
    %(interactive.parameters)s
    """

    def __init__(
        self,
        adata: AnnData,
        img: ImageContainer,
        spatial_key: str = Key.obsm.spatial,
        cont_cmap: str = "viridis",
        cat_cmap: Optional[str] = None,
        blending: str = "opaque",
        key_added: str = "shapes",
    ):
        self._controller = ImageController(
            adata,
            img,
            spatial_key=spatial_key,
            cat_cmap=cat_cmap,
            cont_cmap=cont_cmap,
            blending=blending,
            key_added=key_added,
        )

    @d.dedent
    def show(self, restore: bool = False) -> "Interactive":
        """
        %(cont_show.full_desc)s

        Parameters
        ----------
        %(cont_show.parameters)s

        Returns
        -------
        %(cont_show.returns)s
        """  # noqa: D400
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
            Keyword arguments for :func:`matplotlib.pyplot.imshow`.

        Returns
        -------
        None if ``return_result = False``, otherwise the image array.
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
        """Close the :class:`napari.Viewer`."""
        self._controller.close()

    def __repr__(self) -> str:
        return f"Interactive view of {repr(self._controller.model.container)}"

    def __str__(self) -> str:
        return repr(self)
