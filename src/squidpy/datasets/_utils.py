from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from inspect import Parameter, Signature, signature
from pathlib import Path
from typing import Any, TypeAlias, Union

import anndata
import spatialdata as sd
from anndata import AnnData
from scanpy import logging as logg
from scanpy import read
from scanpy._utils import check_presence_download

PathLike: TypeAlias = os.PathLike[str] | str
Function_t: TypeAlias = Callable[..., AnnData | Any]
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "squidpy"


@dataclass(frozen=True)
class Metadata(ABC):
    """Base class handling metadata."""

    name: str
    url: str

    doc_header: str | None = field(default=None, repr=False)
    path: PathLike | None = field(default=None, repr=False)
    shape: tuple[int, int] | None = field(default=None, repr=False)
    library_id: str | Sequence[str] | None = field(default=None, repr=False)

    _DOC_FMT = ""

    def __post_init__(self) -> None:
        if self.doc_header is None:
            object.__setattr__(self, "doc_header", f"Download `{self.name.title().replace('_', ' ')}` data.")
        if self.path is None:
            object.__setattr__(self, "path", os.path.expanduser(f"~/.cache/squidpy/{self.name}"))

    @property
    @abstractmethod
    def _extension(self) -> str:
        pass

    @abstractmethod
    def _download(self, fpath: PathLike, backup_url: str, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _create_signature(self) -> Signature:
        pass

    def _create_function(self, name: str, glob_ns: dict[str, Any]) -> None:
        if name in globals():
            raise KeyError(f"Function name `{name}` is already present in `{sorted(globals().keys())}`.")

        sig = self._create_signature()
        globals()["NoneType"] = type(None)  # __post_init__ return annotation
        globals()[name] = self

        exec(
            f"def {self.name}{sig}:\n"
            f'    """'
            f"    {self._DOC_FMT.format(doc_header=self.doc_header, shape=self.shape)}"
            f'    """\n'
            f"    return {name}.download(path, **kwargs)".replace(" /,", ""),
            globals(),
            glob_ns,
        )

    def download(self, fpath: PathLike | None = None, **kwargs: Any) -> Any:
        """Download the dataset into ``fpath``."""
        fpath = str(self.path if fpath is None else fpath)
        if not fpath.endswith(self._extension):
            fpath += self._extension

        if os.path.isfile(fpath):
            logg.debug(f"Loading dataset `{self.name}` from `{fpath}`")
        else:
            logg.debug(f"Downloading dataset `{self.name}` from `{self.url}` as `{fpath}`")

        dirname = Path(fpath).parent
        try:
            if not dirname.is_dir():
                logg.info(f"Creating directory `{dirname}`")
                dirname.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logg.error(f"Unable to create directory `{dirname}`. Reason `{e}`")

        data = self._download(fpath=fpath, backup_url=self.url, **kwargs)

        if self.shape is not None and data.shape != self.shape:
            raise ValueError(f"Expected the data to have shape `{self.shape}`, found `{data.shape}`.")

        return data


class AMetadata(Metadata):
    """Metadata class for :class:`anndata.AnnData`."""

    _DOC_FMT = """
    {doc_header}

    The shape of this :class:`anndata.AnnData` object ``{shape}``.

    Parameters
    ----------
    path
        Path where to save the dataset.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    The dataset."""

    def _create_signature(self) -> Signature:
        return signature(lambda _: _).replace(
            parameters=[
                Parameter("path", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=PathLike, default=None),
                Parameter("kwargs", kind=Parameter.VAR_KEYWORD, annotation=Any),
            ],
            return_annotation=anndata.AnnData,
        )

    def _download(self, fpath: PathLike, backup_url: str, **kwargs: Any) -> AnnData:
        kwargs.setdefault("sparse", True)
        kwargs.setdefault("cache", True)

        return read(filename=fpath, backup_url=backup_url, **kwargs)

    @property
    def _extension(self) -> str:
        return ".h5ad"


class ImgMetadata(Metadata):
    """Metadata class for :class:`squidpy.im.ImageContainer`."""

    _DOC_FMT = """
    {doc_header}

    The shape of this image is ``{shape}``.

    Parameters
    ----------
    path
        Path where to save the .tiff image.
    kwargs
        Keyword arguments for :meth:`squidpy.im.ImageContainer.add_img`.

    Returns
    -------
    :class:`squidpy.im.ImageContainer` The image data."""
    # not the perfect annotation, but better than nothing
    _EXT = ".tiff"

    def _create_signature(self) -> Signature:
        return signature(lambda _: _).replace(
            parameters=[
                Parameter("path", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=PathLike, default=None),
                Parameter("kwargs", kind=Parameter.VAR_KEYWORD, annotation=Any),
            ],
        )

    def _download(self, fpath: PathLike, backup_url: str, **kwargs: Any) -> Any:
        from squidpy.im import ImageContainer  # type: ignore[attr-defined]

        check_presence_download(Path(fpath), backup_url)

        img = ImageContainer()
        img.add_img(fpath, layer="image", library_id=self.library_id, **kwargs)

        return img

    @property
    def _extension(self) -> str:
        return ".tiff"


def _get_zipped_dataset(folderpath: Path, dataset_name: str, figshare_id: str) -> sd.SpatialData:
    """Returns a specific dataset as SpatialData object. If the file is not present on disk, it will be downloaded and extracted."""

    if not folderpath.is_dir():
        raise ValueError(f"Expected a directory path for `folderpath`, found: {folderpath}")

    download_zip = folderpath / f"{dataset_name}.zip"
    extracted_path = folderpath / f"{dataset_name}.zarr"

    # Return early if data is already extracted
    if extracted_path.exists():
        logg.info(f"Loading existing dataset from {extracted_path}")
        return sd.read_zarr(extracted_path)

    # Download if necessary
    if not download_zip.exists():
        logg.info(f"Downloading Visium H&E SpatialData to {download_zip}")
        try:
            check_presence_download(
                filename=download_zip,
                backup_url=f"https://ndownloader.figshare.com/files/{figshare_id}",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}") from e

    # Extract if necessary
    if not extracted_path.exists():
        logg.info(f"Extracting dataset from {download_zip} to {extracted_path}")
        try:
            shutil.unpack_archive(str(download_zip), folderpath)
        except Exception as e:
            raise RuntimeError(f"Failed to extract dataset: {e}") from e

    if not extracted_path.exists():
        raise RuntimeError(f"Expected extracted data at {extracted_path}, but not found")

    return sd.read_zarr(extracted_path)
