from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union, Callable, Optional
from inspect import Parameter, signature
from pathlib import Path
from dataclasses import field, dataclass
import os

from scanpy import read, logging as logg
from anndata import AnnData
import anndata

PathLike = Optional[Union[os.PathLike, str]]
Function_t = Callable[..., Union[AnnData, Any]]


@dataclass(frozen=True)  # type: ignore[misc]
class Metadata(ABC):
    """Base class handling metadata."""

    name: str
    url: str

    doc_header: Optional[str] = field(default=None, repr=False)
    path: Optional[PathLike] = field(default=None, repr=False)
    shape: Optional[Tuple[int, int]] = field(default=None, repr=False)

    _EXT: str = ".h5ad"

    def __post_init__(self) -> None:
        if self.doc_header is None:
            object.__setattr__(self, "doc_header", f"Download `{self.name.title().replace('_', ' ')}` dataset.")
        if self.path is None:
            object.__setattr__(self, "path", os.path.expanduser(f"~/.cache/squidpy/{self.name}"))

    @abstractmethod
    def _download(self, backup_url: str, fpath: PathLike = None, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _create_function(self, name: str, glob_ns: Dict[str, Any]) -> None:
        pass

    def download(self, fpath: PathLike = None, **kwargs: Any) -> Any:
        """Download the dataset into ``fpath``."""
        fpath = str(self.path if fpath is None else fpath)
        if not fpath.endswith(self._EXT):
            fpath += self._EXT

        if os.path.isfile(fpath):
            logg.debug(f"Loading dataset `{self.name}` from `{fpath}`")
        else:
            logg.debug(f"Downloading dataset `{self.name}` from `{self.url}` as `{fpath}`")

        dirname = Path(fpath).parent
        try:
            if not dirname.is_dir():
                logg.debug(f"Creating directory `{dirname}`")
                dirname.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logg.debug(f"Unable to create directory `{dirname}`. Reason `{e}`")

        data = self._download(backup_url=self.url, fpath=fpath, **kwargs)

        if self.shape is not None and data.shape != self.shape:
            raise ValueError(f"Expected the data to have shape `{self.shape}`, found `{data.shape}`.")

        return data


class AMetadata(Metadata):
    """Metadata class for :class:`anndata.AnnData`."""

    _DOC_FMT = """
    {doc_header}

    Parameters
    ----------
    path
        Path where to save the dataset.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    The dataset."""
    _EXT = ".h5ad"

    def _create_function(self, name: str, glob_ns: Dict[str, Any]) -> None:
        sig = signature(lambda _: _)
        sig = sig.replace(
            parameters=[
                Parameter("path", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=PathLike, default=None),
                Parameter("kwargs", kind=Parameter.VAR_KEYWORD, annotation=Any),
            ],
            return_annotation=anndata.AnnData,
        )
        globals()["NoneType"] = type(None)  # __post_init__ return annotation
        globals()[name] = self

        exec(
            f"def {self.name}{sig}:\n"
            f'    """'
            f"    {self._DOC_FMT.format(doc_header=self.doc_header)}"
            f'    """\n'
            f"    return {name}.download(path, **kwargs)".replace(" /,", ""),
            globals(),
            glob_ns,
        )

    def _download(self, backup_url: str, fpath: PathLike = None, **kwargs: Any) -> AnnData:
        kwargs.setdefault("sparse", True)
        kwargs.setdefault("cache", True)

        return read(filename=fpath, backup_url=backup_url, **kwargs)


class ImgMetadata(Metadata):
    """Metadata class for :class:`squidpy.im.ImageContainer`."""

    _DOC_FMT = """
    {doc_header}

    Parameters
    ----------
    path
        Path where to save the dataset.
    kwargs
        Keyword arguments for :meth:`squidpy.pl.ImageContainer.from_url`.

    Returns
    -------
    The dataset."""
    _EXT = ".netcdf"

    def _create_function(self, name: str, glob_ns: Dict[str, Any]) -> None:
        sig = signature(lambda _: _)
        sig = sig.replace(
            parameters=[
                Parameter("path", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=PathLike, default=None),
                Parameter("kwargs", kind=Parameter.VAR_KEYWORD, annotation=Any),
            ],
            return_annotation="sq.im.ImageContainer",
        )
        globals()["NoneType"] = type(None)  # __post_init__ return annotation
        globals()[name] = self

        exec(
            f"def {self.name}{sig}:\n"
            f'    """'
            f"    {self._DOC_FMT.format(doc_header=self.doc_header)}"
            f'    """\n'
            f"    return {name}.download(path, **kwargs)".replace(" /,", ""),
            globals(),
            glob_ns,
        )

    def _download(self, backup_url: str, fpath: PathLike = None, **kwargs: Any) -> Any:
        raise NotImplementedError()
