from copy import copy
from typing import Any, Tuple, Union, Callable, Optional
from inspect import Parameter, signature
from dataclasses import field, dataclass
import os

from scanpy import read, logging as logg
from anndata import AnnData
import anndata

PathLike = Optional[Union[os.PathLike, str]]
Function_t = Callable[..., AnnData]


# TODO: this is metadata for anndata
@dataclass(frozen=True)
class Metadata:
    """Dataset metadata class."""

    name: str
    url: str

    doc_header: Optional[str] = field(default=None, repr=False)
    path: Optional[PathLike] = field(default=None, repr=False)
    shape: Optional[Tuple[int, int]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.doc_header is None:
            object.__setattr__(self, "doc_header", f"Download `{self.name.title().replace('_', ' ')}` dataset.")
        if self.path is None:
            object.__setattr__(self, "path", os.path.expanduser(f"~/.cache/squidpy/{self.name}"))

    def download(self, fpath: PathLike = None, **kwargs: Any) -> AnnData:
        """Download the dataset inth ``fpath``."""
        fpath = str(self.path if fpath is None else fpath)
        if not fpath.endswith(".h5ad"):
            fpath += ".h5ad"

        if os.path.isfile(fpath):
            logg.debug(f"Loading dataset `{self.name}` from `{fpath}`")
        else:
            logg.debug(f"Downloading dataset `{self.name}` from `{self.url}` as `{fpath}`")

        dirname, _ = os.path.split(fpath)
        try:
            if not os.path.isdir(dirname):
                logg.debug(f"Creating directory `{dirname!r}`")
                os.makedirs(dirname, exist_ok=True)
        except OSError as e:
            logg.debug(f"Unable to create directory `{dirname!r}`. Reason `{e}`")

        kwargs.setdefault("sparse", True)
        kwargs.setdefault("cache", True)

        adata = read(fpath, backup_url=self.url, **kwargs)

        if self.shape is not None and adata.shape != Metadata.shape:
            raise ValueError(f"Expected `anndata.AnnData` object to have shape `{self.shape}`, found `{adata.shape}`.")

        # TODO: 4i warns, maybe uncomment (and/or add obs_names make_unique?)
        # adata.var_names_make_unique()

        return adata


_four_i = Metadata(
    name="four_i",
    doc_header="This is a simple header, only supports 1 line.",
    url="https://ndownloader.figshare.com/files/26098409?private_link=1d883cf23fda2e9d932c",
)
_imc = Metadata(name="imc", url="https://ndownloader.figshare.com/files/26098406?private_link=91bb0e13dffde129b10d")
_seqfish = Metadata(
    name="seqfish", url="https://ndownloader.figshare.com/files/26098403?private_link=4d4cbc43b4a74c52ce9d"
)
_visium_hne_adata = Metadata(
    name="visium_hne_adata", url="https://ndownloader.figshare.com/files/26098397?private_link=d7e9f517da588e6bd5dc"
)
_visium_fluo_adata = Metadata(
    name="visium_fluo_adata", url="https://ndownloader.figshare.com/files/26098391?private_link=f11ceaac4f55a6ceb817"
)
_visium_hne_adata_crop = Metadata(
    name="visium_hne_adata_crop",
    url="https://ndownloader.figshare.com/files/26098382?private_link=564b10d7fa56c2370daf",
)
_visium_fluo_adata_crop = Metadata(
    name="visium_fluo_adata_crop",
    url="https://ndownloader.figshare.com/files/26098376?private_link=96018cd91855fa6b32dc",
)
# TODO: create Metadate for images
_visium_fluo_image_crop = Metadata(
    name="visium_fluo_image_crop",
    url="https://ndownloader.figshare.com/files/26098364?private_link=4b3abee143a3852b9040",
)
_visium_hne_image_crop = Metadata(
    name="visium_hne_image_crop",
    url="https://ndownloader.figshare.com/files/26098328?private_link=33ef1d60cfcadb91518c",
)
_visium_hne_image = Metadata(
    name="visium_hne_image", url="https://ndownloader.figshare.com/files/26098124?private_link=9339d85896f60e41bf19"
)


# TODO: add note which says it's dimension (and/or//
_doc_fmt = """
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


def _create_function(name: str, md: Metadata) -> None:
    sig = signature(lambda _: _)
    sig = sig.replace(
        parameters=[
            Parameter("path", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=PathLike, default=None),
            Parameter("kwargs", kind=Parameter.VAR_KEYWORD, annotation=Any),
        ],
        return_annotation=anndata.AnnData,
    )
    globals()["NoneType"] = type(None)  # __post_init__ return annotation

    exec(
        f"def {md.name}{sig}:\n"
        f'    """'
        f"    {_doc_fmt.format(doc_header=md.doc_header)}"
        f'    """\n'
        f"    {name}.download(path, **kwargs)".replace(" /,", ""),
        globals(),
        globals(),
    )


for name, var in copy(locals()).items():
    if isinstance(var, Metadata):
        _create_function(name, var)


__all__ = [  # noqa: F822
    "four_i",
    "imc",
    "seqfish",
    "visium_hne_adata",
    "visium_hne_adata_crop",
    "visium_fluo_adata",
    "visium_fluo_adata_crop",
    "visium_fluo_image_crop",
    "visium_hne_image_crop",
    "visium_hne_image",
]
