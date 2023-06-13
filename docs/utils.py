import os
import re
from logging import info, warning
from pathlib import Path
from shutil import copytree, rmtree
from tempfile import TemporaryDirectory
from typing import Any, Dict, ForwardRef, List, Union

from enchant.tokenize import Filter
from git import Repo
from sphinx_gallery.directives import MiniGallery

HERE = Path(__file__).parent


def _fetch_notebooks(repo_url: str) -> None:
    def copy_files(repo_path: Union[str, Path]) -> None:
        repo_path = Path(repo_path)

        for dirname in ["tutorials", "auto_examples", "gen_modules"]:
            rmtree(dirname, ignore_errors=True)  # locally re-cloning
            copytree(repo_path / "docs" / "source" / dirname, dirname)

    def fetch_remote(repo_url: str) -> None:
        info(f"Fetching notebooks from repo `{repo_url}`")
        with TemporaryDirectory() as repo_dir:
            ref = "main"
            repo = Repo.clone_from(repo_url, repo_dir, depth=1, branch=ref)
            repo.git.checkout(ref, force=True)

            copy_files(repo_dir)

    def fetch_local(repo_path: Union[str, Path]) -> None:
        info(f"Fetching notebooks from local path `{repo_path}`")
        repo_path = Path(repo_path)
        if not repo_path.is_dir():
            raise OSError(f"Path `{repo_path}` is not a directory.")

        copy_files(repo_path)

    notebooks_local_path = Path(
        os.environ.get("SQUIDPY_NOTEBOOKS_PATH", HERE.absolute().parent.parent.parent / "squidpy_notebooks")
    )
    try:
        fetch_local(notebooks_local_path)
    except Exception as e:  # noqa: BLE001
        warning(f"Unable to fetch notebooks locally from `{notebooks_local_path}`, reason: `{e}`. Trying remote")
        download = int(os.environ.get("SQUIDPY_DOWNLOAD_NOTEBOOKS", 1))
        if not download:
            # use possibly old files, otherwise, bunch of warnings will be shown
            info(f"Not fetching notebooks from remove because `SQUIDPY_DOWNLOAD_NOTEBOOKS={download}`")
            return

        fetch_remote(repo_url)


class MaybeMiniGallery(MiniGallery):
    def run(self) -> list[str]:
        config = self.state.document.settings.env.config
        backreferences_dir = config.sphinx_gallery_conf["backreferences_dir"]
        obj_list = self.arguments[0].split()

        new_list = []
        for obj in obj_list:
            path = os.path.join("/", backreferences_dir, f"{obj}.examples")  # Sphinx treats this as the source dir

            if (HERE / path[1:]).exists():
                new_list.append(obj)

        self.arguments[0] = " ".join(new_list)
        try:
            return super().run()  # type: ignore[no-any-return]
        except UnboundLocalError:
            # no gallery files
            return []


def _get_thumbnails(root: Union[str, Path]) -> dict[str, str]:
    res = {}
    root = Path(root)
    thumb_path = Path(__file__).parent.parent.parent / "docs" / "source"

    for fname in root.glob("**/*.py"):
        path, name = os.path.split(str(fname)[:-3])
        thumb_fname = f"sphx_glr_{name}_thumb.png"
        if (thumb_path / path / "images" / "thumb" / thumb_fname).is_file():
            res[str(fname)[:-3]] = f"_images/{thumb_fname}"

    res["**"] = "_static/img/squidpy_vertical.png"

    return res


class ModnameFilter(Filter):
    """Ignore module names."""

    _pat = re.compile(r"squidpy\.(im|gr|pl|read|datasets|ImageContainer).+")

    def _skip(self, word: str) -> bool:
        return self._pat.match(word) is not None


class SignatureFilter(Filter):
    """Ignore function signature artifacts."""

    def _skip(self, word: str) -> bool:
        # TODO(michalk8): find a better way
        return word in ("img[", "imgs[", "img", "img_key", "func[", "func", "combine_attrs", "**kwargs", "n_iter")


# allow `<type_1> | <type_2> | ... | <type_n>` expression for sphinx-autodoc-typehints
def _fwd_ref_init(
    self: ForwardRef,
    arg: str,
    is_argument: bool = True,
    module: Any = None,
    *,
    is_class: bool = False,
) -> None:
    if not isinstance(arg, str):
        raise TypeError(f"Forward reference must be a string -- got {arg!r}")
    if " | " in arg:
        arg = "Union[" + ", ".join(arg.split(" | ")) + "]"
    try:
        code = compile(arg, "<string>", "eval")
    except SyntaxError:
        raise SyntaxError(f"Forward reference must be an expression -- got {arg!r}")
    self.__forward_arg__ = arg
    self.__forward_code__ = code
    self.__forward_evaluated__ = False
    self.__forward_value__ = None
    self.__forward_is_argument__ = is_argument
    try:
        self.__forward_is_class__ = is_class
    except AttributeError:
        pass
    try:
        self.__forward_module__ = module
    except AttributeError:
        pass


ForwardRef.__init__ = _fwd_ref_init  # type: ignore[method-assign]
