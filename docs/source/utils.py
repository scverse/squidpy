from git import Repo
from shutil import rmtree, copytree
from typing import Dict, List, Union
from logging import info, warning
from pathlib import Path
from tempfile import TemporaryDirectory
from enchant.tokenize import Filter
from sphinx_gallery.directives import MiniGallery
import os
import re
import subprocess

HERE = Path(__file__).parent


def _is_master() -> bool:
    try:
        r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"Subprocess returned return code `{r.returncode}`.")

        ref = r.stdout.decode().strip()
        if ref not in ("master", "dev"):
            # most updates happen on branch that is merged to dev
            return False

        return ref == "master"

    except Exception as e:
        warning(f"Unable to fetch ref, reason: `{e}`. Using `master`")
        return True


def _fetch_notebooks(repo_url: str) -> None:
    def copy_files(repo_path: Union[str, Path]) -> None:
        repo_path = Path(repo_path)

        for dirname in ["external_tutorials", "auto_examples", "auto_tutorials", "gen_modules"]:
            rmtree(dirname, ignore_errors=True)  # locally re-cloning
            copytree(repo_path / "docs" / "source" / dirname, dirname)

    def fetch_remote(repo_url: str) -> None:
        info(f"Fetching notebooks from repo `{repo_url}`")
        with TemporaryDirectory() as repo_dir:
            ref = "master" if _is_master() else "dev"
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
    except Exception as e:
        warning(f"Unable to fetch notebooks locally from `{notebooks_local_path}`, reason: `{e}`. Trying remote")
        download = int(os.environ.get("SQUIDPY_DOWNLOAD_NOTEBOOKS", 1))
        if not download:
            # use possibly old files, otherwise, bunch of warnings will be shown
            info(f"Not fetching notebooks from remove because `SQUIDPY_DOWNLOAD_NOTEBOOKS={download}`")
            return

        fetch_remote(repo_url)


class MaybeMiniGallery(MiniGallery):
    def run(self) -> List[str]:
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


def _get_thumbnails(root: Union[str, Path]) -> Dict[str, str]:
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

    _pat = re.compile(r"squidpy\.(im|gr|pl|datasets|ImageContainer).+")

    def _skip(self, word: str) -> bool:
        return self._pat.match(word) is not None


class SignatureFilter(Filter):
    """Ignore function signature artifacts."""

    def _skip(self, word: str) -> bool:
        # TODO: find a better way (img/func is problem)
        return word in ("img[", "imgs[", "img", "func[", "func", "combine_attrs", "**kwargs", "n_iter")
