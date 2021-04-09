from typing import Any, Dict, List, Tuple, Union, Callable, Optional
from logging import info, warning
from pathlib import Path
from urllib.parse import urljoin
from enchant.tokenize import Filter
from sphinx_gallery.directives import MiniGallery
import os
import re
import requests

HERE = Path(__file__).parent
ENDPOINT_FMT = "https://api.github.com/repos/{org}/{repo}/contents/docs/source/"
REF = "master"

HEADERS = {"accept": "application/vnd.github.v3+json"}
CHUNK_SIZE = 4 * 1024
DEPTH = 5

# nbsphinx
FIXED_TUTORIALS_DIR = "external_tutorials"
# sphinx-gallery
EXAMPLES_DIR = "auto_examples"
TUTORIALS_DIR = "auto_tutorials"
GENMOD_DIR = "gen_modules"


def _cleanup(fn: Callable[..., Tuple[bool, Any]]) -> Callable[..., Tuple[bool, Any]]:
    def decorator(*args: Any, **kwargs: Any) -> Tuple[bool, Any]:
        try:
            ok, resp = fn(*args, **kwargs)
        except Exception as e:
            ok, resp = False, e

        if not ok:
            path = Path(kwargs.pop("path"))
            try:
                if path.is_dir():
                    path.rmdir()
                elif path.is_file():
                    path.unlink()
            except OSError as e:
                info(f"Not cleaning `{path}`. Reason: `{e}`")

        return ok, resp

    return decorator


@_cleanup
def _download_file(url: str, path: str) -> Tuple[bool, int]:
    ix = url.rfind("?")
    if ix != -1:
        url = url[:ix]

    url = f"{url}?ref={REF}"

    info(f"Processing URL `{url}`")
    resp = requests.get(url, headers=HEADERS)
    if resp.ok:
        with open(path, "wb") as fout:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                fout.write(chunk)

    return resp.ok, resp.status_code


@_cleanup
def _download_dir(url: str, *, path: Union[str, Path], depth: int) -> Tuple[bool, Optional[Union[Exception, str]]]:
    if depth == 0:
        return False, f"Maximum depth `{DEPTH}` reached."

    info(f"Processing URL `{url}`")
    resp = requests.get(url, headers=HEADERS)
    if not resp.ok:
        return False, f"Unable to fetch `{url}`, status: {resp.status_code}."

    path = Path(path)
    path.mkdir(exist_ok=True)

    for item in resp.json():
        dest = path / item["name"]
        if item["type"] == "file":
            ok, status = _download_file(item["download_url"], path=dest)
        elif item["type"] == "dir":
            ok, status = _download_dir(item["url"], path=dest, depth=depth - 1)
        else:
            raise NotImplementedError(f"Invalid type: `{item['type']}`.")

        if not ok:
            raise RuntimeError(f"Unable to process `{dest}`. Reason: `{status}`.")

    return True, None


def _download_notebooks(org: str, repo: str, raise_exc: bool = False) -> None:
    if not int(os.environ.get("SQUIDPY_DOWNLOAD_NOTEBOOKS", 1)):
        info("Not downloading notebooks because a flag is set")
        return

    ep = ENDPOINT_FMT.format(org=org, repo=repo)
    for path in [FIXED_TUTORIALS_DIR, TUTORIALS_DIR, EXAMPLES_DIR, GENMOD_DIR]:
        ok, reason = _download_dir(urljoin(ep, path), path=path, depth=DEPTH)

        if not ok:
            if raise_exc:
                raise RuntimeError(reason)
            warning(reason)


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
        else:
            res[str(fname)[:-3]] = "_static/img/squidpy_vertical.png"

    return res


class ModnameFilter(Filter):
    """
    Ignore module names.
    """

    _pat = re.compile(r"squidpy\.(im|gr|pl|datasets)\..+")

    def _skip(self, word: str) -> bool:
        return self._pat.match(word) is not None


class SignatureFilter(Filter):
    """
    Ignore function signature artifacts.
    """

    _pat = re.compile(r"\([^,]+?(\[?, [^,]*)*\)")

    def _skip(self, word: str) -> bool:
        return word == "img[" or word == "adata,"
