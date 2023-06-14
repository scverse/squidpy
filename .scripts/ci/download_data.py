#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any

_CNT = 0  # increment this when you want to rebuild the CI cache
_ROOT = Path.home() / ".cache" / "squidpy"


def _print_message(func_name: str, path: Path, *, dry_run: bool = False) -> None:
    prefix = "[DRY RUN]" if dry_run else ""
    if path.is_file():
        print(f"{prefix}[Loading]     {func_name:>25} <- {str(path):>25}")
    else:
        print(f"{prefix}[Downloading] {func_name:>25} -> {str(path):>25}")


def _maybe_download_data(func_name: str, path: Path) -> Any:
    import squidpy as sq

    try:
        return getattr(sq.datasets, func_name)(path=path)
    except Exception as e:  # noqa: BLE001
        print(f"File {str(path):>25} seems to be corrupted: {e}. Removing and retrying")
        path.unlink()

        return getattr(sq.datasets, func_name)(path=path)


def main(args: argparse.Namespace) -> None:
    import squidpy as sq
    from anndata import AnnData

    all_datasets = sq.datasets._dataset.__all__ + sq.datasets._image.__all__
    all_extensions = ["h5ad"] * len(sq.datasets._dataset.__all__) + ["tiff"] * len(sq.datasets._image.__all__)

    if args.dry_run:
        for func_name, ext in zip(all_datasets, all_extensions):
            path = _ROOT / f"{func_name}.{ext}"
            _print_message(func_name, path, dry_run=True)
        return

    # could be parallelized, but on CI it largely does not matter (usually limited to 2 cores + bandwidth limit)
    for func_name, ext in zip(all_datasets, all_extensions):
        path = _ROOT / f"{func_name}.{ext}"

        _print_message(func_name, path)
        obj = _maybe_download_data(func_name, path)

        # we could do without the AnnData check as well (1 less req. in tox.ini), but it's better to be safe
        assert isinstance(obj, (AnnData, sq.im.ImageContainer)), type(obj)
        assert path.is_file(), path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data used for tutorials/examples.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not download any data, just print what would be downloaded."
    )

    main(parser.parse_args())
