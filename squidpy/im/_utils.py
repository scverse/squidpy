from typing import Set, List, Tuple, Hashable, Iterable

import tifffile


def _num_pages(fname: str) -> int:
    """Use tifffile to get the number of pages in the tif."""
    with tifffile.TiffFile(fname) as img:
        num_pages = len(img.pages)
    return num_pages


def _unique_order_preserving(iterable: Iterable[Hashable]) -> Tuple[List[Hashable], Set[Hashable]]:
    """Remove items from an iterable while preserving the order."""
    seen = set()
    return [i for i in iterable if i not in seen and not seen.add(i)], seen
