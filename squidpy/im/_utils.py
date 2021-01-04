import tifffile


def _num_pages(fname: str) -> int:
    """Use tifffile to get the number of pages in the tif."""
    with tifffile.TiffFile(fname) as img:
        num_pages = len(img.pages)
    return num_pages
