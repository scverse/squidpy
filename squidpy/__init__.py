from squidpy import gr, im, pl, read, datasets

__author__ = __maintainer__ = "Theislab"
__email__ = ", ".join(
    [
        "giovanni.palla@helmholtz-muenchen.de",
        "hannah.spitzer@helmholtz-muenchen.de",
    ]
)
__version__ = "1.2.1"

try:
    from importlib_metadata import version  # Python < 3.8
except ImportError:
    from importlib.metadata import version  # Python = 3.8

from packaging.version import parse

try:
    __full_version__ = parse(version(__name__))
    __full_version__ = f"{__version__}+{__full_version__.local}" if __full_version__.local else __version__
except ImportError:
    __full_version__ = __version__

del version, parse
