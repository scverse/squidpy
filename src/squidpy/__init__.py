from importlib.metadata import version

from packaging.version import parse

from squidpy import datasets, gr, im, pl, read, tl

__author__ = __maintainer__ = "Theislab"
__email__ = ", ".join(
    [
        "giovanni.palla@helmholtz-muenchen.de",
        "hannah.spitzer@helmholtz-muenchen.de",
    ]
)
__version__ = "1.2.3"


try:
    __full_version__ = parse(version(__name__))
    __full_version__ = f"{__version__}+{__full_version__.local}" if __full_version__.local else __version__
except ImportError:
    __full_version__ = __version__

del version, parse
