from pathlib import Path
from setuptools import setup, find_packages

try:
    from squidpy import __email__, __author__, __version__, __maintainer__
except ImportError:
    __author__ = __maintainer__ = "Theislab"
    __email__ = ", ".join(
        [
            "giovanni.palla@helmholtz-muenchen.de",
            "hannah.spitzer@helmholtz-muenchen.de",
        ]
    )
    __version__ = "1.0.1"

setup(
    name="squidpy",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    version=__version__,
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,
    description=Path("README.rst").read_text("utf-8").splitlines()[2],
    long_description=Path("README.rst").read_text("utf-8"),
    long_description_content_type="text/x-rst; charset=UTF-8",
    url="https://github.com/theislab/squidpy",
    download_url="https://pypi.org/project/squidpy/",
    project_urls={
        "Documentation": "https://squidpy.readthedocs.io/en/latest",
        "Source Code": "https://github.com/theislab/squidpy",
    },
    license="BSD",
    platforms=["Linux", "MacOSX"],
    packages=find_packages(),
    zip_safe=False,
    install_requires=[l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()],
    extras_require=dict(
        dev=["pre-commit>=2.9.0"],
        test=["tox>=3.20.1", "pytest-mock"],
        docs=[
            l.strip()
            for l in (Path("docs") / "requirements.txt").read_text("utf-8").splitlines()
            if not l.startswith("-r")
        ],
        interactive=["PyQt5>=5.15.0", "napari>=0.4.2"],
        all=["PyQt5>=5.15.0", "napari<0.4.9", "esda>=2.3.1", "libpysal>=4.3.0", "astropy>=4.1"],
    ),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Typing :: Typed",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords=sorted(
        [
            "single-cell",
            "bio-informatics",
            "spatial transcriptomics",
            "spatial data analysis",
            "image analysis",
            "spatial data analysis",
        ]
    ),
)
