from pathlib import Path

from setuptools import setup, find_packages

long_description = Path("README.rst").read_text("utf-8")

try:
    from squidpy import __email__, __author__, __version__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ""
    __version__ = "0.0.0"

setup(
    name="squidpy",
    version=__version__,
    description="tools for spatial transcriptomics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theislab/squidpy",
    author=__author__,
    author_email=__email__,
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()],
    extras_require=dict(
        dev=["pre-commit>=2.7.1"],
        test=["tox>=3.20.1"],
        docs=[
            l.strip()
            for l in (Path("docs") / "requirements.txt").read_text("utf-8").splitlines()
            if not l.startswith("-r")
        ],
    ),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
    ],
)
