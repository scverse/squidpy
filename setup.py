from pathlib import Path

from setuptools import setup, find_packages

long_description = Path("README.md").read_text("utf-8")

try:
    from spatial_tools import __email__, __author__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ""

setup(
    name="spatial_tools",
    version="0.0.1",
    description="tools for spatial transcriptomics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theislab/spatial-tools",
    author=__author__,
    author_email=__email__,
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()],
    extras_require=dict(
        dev=["pre-commit>=2.7.1"],
        test=["tox>=3.20.1", "pytest-xdist>=2.1.0"],
    ),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
    ],
)
