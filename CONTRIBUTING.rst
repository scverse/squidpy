Contributing guide
~~~~~~~~~~~~~~~~~~

Table of Contents
=================
- `Contributing to Squidpy`_
- `Codebase structure`_
- `Code style guide`_
- `Testing`_
- `Writing documentation`_
- `Writing tutorials/examples`_
- `Making a new release`_
- `Creating release notes`_
- `Submitting a PR`_
- `Troubleshooting`_

Contributing to Squidpy
-----------------------
Clone Squidpy from source as::

    git clone https://github.com/scverse/squidpy
    cd squidpy
    git checkout main

Install the test and development mode::

    pip install -e'.[dev,test]'

Optionally install pre-commit. This will ensure that the pushed code passes the linting steps::

    pre-commit install

Although the last step is not necessary, it is highly recommended, since it will help you to pass the linting step
(see `Code style guide`_). If you did install ``pre-commit`` but are unable in deciphering some flags, you can
still commit using the ``--no-verify``.

Codebase structure
------------------
The Squidpy project:

- `squidpy <squidpy>`_: the root of the package.

  - `squidpy/gr <squidpy/gr>`_: the graph module, which deals with building a spatial graph,
    running statistical tests on graphs and features etc.
  - `squidpy/im <squidpy/im>`_: the image module, which deals with image feature calculation, cropping, etc.
  - `squidpy/pl <squidpy/pl>`_: the plotting module, which contains all the plotting functions
    from the graph and image modules.
  - `squidpy/constants <squidpy/constants>`_: contains internal and (possibly in the near future) external constants.

Tests structure:

- `tests <tests>`_: the root of the package

  - `tests/graph <tests/graph>`_: tests for the graph module.
  - `tests/image <tests/image>`_: tests for the image module.
  - `tests/plotting <tests/plotting>`_ tests for the plotting module.
  - `tests/conftest.py <tests/conftest.py>`_: ``pytest`` fixtures and utility functions.
  - `tests/_images <tests/_images>`_: ground-truth images for plotting tests.
  - `tests/_data <tests/_data>`_: data used for testing, such as ``anndata.AnnData`` or images.

Code style guide
----------------
We rely on ``black`` and ``isort`` to do the most of the formatting - both of them are integrated as pre-commit hooks.
You can use ``tox`` to check the changes::

    tox -e lint

Furthermore, we also require that:

- functions are fully type-annotated.
- exception messages are capitalized and end with ``.``.
- warning messages are capitalized and do not end with ``.``.
- when referring to variable inside an error/warning message, enclose its name in \`.
- when referring to variable inside a docstrings, enclose its name in \``.

Testing
-------
We use ``tox`` to automate our testing, as well as linting and documentation creation. To run the tests, run::

    tox -e py{38,39,310}-{linux,macos}

depending on the Python version(s) in your ``PATH`` and your operating system. We use ``flake8`` and ``mypy`` to further
analyze the code. Use ``# noqa: <error1>,<error2>`` to ignore certain ``flake8`` errors and
``# type: ignore[error1,error2]`` to ignore specific ``mypy`` errors.

To run only a subset of tests, run::

    tox -e <environment> -- <name>

where ``<name>`` can be a path to a test file/directory or a name of a test function/class.
For example, to run only the tests in the ``nhood`` module, use::

    tox -e py39-linux -- tests/graph/test_nhood.py

If needed, a specific ``tox`` environment can be recreated as::

    tox -e <environment> --recreate

Writing documentation
---------------------
We use ``numpy``-style docstrings for the documentation with the following additions and modifications:

- no type hints in the docstring (applies also for the return statement) are allowed,
  since all functions are required to have the type hints in their signatures.
- when referring to some argument within the same docstring, enclose that reference in \`\`.
- prefer putting references in the ``references.bib`` instead under the ``References`` sections of the docstring.
- use ``docrep`` for repeating documentation.

In order to build the documentation, run::

    tox -e docs

Since the tutorials are hosted on a separate repository (see `Writing tutorials/examples`_), we download the newest
tutorials/examples from there and build the documentation here.

To validate the links inside the documentation, run::

    tox -e check-docs

If you need to clean the artifacts from previous documentation builds, run::

    tox -e clean-docs

Writing tutorials/examples
--------------------------
Tutorials and examples are hosted on a separate repository called `squidpy_notebooks
<https://github.com/scverse/squidpy_notebooks>`_.
Please refer to this `guide <https://github.com/scverse/squidpy_notebooks/CONTRIBUTING.rst>`_ for more information.

Submitting a PR
---------------
Before submitting a new pull request, please make sure you followed these instructions:

- make sure that you've branched off ``main`` and are merging into ``main``
- make sure that your code follows the above specified conventions
  (see `Code style guide`_ and `Writing documentation`_).
- if applicable, make sure you've added/modified at least 1 test to account for the changes you've made
- make sure that all tests pass locally (see `Testing`_).
- if there is no issue which this PR solves, create a new `one <https://github.com/scverse/squidpy/issues/new>`_
  briefly explaining what the problem is.
- make sure that the section under ``## Description`` is properly formatted if automatically generating release notes,
  see also `Creating release notes`_.

Making a new release
--------------------
New release is always created when a new tag is pushed to GitHub. When that happens, a new CI job starts the
testing machinery. If all the tests pass, new release will be created on PyPI. Bioconda will automatically notice that
a new release has been made and an automatic PR will be made to
`bioconda-recipes <https://github.com/bioconda/bioconda-recipes/pulls>`_.
Extra care has to be taken when updating runtime dependencies - this is not automatically picked up by Bioconda
and a separate PR with the updated ``recipe.yaml`` will have to be made.

Easiest way to create a new release it to create a branch named ``release/vX.X.X`` and push it onto GitHub. The CI
will take care of the following:

- create the new release notes
- bump the version and create a new tag
- run tests on the ``release/vX.X.X`` branch
- publish on PyPI after all the tests have passed
- merge ``release/vX.X.X`` into ``main``

It is possible to create a new release using ``bump2version``, which can be installed as::

    pip install bump2version

Depending on what part of the version you want to update, you can run on ``main``::

    bump2version {major,minor,patch}

By default, this will create a new tagged commit, automatically update the ``__version__`` wherever necessary.
Afterwards, you can just push the changes to upstream by running::

    git push --atomic <branch> <tag>

or set ``push.followtags=true`` in your git config and do a regular ``git push``. In this case, CI will not
create any release notes, run tests or do any merges.

Creating release notes
----------------------
Please take a look at the other release notes for formatting style. We are exploring other options for automatic release notes generation.

Troubleshooting
---------------
- **The enchant C library was not found**
  This can happen during the documentation build and because of a missing dependency for spell checker.
  The installation instructions for the dependency can be found
  `here <https://pyenchant.github.io/pyenchant/install.html#installing-the-enchant-c-library>`_.
