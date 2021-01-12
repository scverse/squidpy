Contributing guide
~~~~~~~~~~~~~~~~~~

Table of Contents
=================
- `Contributing to SquidPy`_
- `Codebase structure`_
- `Code style guide`_
- `Testing`_
- `Writing documentation`_
- `Writing tutorials/examples`_
- `Creating a new release`_
- `Submitting a PR`_

Contributing to SquidPy
-----------------------
Clone SquidPy from source as::

    git clone https://github.com/theislab/squidpy
    cd squidpy

Install the test and development mode::

    pip install -e'.[dev,test]'

Optionally install pre-commit. This will ensure that the pushed code passes the linting steps::

    pre-commit install

Although the last step is not necessary, it is highly recommended, since it will help you to pass the linting step
(see `Code style guide`_).

Codebase structure
------------------
The SquidPy project:

- `squidpy <squidpy>`_: the root of the package.

  - `squidpy/gr <squidpy/gr>`__: the graph module, which deals with building a spatial graph,
    running statistical tests on graphs, etc.
  - `squidpy/im <squidpy/im>`__: the image module, which deals with image feature calculation, cropping, etc.
  - `squidpy/pl <squidpy/pl>`__: the plotting module, which contains all the plotting functions
    from the graph and image modules.
  - `squidpy/constants <squidpy/constants>`__: contains internal and (possibly in the near future) external constants.

Tests structure:

- `tests <tests>`_: the root of the package

  - `tests/tests_graph <tests/tests_graph>`__: tests for the graph module.
  - `tests/tests_image <tests/tests_image>`__: tests for the image module.
  - `tests/tests_plotting <tests/tests_plotting>`__ tests for the plotting module.
  - `tests/conftest.py <tests/conftest.py>`__: ``pytest`` fixtures and utility functions.
  - `tests/_images <tests/_images>`__: ground-truth images for plotting tests.
  - `tests/_data <tests/_data>`__: data used for testing, such as ``anndata.AnnData`` or images.

Code style guide
----------------
We rely on ``black`` and ``isort`` to do the most of the formatting - both of them are integrated as pre-commit hooks.
You can use ``tox`` to check the changes::

    tox -e lint

Furthermore, we also require that:

- functions are fully type-annotated.
- exceptions messages are capitalized and end with ``.``.
- warning messages are capitalized and do not end with ``.``.
- when referring to variable inside a message, enclose its name in \`.


Testing
-------
We use ``tox`` to automate our testing, as well as linting and documentation creation. To run the tests, run::

    tox -e py{37,38,39}-{linux,macos}

depending on the Python version(s) in your ``PATH`` and your operating system. We use ``flake8`` and ``mypy`` to further
analyze the code. Use ``# noqa: <error1>,<error2>`` to ignore certain ``flake8`` errors and
``# type: ignore[error1,error2]`` to ignore specific ``mypy`` errors.

To run only a subset of tests, run::

    tox -e <environment> -- <name>

where ``<name>`` can be a path to a file/directory or a name of a test function/class.

If needed, a specific ``tox`` environment can be recreated as::

    tox -e <environment> --recreate

Writing documentation
---------------------
We use ``numpy``-style docstrings for the documentation with the following additions and modifications:

- no type hints in the docstring (applies also for the return statement) are allowed,
  since all functions are required to have the type hints in their signatures.
- when referring to some argument within the same docstring, enclose that reference in \`\`.
- prefer putting references in the ``references.rst`` instead under the ``References`` sections of the docstring.
- use ``docrep`` for repeating documentation

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
<https://github.com/theislab/squidpy_notebooks>`__.
Please refer to this `guide <https://github.com/theislab/squidpy_notebooks/CONTRIBUTING.rst>`__ for more information.

Creating a new release
----------------------
If you are a core developer and you want to create a new release, you need to install ``bump2version`` first as::

    pip install bump2version

Depending on what part of the release you want to update, you can run::

    bump2version {major,minor,patch}

By default, this will create a new tag and automatically update the ``__version__`` whereever necessary, commit the
changes and create a new tag. If you have uncommited files in the tree, you can use ``--allow-dirty`` flag to include
them in the commit -

After the version has been bumped, make sure to push the commit **AND** the newly create tag to the upstream. This
can be done by e.g. setting ``push.followtags=true`` in your git config or use ``git push --atomic <branch> <tag>``.

Submitting a PR
---------------
Before submitting a new pull request, please make sure you followed these instructions:

- make sure that your code follows the above specified conventions
  (see `Code style guide`_ and `Writing documentation`_).
- if applicable, make sure you've added/modified at least 1 test to account for the changes you've made
- make sure that all tests pass locally (see `Testing`_).
- if there is no issue which this PR solves, create a new `one <https://github.com/theislab/squidpy/issues/new>`__
  briefly explaining problem is
