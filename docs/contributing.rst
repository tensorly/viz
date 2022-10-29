=======================
Contribution guidelines
=======================

All contributions to TLViz are welcome! If you find a bug or an error in the documentation, we very much appreciate
an issue or a pull request!

-----------------
How to contribute
-----------------

If you find a bug, or have an idea for a new feature, and you want to know if the contribution is relevant and not
being worked on, you can open a `new issue <https://github.com/tensorly/viz/issues>`_. For major
bugs, it can also be useful to include a `minimal, reproducible example <https://stackoverflow.com/help/minimal-reproducible-example>`_,
to make it as easy as possible to fix it.

You can submit implementation of new features or bug/documentation fixes as a `pull-request <https://github.com/tensorly/viz/pulls>`_.

-----------------------
Development environment
-----------------------

We recommend using a virtual environment to develop TLViz locally on your machine. For example, with Anaconda

.. code:: bash

    conda create -n tlviz python=3.8 anaconda

Then, you can download the TLViz source code and install it together with all the development dependencies

.. code:: bash

    git clone https://github.com/tensorly/viz.git
    cd viz
    pip install -e .[devel,test,docs]

This will install TLViz in editable mode, so any change to the source code will be applied to the installed
version too.

-----------
Style guide
-----------

TLViz follows the `Black <https://github.com/psf/black>`_ style (with a maximum line length of 120 characters) and
follows most of the `flake8 <https://flake8.pycqa.org/en/latest/>`_ guidelines (except E203, W503). Most style errors
will be fixed automatically in VSCode if you include the following lines in your `settings.json` file

.. code:: json

    {
        "python.linting.flake8Enabled": true,
        "python.formatting.provider": "black",
        "editor.formatOnSave": false,
        "python.linting.flake8Args": [
            "--max-line-length=120"
        ],
        "python.sortImports.args": [
            "--profile",
            "black"
        ],
        "[python]": {
            "editor.codeActionsOnSave": {
                "source.organizeImports": true
            }
        }
    }

----------
Unit tests
----------

TLViz aims to have a high test coverage, so any new code should also include tests, including visualisation.
You can run the tests by running

.. code:: bash

    pytest

To also check the test coverage

.. code:: bash

    coverage html

This will generate a HTML report where you can see all lines not covered by any tests.

-------------
Documentation
-------------

The documentation is generated using sphinx with automatic API documentation from the docstrings and
MathJax for equations. We use `sphinx-gallery <https://sphinx-gallery.github.io/stable/index.html>`_
to generate the example gallery. To expand this, simply add a new example script with a name matching
the pattern `plot_*.py` in the `examples`-directory (make sure to follow the `sphinx-gallery style <https://sphinx-gallery.github.io/stable/syntax.html>`_
for your scripts).

To ensure that the documentation is up to date, we use `doctest <https://docs.python.org/3/library/doctest.html>`_,
which will evaluate all examples and compare with the expected output. Examples should therefore be seeded.
