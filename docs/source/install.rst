Install
=======

This section contains the instruction to install ``scnndeconv`` and the toolboxes configuration files
to have a local working instance of ``scnndeconv`` for development.

Install scnndeconv
------------------

To install ``scnndeconv`` you need to have `Pipenv <https://pipenv.pypa.io/en/latest/#install-pipenv-today>`_ installed, as this is used to manage the dependencies.

.. code-block:: shell

    git clone https://github.com/sylvainprigent/scnndeconv.git
    cd scnndeconv
    pipenv sync

Alternatively, it is possible to install the dependencies with pip, but this is not recommended as they might clash with other projects on your computer:

.. code-block:: shell

    pip install --user -r requirements.txt
