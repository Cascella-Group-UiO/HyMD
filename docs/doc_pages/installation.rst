.. _installation-label:

Installation
############
HyMD may be installed using :code:`pip` by

.. code-block:: bash

   python3 -m pip install hymd


Dependencies
============
HyMD **requires** a working MPI compiler and HDF5. On an Ubuntu system, this
may be installed via

.. code-block:: bash

   sudo apt-get update -y
   sudo apt-get install -y python3-mpi4py libhdf5-dev

It is **recommended** to have MPI-enabled HDF5 and h5py. Installing both may
be done by

.. code-block:: bash

   sudo apt-get update -y
   sudo apt-get install python3-pip libhdf5-mpi-dev
   export CC=mpicc
   export HDF5_MPI="ON"
   python3 -m pip install --no-binary=h5py h5py

A non-MPI-enabled HDF5 library with corresponding :code:`h5py` will work
(mostly), but is inconvenient and slow. Running parallel simulations without a
MPI-enabled HDF5 library available necessitates the use of the
:code:`--disable-mpio` argument to HyMD, see :ref:`commandline-label`. Note that
due to the way HyMD is built, a working MPI compiler is required even if all
intended simulations are serial.

Install in docker
=================
A docker image with build essentials setup is available at `dockerhub`_ with tag
:code:`mortele/hymd`,

.. code-block:: bash

   docker pull mortele/hymd:1.0
   docker run -it mortele/hynd:1.0
   /app# python3 -m pip install hymd
   /app# python3 -m hymd [CONFIGURATION_FILE] [TOPOLOGY_FILE]

.. _dockerhub:
   https://hub.docker.com/repository/docker/mortele/hymd


Common build issues
===================

Numpy errors while importing the Fortran force kernels
------------------------------------------------------

.. code-block:: python3

    RuntimeError: module compiled against API version 0xe but this version of numpy is 0xd

    Traceback (most recent call last):

      (...)

    File "/..../HyMD/hymd/__init__.py", line 2, in <module>
      from .main import main  # noqa: F401
    File "/..../HyMD/hymd/main.py", line 10, in <module>
      from .configure_runtime import configure_runtime
    File "/..../hymd/configure_runtime.py", line 12, in <module>
      from .input_parser import read_config_toml, parse_config_toml
    File "/..../HyMD/hymd/input_parser.py", line 12, in <module>
      from .force import Bond, Angle, Dihedral, Chi
    File "/..../HyMD/hymd/force.py", line 8, in <module>
      from force_kernels import (  # noqa: F401
    ImportError: numpy.core.multiarray failed to import

can normally be fixed by updating numpy versions,

.. code-block:: bash

    python3 -m pip install -U numpy
