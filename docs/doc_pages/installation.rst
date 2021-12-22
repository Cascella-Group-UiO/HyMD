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
