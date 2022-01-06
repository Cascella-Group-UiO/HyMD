.. _installation-label:

Installation
############
HyMD may be installed using :code:`pip` by

.. code-block:: bash

   python3 -m pip install --upgrade pip
   python3 -m pip install --upgrade numpy mpi4py cython
   python3 -m pip install hymd


Dependencies
============
HyMD **requires** a working MPI compiler and HDF5. On an Ubuntu system, this
may be installed via

.. code-block:: bash

   sudo apt-get update -y
   sudo apt-get install -y libhdf5-openmpi-dev python3-mpi4py
   CC="mpicc" HDF5_MPI="ON" python3 -m pip install --no-cache-dir --no-binary=h5py h5py

It is **highly recommended** to have MPI-enabled HDF5 and h5py. A
non-MPI-enabled h5py may be installed by simply

.. code-block:: bash

   sudo apt-get update -y
   sudo apt-get install libhdf5-serial-dev
   python3 -m pip --upgrade install h5py

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


Common issues
=============

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


Error building pfft-python due to missing curl/wget
---------------------------------------------------

.. code-block:: python3

    Building wheel for pfft-python (setup.py) ... error
    ERROR: Command errored out with exit status 1:
    command: /usr/bin/python3 -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-fr6nt9m4/pfft-python/setup.py'"'"'; __file__='"'"'/tmp/pip-install-fr6nt9m4/pfft-python/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' bdist_wheel -d /tmp/pip-wheel-ne5et1y_
    cwd: /tmp/pip-install-fr6nt9m4/pfft-python/
    Complete output (56 lines):
    running bdist_wheel
    running build
    running build_py

      (...)

    curl -L -o /tmp/pip-install-fr6nt9m4/pfft-python/depends/..//depends/pfft-1.0.8-alpha3-fftw3-2don2d.tar.gz https://github.com/rainwoodman/pfft/releases/download/1.0.8-alpha3-fftw3-2don2d/pfft-1.0.8-alpha3-fftw3-2don2d.tar.gz
    /tmp/pip-install-fr6nt9m4/pfft-python/depends/install_pfft.sh: 19: curl: not found
    wget -P /tmp/pip-install-fr6nt9m4/pfft-python/depends/..//depends/ https://github.com/rainwoodman/pfft/releases/download/1.0.8-alpha3-fftw3-2don2d/pfft-1.0.8-alpha3-fftw3-2don2d.tar.gz
    /tmp/pip-install-fr6nt9m4/pfft-python/depends/install_pfft.sh: 26: wget: not found
    Failed to get https://github.com/rainwoodman/pfft/releases/download/1.0.8-alpha3-fftw3-2don2d/pfft-1.0.8-alpha3-fftw3-2don2d.tar.gz
    Please check curl or wget
    You can also download it manually to /tmp/pip-install-fr6nt9m4/pfft-python/depends/..//depends/
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/tmp/pip-install-fr6nt9m4/pfft-python/setup.py", line 86, in <module>
        setup(
      File "/usr/lib/python3/dist-packages/setuptools/__init__.py", line 144, in setup
        return distutils.core.setup(**attrs)
      File "/usr/lib/python3.8/distutils/core.py", line 148, in setup

      (...)

      File "/tmp/pip-install-fr6nt9m4/pfft-python/setup.py", line 56, in build_extensions
        build_pfft(self.pfft_build_dir, self.mpicc, ' '.join(self.compiler.compiler_so[1:]))
      File "/tmp/pip-install-fr6nt9m4/pfft-python/setup.py", line 28, in build_pfft
        raise ValueError("could not build fftw; check MPICC?")
    ValueError: could not build fftw; check MPICC?
    ----------------------------------------
    ERROR: Failed building wheel for pfft-python
    Running setup.py clean for pfft-python
    Failed to build pfft-python

can be fixed by installing either `curl`_ or `wget`_

.. code-block:: bash

    apt-get install -y curl wget


.. _`curl`:
   https://curl.se/

.. _`wget`:
   https://www.gnu.org/software/wget/
