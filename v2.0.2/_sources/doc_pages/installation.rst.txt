.. _installation-label:

Installation
############

Dependencies
============

**Installing non-Python dependencies** may be done by

.. tabs::

   .. group-tab:: Ubuntu (apt)

       .. code-block:: bash

           sudo apt-get update -y
           sudo apt-get install -y python3 python3-pip  # Install python3 and pip
           sudo apt-get install -y libopenmpi-dev       # Install MPI development headers
           sudo apt-get install -y libhdf5-openmpi-dev  # Install MPI-enabled HDF5
           sudo apt-get install -y pkg-config           # Install pkg-config, required for h5py install
           sudo apt-get install -y curl wget

   .. group-tab:: Fedora (dnf)

       .. code-block:: bash

           sudo dnf update -y
           sudo dnf install -y python3.9 python3-devel  # Install python3 and pip
           sudo dnf install -y openmpi-devel            # Install MPI development headers
           sudo dnf install -y hdf5-openmpi-devel       # Install MPI-enabled HDF5
           sudo dnf install -y curl wget


      The automatic download and building of PFFT has some issues, so we
      manually build FFTW and PFFT before calling :code:`pip3 install pfft-python`.

      .. code-block:: bash

           sudo dnf install -y git
           sudo dnf install -y libtool
           sudo dnf install -y fftw-openmpi-devel fftw-openmpi-libs
           export PATH=$PATH:/usr/lib64/openmpi/bin/                 # Ensure MPICC is in path
           git clone https://github.com/mpip/pfft.git
           cd pfft/
           ./bootstrap.sh
           ./configure
           make
           make install


   .. group-tab:: Mac OSX (brew)

       .. code-block:: bash

           brew update
           brew install python      # Install python3 and pip
           brew install open-mpi    # Install MPI development headers
           brew install hdf5-mpi    # Install MPI-enabled HDF5
           brew install pkg-config  # Install pkg-config, required for h5py install
           brew install curl wget

.. warning::
   There might be memory leaks when using HyMD with OpenMPI 4.1.1.
   Therefore, using a newer version of OpenMPI is recommended.
   See `Issue #186 <https://github.com/Cascella-Group-UiO/HyMD/issues/186>`_ for more details.


**Installing Python dependencies** may be done by

.. tabs::

   .. group-tab:: Ubuntu (apt)

       .. code-block:: bash

           python3 -m pip install --upgrade pip
           CC="mpicc" HDF5_MPI="ON" python3 -m pip install --no-binary=h5py h5py
           python3 -m pip install mpi4py numpy cython


   .. group-tab:: Fedora (dnf)

       .. code-block:: bash

          python3.9 -m ensurepip --upgrade
          PATH=$PATH:/usr/lib64/openmpi/bin/:/usr/lib64/openmpi/lib/
          python3.9 -m pip install mpi4py numpy cython
          export HDF5_DIR="/usr/lib64/openmpi/"
          CC="mpicc" HDF5_MPI="ON" python3.9 -m pip install --no-binary=h5py h5py

   .. group-tab:: Mac OSX (brew)

       Find the location of the installed :code:`hdf5-mpi` package by

       .. code-block:: bash

           find /usr -iname "*hdf5.h"

       or

       .. code-block:: bash

           brew info hdf5-mpi

      and extract the path, which will look like for example
      :code:`/usr/local/Cellar/hdf5-mpi/1.13.0/`. Export it as :code:`HDF5_DIR`

       .. code-block:: bash

           python3 -m ensurepip --upgrade
           export HDF5_DIR="/usr/local/Cellar/hdf5-mpi/1.13.0/"
           CC="mpicc" HDF5_MPI="ON" python3 -m pip install --no-binary=h5py h5py
           python3 -m pip install mpi4py numpy cython


.. warning::

   If MPI-enabled HDF5 and :code:`h5py` can not be installed, limited support
   for serial HDF5 is available. Note that having MPI-enabled file IO is
   **highly recommended**, and simulation performance under serial HDF5 will
   potentially be very low.

   Example dependency install on Ubuntu (apt) using serial HDF5:

   .. code-block:: bash

       sudo apt-get update -y
       sudo apt-get install -y python3 python3-pip  # Install python3 and pip
       sudo apt-get install -y libopenmpi-dev       # Install MPI development headers
       sudo apt-get install -y libhdf5-serial-dev   # Install serial HDF5
       sudo apt-get install -y curl wget

       python3 -m pip install h5py mpi4py numpy cython

   Running parallel simulations without a
   MPI-enabled HDF5 library available necessitates the use of the
   :code:`--disable-mpio` argument to HyMD, see :ref:`commandline-label`. Note that
   due to the way HyMD is built, a working MPI compiler is required even if all
   intended simulations are serial.


Installing HyMD
===============
HyMD may be installed using :code:`pip` by

.. code-block:: bash

   python3 -m pip install hymd



Install in docker
=================
A docker image with build essentials setup is available at `dockerhub`_ with tag
:code:`mortele/hymd`,

.. code-block:: bash

   docker pull mortele/hymd:latest
   docker run -it mortele/hymd
   /app$ python3 -m pip install hymd

   # Grab example input files
   /app$ curl -O https://raw.githubusercontent.com/Cascella-Group-UiO/HyMD-tutorial/main/ideal_chain/ideal_chain.toml
   /app$ curl -O https://raw.githubusercontent.com/Cascella-Group-UiO/HyMD-tutorial/main/ideal_chain/ideal_chain.HDF5

   # Run simulation
   /app$ python3 -m hymd ideal_chain.toml ideal_chain.HDF5 --verbose

.. _dockerhub:
   https://hub.docker.com/repository/docker/mortele/hymd


Run interactively in Google Colaboratory
========================================
A `Google Colaboratory`_ jupyter notebook is setup `here`_ with a working HyMD
fully installed and executable in the browser. We do not recommend running
large-scale simulations in colab for pretty obvious reasons.

.. _`Google colaboratory` :
   https://colab.research.google.com/
.. _`here` :
   https://colab.research.google.com/drive/1jfzRaXjL3q53J4U8OrCgADepmf_HuCOh?usp=sharing


Common issues
=============

Numpy errors while importing the Fortran force kernels
------------------------------------------------------

.. code-block:: python

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

.. code-block:: python

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


Error running parallel HyMD without MPI-enabled h5py
----------------------------------------------------

.. code-block:: python

   Traceback (most recent call last):
     File "/usr/local/Cellar/python@3.9/3.9.8/Frameworks/Python.framework/Versions/3.9/lib/python3.9/runpy.py", line 197, in _run_module_as_main
   Traceback (most recent call last):
     File "/usr/local/Cellar/python@3.9/3.9.8/Frameworks/Python.framework/Versions/3.9/lib/python3.9/runpy.py", line 197, in _run_module_as_main
       return _run_code(code, main_globals, None,
     File "/usr/local/Cellar/python@3.9/3.9.8/Frameworks/Python.framework/Versions/3.9/lib/python3.9/runpy.py", line 87, in _run_code
       return _run_code(code, main_globals, None,
     File "/usr/local/Cellar/python@3.9/3.9.8/Frameworks/Python.framework/Versions/3.9/lib/python3.9/runpy.py", line 87, in _run_code
       exec(code, run_globals)
     File "/usr/local/lib/python3.9/site-packages/hymd/__main__.py", line 2, in <module>
       exec(code, run_globals)
     File "/usr/local/lib/python3.9/site-packages/hymd/__main__.py", line 2, in <module>
       main()
     File "/usr/local/lib/python3.9/site-packages/hymd/main.py", line 64, in main
       with h5py.File(args.input, "r", **_kwargs) as in_file:
     File "/usr/local/lib/python3.9/site-packages/h5py/_hl/files.py", line 502, in __init__
       with h5py.File(args.input, "r", **_kwargs) as in_file:
       fapl = make_fapl(driver, libver, rdcc_nslots, rdcc_nbytes, rdcc_w0,
     File "/usr/local/lib/python3.9/site-packages/h5py/_hl/files.py", line 166, in make_fapl
       fapl = make_fapl(driver, libver, rdcc_nslots, rdcc_nbytes, rdcc_w0,
     File "/usr/local/lib/python3.9/site-packages/h5py/_hl/files.py", line 166, in make_fapl
       set_fapl(plist, **kwds)
     File "/usr/local/lib/python3.9/site-packages/h5py/_hl/files.py", line 52, in _set_fapl_mpio
       set_fapl(plist, **kwds)
     File "/usr/local/lib/python3.9/site-packages/h5py/_hl/files.py", line 52, in _set_fapl_mpio
       raise ValueError("h5py was built without MPI support, can't use mpio driver")
   ValueError: h5py was built without MPI support, can't use mpio driver

Can be fixed by installing a MPI-enabled :code:`h5py` through

.. code-block:: bash

   python3 -m pip uninstall -y h5py
   HDF5_MPI="ON" python3 -m pip install --no-binary=h5py h5py
