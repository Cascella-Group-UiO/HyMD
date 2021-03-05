Hamiltonian and alias-free hPF-MD &middot; [![License: GPL v3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0.html) ![build](https://github.com/sigbjobo/hPF_MD_PMESH_MPI/workflows/build/badge.svg)
---------
A simple python implementation of hybrid particle-field molecular dynamics (hPF-MD) which allows conservation of energy and reduces aliasing by refinement of the grid.

USE BLACK FORMATTER

Compile FORTRAN modules:
```bash
> cd hymd/
> make clean
> make
> cd ..
```

Run a simple example simulation with (not working right now)
```bash
> mkdir RUN/
> python3 utils/make_input.py examples/CONF.py && mv input.hdf5 RUN/
> mpirun -n 6 python3 hymd/main.py examples/CONF.py RUN/input.hdf5 --destdir=RUN
```

#### Run using Docker
Pull a pre-build docker image with everything setup by `docker pull mortele/hpf:1.1` and then run in a container by
```bash
> git clone git@github.com:sigbjobo/hPF_MD_PMESH_MPI.git hPF-MD
> docker run -it --mount src=$(pwd)/hPF-MD/,target=/app/hPF,type=bind mortele/hpf:1.1
```
```C
root@d6abeeee1a2d:/app# python3 utils/make_input.py examples/CONF.py
root@d6abeeee1a2d:/app# export OMP_NUM_THREADS=1
root@d6abeeee1a2d:/app# export OMPI_MCA_btl_vader_single_copy_mechanism=none
root@d6abeeee1a2d:/app# export NPROCS=4
root@d6abeeee1a2d:/app# mpirun --allow-run-as-root -n ${NPROCS} python3 hPF/main.py CONF.py input.hdf5 --destdir=CONF
```

If you need to build the image yourself, you can do so by
```bash
> git clone git@github.com:sigbjobo/hPF_MD_PMESH_MPI.git hPF-MD
> cd hPF-MD/.docker/hpf/
> docker build --tag hpf:1.1 .
```

### Build
Installing the necessary dependencies requires building `h5py` with MPI support from source.

#### Ubuntu
Building on Ubuntu 20.04 (assumes Python`>=3.7` with pip, and curl are present):
```bash
> apt-get update
> apt-get install libopenmpi-dev libhdf5-openmpi-dev pkg-configx

> pip3 install --upgrade pip
> pip3 install --upgrade numpy mpi4py cython
> pip3 install networkx sympy pytest mpsort pfft-python pmesh

# Build h5py from source with MPI support
> git clone git@github.com:h5py.git
> cd h5py
> git checkout 6f4c578f78321b857da31eee0ce8d9b1ba291888
> HDF5_MPI="ON" pip3 install -v .
```

#### MacOS
Building on OSX Catalina 10.15.3:
```bash
> brew --version
Homebrew 2.4.9
> brew install python@3.8
> python3 --version
Python 3.8.5
> pip3 --version
pip 20.1.1 from usr/local/lib/python3.8/site-packages/pip (python 3.8 )
> brew install open-mpi
> brew install hdf5-mpi
> brew install pgk-config
> pip3 install mpi4py pmesh numpy sympy pfft-python mpsort  cython

# Using LLVM default compiler from Xcode
> mpicc --showme
clang -I/usr/local/Cellar/open-mpi/4.0.4_1/include -L/usr/local/opt/libevent/lib -L/usr/local/Cellar/open-mpi/4.0.4_1/lib -lmpi

# compile h5py from source
> git clone git@github.com:h5py.git
> cd h5py

# master is probably fine, but I used this specific one
# The lastest github release 2.10.0 from Sep 2019 does *not* work
> git checkout 6f4c578f78321b857da31eee0ce8d9b1ba291888
> HDF5_MPI="ON" pip3 install -v .
```
