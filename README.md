<a href="https://cascella-group-uio.github.io/HyMD/">
  <img src="https://github.com/Cascella-Group-UiO/HyMD/blob/main/docs/img/hymd_logo_text_black.png?raw=true" width="500" title="HylleraasMD">
</a>

[![License: GPL v3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0.html) ![build](https://github.com/Cascella-Group-UiO/HyMD-2021/workflows/build/badge.svg) [![docs](https://github.com/Cascella-Group-UiO/HyMD/actions/workflows/docs_pages.yml/badge.svg)](https://cascella-group-uio.github.io/HyMD/) [![codecov](https://codecov.io/gh/Cascella-Group-UiO/HyMD/branch/main/graph/badge.svg?token=BXZ7B9RXV9)](https://codecov.io/gh/Cascella-Group-UiO/HyMD) [![PyPI version](https://badge.fury.io/py/hymd.svg)](https://badge.fury.io/py/hymd)

---------
**HylleraasMD** (HyMD) is a massively parallel Python package for Hamiltonian hybrid particle-field molecular dynamics (HhPF-MD) simulations of coarse-grained bio- and soft-matter systems.

HyMD can run canonical hPF-MD simulations, or filtered density Hamiltonian hPF (HhPF-MD) simulations [[1]](#1),[[2]](#2),[[3]](#3) with or without explicit PME electrostatic interactions. It includes all standard intramolecular interactions,
including stretching, bending, torsional, and combined bending-dihedral potentials. Additionally, topological reconstruction of permanent peptide chain backbone dipoles is possible for accurate recreation of protein conformational dynamics.
It can run simulations in constant energy (NVE), constant volume (NVT) [[1]](#1) or constant pressure (NPT) conditions [[4]](#4).

HyMD uses the [pmesh](github.com/rainwoodman/pmesh) library for particle-mesh operations, with the PPFT [[5]](#5) backend for FFTs through the [pfft-python bindings](github.com/rainwoodman/pfft-python).
File IO is done via HDF5 formats to allow MPI parallel reads.

## User Guide
Detailed installation and user guide, together with comprehensive example simulations are located in the [HylleraasMD documentation](https://cascella-group-uio.github.io/HyMD/index.html).

Run simulations by
```bash
python3 -m hymd [CONFIGURATION_FILE] [TOPOLOGY_FILE] (--OPTIONAL_ARGS)
```

#### Run interactively in Google Colaboratory
A [Google Colaboratory](https://colab.research.google.com/) jupyter notebook is setup [here](https://colab.research.google.com/drive/1jfzRaXjL3q53J4U8OrCgADepmf_HuCOh?usp=sharing) with a working HyMD fully installed and executable in the browser.

## Installation

#### Non-Python dependencies
HyMD installation **requires** a working MPI compiler. It is highly recommended to have *MPI-enabled* HDF5 and [h5py](https://docs.h5py.org/en/stable/mpi.html) for running parallel simulations with HyMD. Install both on Ubuntu with
```bash
sudo apt-get update -y
sudo apt-get install -y pkg-config libhdf5-mpi-dev libopenmpi-dev
python3 -m pip uninstall h5py  # Remove any serial h5py installation present
CC="mpicc" HDF5_MPI="ON" python3 -m pip install --no-binary=h5py h5py
```

> **Note**
> There might be memory leaks if you use OpenMPI <= 4.1.1. See [#186](https://github.com/Cascella-Group-UiO/HyMD/issues/186) for more details.

#### Python dependencies
Install HyMD with `pip` by
```bash
python3 -m pip install --upgrade numpy mpi4py cython
python3 -m pip install hymd
```
See [HyMD docs](https://cascella-group-uio.github.io/HyMD/doc_pages/installation.html) for more information, including install steps for macOS and non-Debian linux distributions.

#### Run in docker
Alternatively, an up-to-date docker image is available from [docker hub](https://hub.docker.com/repository/docker/mortele/hymd)
```bash
docker pull mortele/hymd
docker run -it mortele/hymd
/app$ python3 -m pip install hymd
/app$
/app$ # Grab example input files
/app$ curl -O https://raw.githubusercontent.com/Cascella-Group-UiO/HyMD-tutorial/main/ideal_chain/ideal_chain.toml
/app$ curl -O https://raw.githubusercontent.com/Cascella-Group-UiO/HyMD-tutorial/main/ideal_chain/ideal_chain.HDF5
/app$
/app$ # Run simulation
/app$ python3 -m hymd ideal_chain.toml ideal_chain.HDF5 --verbose
```

## Run tests
Clone the repository and run tests with [pytest](https://docs.pytest.org/en/latest)
```bash
git clone https://github.com/Cascella-Group-UiO/HyMD.git hymd
cd hymd
python3 -m pip install pytest pytest-mpi
pytest
```
Running MPI enabled pytest tests is simplified with a convenient script
```bash
chmod +x pytest-mpi
pytest-mpi -oo -n 2 -ns
```

## Please cite our work
If you use HyMD for your purposes, please cite the appropriate references from the section below.  
If you cannot cite all, the fundamental works to be cited are [[1]](#1) and [[4]](#4). 

---------

### References
<a id="1">[1]</a>
Ledum, M.; Sen, S.; Li, X.; Carrer, M.; Feng Y.; Cascella, M.; Bore, S. L. HylleraasMD: A Domain Decomposition-Based Hybrid Particle-Field Software for Multi-Scale Simulations of Soft Matter. ChemRxiv 2021

<a id="2">[2]</a>
Ledum, M.; Carrer, M.; Sen, S.; Li, X.; Cascella, M.; Bore, S. L. HyMD: Massively parallel hybrid particle-field molecular dynamics in Python.

<a id="3">[3]</a>
Bore, S. L.; Cascella, M. Hamiltonian and alias-free hybrid particle–field molecular dynamics. J. Chem. Phys. 2020, 153, 094106.

<a id="4">[4]</a>
Sen, S.; Ledum, M.; Bore, S. L.; Cascella, M. Soft Matter under Pressure: Pushing Particle–Field Molecular Dynamics to the Isobaric Ensemble. ChemRxiv 2023

<a id="5">[5]</a>
Pippig, M. PFFT: An extension of FFTW to massively parallel architectures. SIAM J. Sci. Comput. 2013, 35, C213–C236.
