<a href="https://cascella-group-uio.github.io/HyMD/">
  <img src="docs/img/hymd_logo_text_black.png" width="500" title="HylleraasMD">
</a>

[![License: GPL v3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0.html) ![build](https://github.com/Cascella-Group-UiO/HyMD-2021/workflows/build/badge.svg) [![docs](https://github.com/Cascella-Group-UiO/HyMD/actions/workflows/docs_pages.yml/badge.svg)](https://github.com/Cascella-Group-UiO/HyMD/actions/workflows/docs_pages.yml) [![codecov](https://codecov.io/gh/Cascella-Group-UiO/HyMD/branch/main/graph/badge.svg?token=BXZ7B9RXV9)](https://codecov.io/gh/Cascella-Group-UiO/HyMD) [![PyPI version](https://badge.fury.io/py/hymd.svg)](https://badge.fury.io/py/hymd)

---------
**HylleraasMD** (HyMD) is a massively parallel Python package for hybrid particle-field molecular dynamics (hPF-MD) simulations of coarse-grained bio- and soft-matter systems.

HyMD can run canonical hPF-MD simulations [[1]](#1), or filtered density Hamiltonian hPF (HhPF-MD) simulations [[2]](#2), with or without explicit PME electrostatic interactions [[3]](#3). It includes all standard intramolecular interactions, including stretching, bending, torsional, and combined bending-dihedral potentials. Additionally, topological reconstruction of permanent peptide chain backbone dipoles is possible for accurate recreation of protein conformational dynamics [[4]](#4). Martini style elastic networks (ElNeDyn) [[5]](#5) are also supported.

HyMD uses the [pmesh](github.com/rainwoodman/pmesh) library for particle-mesh operations, with the PPFT [[6]](#6) backend for FFTs through the [pfft-python bindings](github.com/rainwoodman/pfft-python). File IO is done via HDF5 formats to allow MPI parallel reads.

## User Guide
Detailed installation and user guide, together with comprehensive example simulations are located in the [HylleraasMD documentation](https://cascella-group-uio.github.io/HyMD/index.html).

Run simulations by
```bash
python3 -m hymd [CONFIGURATION_FILE] [TOPOLOGY_FILE]
```

## Installation
Install HyMD with `pip` by
```bash
python3 -m pip install hymd
```
#### Install dependencies
HyMD installation **requires** a working MPI compiler. It is highly recommended to have *MPI-enabled* HDF5 and [h5py](https://docs.h5py.org/en/stable/mpi.html) for running parallel simulations with HyMD. Install both on Ubuntu with
```bash
sudo apt-get update -y
sudo apt-get install -y curl pkg-config python3-pip python3-mpi4py libhdf5-mpi-dev
export CC=mpicc
export HDF5_MPI="ON"
python3 -m pip install --no-binary=h5py h5py
```

#### Run in docker
Alternatively, an up-to-date docker image is available from [docker hub](https://hub.docker.com/repository/docker/mortele/hymd)
```bash
docker pull mortele/hymd
docker run -it mortele/hymd
/app# apt-get install -y git
/app# git clone https://github.com/Cascella-Group-UiO/HyMD.git hymd
/app# cd hymd/
/app/hymd# pip3 install .
/app/hymd# pytest
```

## Run tests
Clone the repository and run tests with [pytest](https://docs.pytest.org/en/latest)
```bash
git clone https://github.com/Cascella-Group-UiO/HyMD.git hymd
cd hymd
pytest
```
Running MPI enabled pytest tests is simplified with a convenient script
```bash
chmod +x pytest-mpi
pytest-mpi -oo -n 2 -ns
```

---------

### References
<a id="1">[1]</a>
Milano, G.; Kawakatsu, T. Hybrid particle-field molecular dynamics simulations for densepolymer systems. J. Chem. Phys. **2009**, 130, 214106.

<a id="2">[2]</a>
Bore, S. L.; Cascella, M. Hamiltonian and alias-free hybrid particle–field molecular dynam-ics. J. Chem. Phys. **2020**, 153, 094106.

<a id="3">[3]</a>
Kolli, H. B.; De Nicola, A.; Bore, S. L.; Schäfer, K.; Diezemann, G.; Gauss, J.; Kawakatsu, T.;Lu, Z.-Y.; Zhu, Y.-L.; Milano, G.; Cascella, M. Hybrid Particle-Field Molecular DynamicsSimulations of Charged Amphiphiles in an Aqueous Environment. J. Chem. Theory Comput. **2018**, 14, 4928–4937.

<a id="4">[4]</a>
Bore, S. L.; Milano, G.; Cascella, M. Hybrid Particle-Field Model for Conformational Dy-namics of Peptide Chains. J. Chem. Theory Comput. **2018**, 14, 1120–1130.

<a id="5">[5]</a>
Periole, X.; Cavalli, M.; Marrink, S. J.; Ceruso, M. A. Combining an elastic network with a coarse-grained molecular force field: structure, dynamics, and intermolecular recognition. J. Chem. Theory Comput. **2009**, 5.9, 2531-2543.

<a id="6">[6]</a>
Pippig, M. PFFT: An extension of FFTW to massively parallel architectures. SIAM J. Sci. Comput. **2013**, 35, C213–C236.
