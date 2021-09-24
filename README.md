HyMD testing and development &middot; [![License: GPL v3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0.html) ![build](https://github.com/Cascella-Group-UiO/HyMD-2021/workflows/build/badge.svg)
---------
HyMD is a software that can run coarse-grained molecular dynamics simulations
using the hybrid-particle field model approach, introduced initially in [1].
In HyMD we implement the formulation presented in [2].

## Setup
First compile the FORTRAN modules:
```bash
> cd hymd/
> make clean
> make
> cd ..
```

Check out all the available options with
```bash
> python3 hymd/main.py --help
```

Start a simple example simulation with:
```bash
> mpirun -n 4 python3 hymd/main.py peptide.toml peptide.h5 -v
```

## References
[1] Milano, G. & Kawakatsu, T. Hybrid particle-field molecular dynamics
simulations for dense polymer systems Journal of Chemical Physics, American
Institute of Physics, 2009, 130, 214106 

[2] Bore, S. L. & Cascella, M. Hamiltonian and alias-free hybrid
particle--field molecular dynamics The Journal of Chemical Physics, AIP
Publishing LLC, 2020, 153, 094106 




