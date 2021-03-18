HyMD testing and development &middot; [![License: GPL v3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0.html) ![build](https://github.com/sigbjobo/hPF_MD_PMESH_MPI/workflows/build/badge.svg)
---------
Compile FORTRAN modules:
```bash
> cd hymd/
> make clean
> make
> cd ..
```

Run a simple example simulation with
```bash
> mpirun -n 4 python3 hymd/main.py config.toml dppc.h5 --verbose -logfile log.txt
```

Notes:
- Changed numpy data types (Float32, Int32 -> float32, int32) to run in local computer
- TODO: Fix for running monoatomic (or non molecular) systems
