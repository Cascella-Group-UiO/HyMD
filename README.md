Hamiltonian and alias-free hPF-MD testing and development
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
