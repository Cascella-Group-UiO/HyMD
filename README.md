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
> mpirun -n 4 python3 hymd/main.py config.toml dppc.h5 --verbose --logfile log.txt
```

##Installation on Mac OSX  
```bash
for python3.8
    conda create -n py38 python=3.8
#
# To activate this environment, use
#
#     $ conda activate py38
#
# To deactivate an active environment, use
#
#     $ conda deactivate
#
   conda install h5py
   pip3 install mpi4py
   conda install -c bccp pmesh
   conda install -c anaconda intel-openmp
   conda install numba
   pip3 install numpy sympy pfft-python mpsort  cython
   pip install networkx
   conda install -c conda-forge "h5py>=2.9=mpi*"
   pip install h5glance
```


Notes:
- Changed numpy data types (Float32, Int32 -> float32, int32) to run in local computer
- TODO: Fix for running monoatomic (or non molecular) systems
