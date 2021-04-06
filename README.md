## Hamiltonian and alias-free hPF-MD testing and development
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

## Installation on Mac OSX  
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

## Running HyMD on a Cluster
The following is the way that worked for me on Saga:
```bash
module restore
module load h5py/2.10.0-foss-2020a-Python-3.8.2
module load pfft-python/0.1.21-foss-2020a-Python-3.8.2
git clone https://github.com/Cascella-Group-UiO/HyMD-2021.git hymdruns
cd hymdruns/hymd
make
cd *folder where you have 1) job script (eg: job_binary.sh), 2) input (eg: binaryAB.h5, configAB.toml)*
sbatch job_binary.sh
```


## Other README.md 's in this repository
How to build a binary system -- [examples/binary/README.md](https://github.com/Cascella-Group-UiO/HyMD-2021/tree/pressure/examples/binary)

Notes:
- Changed numpy data types (Float32, Int32 -> float32, int32) to run in local computer
- TODO: Fix for running monoatomic (or non molecular) systems
