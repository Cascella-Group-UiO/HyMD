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
For python3.8
```bash
    conda create -n py38 python=3.8
#
# To activate this environment, use
#     $ conda activate py38
# To deactivate an active environment, use
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
The following is a typical way to run a job script on SAGA:
```bash
> module restore
> module load h5py/2.10.0-foss-2020a-Python-3.8.2
> module load pfft-python/0.1.21-foss-2020a-Python-3.8.2
> git clone https://github.com/Cascella-Group-UiO/HyMD-2021.git
> export HYMD_PATH ="/cluster/projects/nn4654k/samiran/HyMD-2021"
> cd HyMD-2021/hymd
> make
> cd ..
> ls job_scripts
job_binary.sh  job.sh  job_test.sh  job_water.sh
sbatch job_scripts/job_binary.sh $HYMD_PATH/examples/binary/configAB.toml $HYMD_PATH/examples/binary/binary_eq.h5
```
`HYMD_PATH`is just a variable to make the `sbatch` line cleaner, nothing else.  
### OUTPUT  
A folder called `out_XXXXXXX` is created in the slurm submit dir (in this case, at the location of `HyMD-2021`), where `XXXXXXX` is the job ID.
```bash
> ls out_2717351/
config.toml  hymd  input.h5  log.txt  sim.h5  srun-2717351.err
```
See that the files copied include: inputs (`config.toml`,`input.h5`), a log file (`log.txt`) and the output trajectory (`sim.h5`)

## Other README.md 's in this repository
How to build a binary system -- [examples/binary/README.md](https://github.com/Cascella-Group-UiO/HyMD-2021/tree/pressure/examples/binary)

Notes:
- Changed numpy data types (Float32, Int32 -> float32, int32) to run in local computer
