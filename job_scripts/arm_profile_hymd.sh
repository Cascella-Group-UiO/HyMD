# start interactive job @betzy
srun --job-name="armprof" --nodes=4 --ntasks-per-node=128 --time=0-1:0:0 --account=nn4654k --pty bash -i

# arm-map profile module
module load Arm-Forge/20.1.2
module load h5py/2.10.0-foss-2020a-Python-3.8.2
module load pfft-python/0.1.21-foss-2020a-Python-3.8.2

# run arm-map profile on hymd
map -profile  mpirun -n 32  python3 hymd/main.py $1 $2 --logfile=log_profile.txt --verbose 2 --velocity-output --destdir=RUN_profile --seed 1
>>> Arm Forge 20.1.2 - Arm MAP
>>> 
>>> MAP: Your Arm MAP support for the licence 10168 expired on Fri Mar 26 2021.
>>> MAP: Please e-mail HPCToolsSales@arm.com for renewal options.
>>> Profiling             : mpirun -n 64 python3 HyMD-2021/hymd/main.py config.toml dppc_20x20nm.h5 --logfile=log6.txt --verbose 2 --velocity-output --destdir=RUN6 --seed 1
>>> Allinea sampler       : preload (Express Launch)
>>> MPI implementation    : Auto-Detect (Open MPI)
>>> * number of processes : 64
>>> ....
>>> ....
>>> .... normal program output
>>> ....
>>> ....
>>> MAP analysing program...
>>> MAP gathering samples...
>>> MAP generated /cluster/work/users/mortele/interactive/python3_main_py_64p_1n_1t_2021-06-24_10-39.map

# arm-map show result 
map python3_main_py_32p_1n_1t_2021-06-23_17-26.map
