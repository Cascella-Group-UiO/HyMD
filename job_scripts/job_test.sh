#!/bin/bash
#SBATCH --job-name=test_run
#SBATCH --account=nn4654k
#SBATCH --time=0-1:0:0
#SBATCH --ntasks=4
#SBATCH --qos=devel
# ^For Betzy

set -o errexit # exit on errors
module load h5py/2.10.0-foss-2020a-Python-3.8.2
module load pfft-python/0.1.21-foss-2020a-Python-3.8.2
set -x

export MPI_NUM_RANKS=32
ERR_LOG="srun-${SLURM_JOB_ID}.err"

export SCRATCH="/cluster/work/users/samiransen23/${SLURM_JOB_ID}"
export HYMD_PATH=${PWD}

mkdir ${SCRATCH}
cd ${SCRATCH}

cp ${HYMD_PATH}/$1 ${SCRATCH}/config.toml
cp ${HYMD_PATH}/$2 ${SCRATCH}/input.h5
mkdir hymd
cp -r ${HYMD_PATH}/hymd/* hymd/.


date
#srun -K --error=${ERR_LOG} -n ${SLURM_NTASKS} --mpi=pmi2  python3 hymd/main.py config.toml input.h5 --logfile=${LOG_FILE}  --seed 1 --verbose 2


srun --exclusive --ntasks ${MPI_NUM_RANKS}            \
     python3 hymd/main.py config.toml input.h5        \
     --logfile=log.txt --verbose 2 --seed 5           \
     --velocity-out                                         
     #--destdir ${DEST}                               \
     #--double-precision

wait
mkdir ${SLURM_SUBMIT_DIR}/out_test_${SLURM_JOB_ID}
cp -r ${SCRATCH}/* ${SLURM_SUBMIT_DIR}/out__test_${SLURM_JOB_ID}
cp ${SCRATCH}/{config.toml,input.h5} ${SLURM_SUBMIT_DIR}/out__test_${SLURM_JOB_ID}/. 

