#!/bin/bash
#SBATCH --job-name=duibhai_test
#SBATCH --account=nn4654k
#SBATCH --time=0-1:0:0
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=2G
#SBATCH --qos=devel

set -o errexit # exit on errors
module load h5py/2.10.0-foss-2020a-Python-3.8.2
module load pfft-python/0.1.21-foss-2020a-Python-3.8.2
set -x

ERR_LOG="srun-${SLURM_JOB_ID}.err"
LOG_FILE="log.txt"
export OMP_NUM_THREADS=1
export SCRATCH="/cluster/work/users/samiransen23/${SLURM_JOB_ID}"
export HYMD_PATH=${PWD}

mkdir ${SCRATCH}
cd ${SCRATCH}
cp $1 ${SCRATCH}/config.toml
cp $2 ${SCRATCH}/input.h5
mkdir hymd/
cp ${HYMD_PATH}/hymd/* hymd/

date
#Do not change the actual run statement for clarity
srun -K --error=${ERR_LOG} -n ${SLURM_NTASKS} --mpi=pmi2  python3 hymd/main.py config.toml input.h5 --logfile=${LOG_FILE}  --seed 1 --verbose 2

mkdir ${SLURM_SUBMIT_DIR}/out_${SLURM_JOB_ID}
cp -r ${SCRATCH}/* ${SLURM_SUBMIT_DIR}/out_${SLURM_JOB_ID}
cp ${SCRATCH}/{config.toml,input.h5} ${SLURM_SUBMIT_DIR}/out_${SLURM_JOB_ID}/. 

