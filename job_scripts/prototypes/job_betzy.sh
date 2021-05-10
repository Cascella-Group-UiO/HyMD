#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=nn4654k
#SBATCH --time=0-1:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --qos=devel


set -o errexit # exit on errors
module load h5py/2.10.0-foss-2020a-Python-3.8.2
module load pfft-python/0.1.21-foss-2020a-Python-3.8.2
set -x


export MPI_NUM_RANKS=32


# Copy data to /cluster/work/jobs/$SLURM_JOB_ID
export SCRATCH="/cluster/work/users/samiransen23/${SLURM_JOB_ID}"
export HYMD_PATH=${PWD}

mkdir ${SCRATCH}
cd ${SCRATCH}
#mkdir ${SCRATCH}/utils/
cp $1 ${SCRATCH}/config.toml
cp $2 ${SCRATCH}/input.h5
mkdir hymd
cp -r ${HYMD_PATH}/hymd/* hymd/.
#cp ${HYMD_PATH}/utils/*.* utils/

date
srun --exclusive --ntasks ${MPI_NUM_RANKS}            \
     python3 hymd/main.py config.toml input.h5        \
     --logfile=log.txt --verbose 2 --seed 5                                         
     #--destdir ${DEST}                               \
     #--double-precision


wait
mkdir ${SLURM_SUBMIT_DIR}/out_${SLURM_JOB_ID}
cp -r ${SCRATCH}/* ${SLURM_SUBMIT_DIR}/out_${SLURM_JOB_ID}
cp ${SCRATCH}/{config.toml,input.h5} ${SLURM_SUBMIT_DIR}/out_${SLURM_JOB_ID}/. 
