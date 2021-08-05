#!/bin/bash

##SBATCH --job-name=dppc_npt_profile
##SBATCH --account=nn4654k
##SBATCH --time=1-0:0:0
##SBATCH --nodes=4
##SBATCH --ntasks-per-node=128
## ^For Betzy

#SBATCH --job-name=dppc_npt_profile
#SBATCH --account=nn4654k
#SBATCH --time=0-1:0:0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4G
#SBATCH --qos=devel
# ^For Saga

set -o errexit # exit on errors
# arm-map profile module
#module load Arm-Forge/20.1.2 
#^ For Betzy
module load Arm-Forge/21.0
#^For Saga
module load h5py/2.10.0-foss-2020a-Python-3.8.2
module load pfft-python/0.1.21-foss-2020a-Python-3.8.2
set -x


export MPI_NUM_RANKS=32
ERR_LOG="srun-${SLURM_JOB_ID}.err"


# Copy data to /cluster/work/jobs/$SLURM_JOB_ID
export SCRATCH="/cluster/work/users/samiransen23/${SLURM_JOB_ID}"
export HYMD_PATH=${PWD}

mkdir ${SCRATCH}
cd ${SCRATCH}
#mkdir ${SCRATCH}/utils/
cp ${HYMD_PATH}/$1 ${SCRATCH}/config.toml
cp ${HYMD_PATH}/$2 ${SCRATCH}/input.h5
mkdir hymd
cp -r ${HYMD_PATH}/hymd/* hymd/.
#cp ${HYMD_PATH}/utils/*.* utils/

date
# run arm-map profile on hymd
#ddt mpiexec                       \
map -profile mpirun -n ${MPI_NUM_RANKS}        \
python3 hymd/main.py config.toml input.h5       \
--logfile=log.txt --verbose 2 \
--destdir=RUN_profile --seed 5

wait
mkdir ${SLURM_SUBMIT_DIR}/out_${SLURM_JOB_ID}
cp -r ${SCRATCH}/* ${SLURM_SUBMIT_DIR}/out_${SLURM_JOB_ID}
cp ${SCRATCH}/{config.toml,input.h5} ${SLURM_SUBMIT_DIR}/out_${SLURM_JOB_ID}/. 
