#!/bin/bash
#SBATCH --job-name=jol_test
#SBATCH --account=nn4654k
#SBATCH --time=0-2:0:0
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
mkdir ${SCRATCH}
cd ${SCRATCH}
cp ${SLURM_SUBMIT_DIR}/config_water.toml $SCRATCH/config.toml
cp ${SLURM_SUBMIT_DIR}/water.h5 $SCRATCH/input.h5
mkdir hymd/
cp /cluster/home/samiransen23/HyMD-2021/hymd/* hymd/

date
srun -K --error=${ERR_LOG} -n ${SLURM_NTASKS} --mpi=pmi2  python3 hymd/main.py config.toml input.h5 --logfile=${LOG_FILE}  --seed 1 --verbose 2

mkdir ${SLURM_SUBMIT_DIR}/out
cp -r ${SCRATCH}/* ${SLURM_SUBMIT_DIR}/out/

