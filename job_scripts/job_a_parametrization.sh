#!/bin/bash
#SBATCH --job-name=a_parametrisation
#SBATCH --account=nn4654k
#SBATCH --time=0-10:0:0
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4G
# ^For Saga
##  a series of NPT runs with water with varying a parameter

set -o errexit # exit on errors
module load h5py/2.10.0-foss-2020a-Python-3.8.2
module load pfft-python/0.1.21-foss-2020a-Python-3.8.2
set -x


export MPI_NUM_RANKS=${SLURM_NTASKS}
ERR_LOG="srun-${SLURM_JOB_ID}.err"
export OMP_NUM_THREADS=1

# Copy data to /cluster/work/jobs/$SLURM_JOB_ID
export SCRATCH="/cluster/work/users/samiransen23/${SLURM_JOB_ID}"
export HYMD_PATH="/cluster/projects/nn4654k/samiran/HyMD-2021"
export INPUT_PATH="/cluster/projects/nn4654k/samiran/HyMD-2021"
declare -a sigma
sigma=(0.238 0.338 0.438 1.00)
declare -a a
a=(9.15 9.25 9.30)

mkdir ${SCRATCH}
cd ${SCRATCH}
cp ${INPUT_PATH}/examples/water/config.toml ${SCRATCH}/config.toml
cp ${INPUT_PATH}/examples/water/water.h5 ${SCRATCH}/input.h5
mkdir hymd
cp -r ${HYMD_PATH}/hymd/* hymd/.
for ii in ${sigma[*]}
do
    sigmavalue=${ii}
    mkdir -p "sigma="${sigmavalue}
    cd "sigma="${sigmavalue}
    for i in ${a[*]}
    do
        avalue=${i}
        mkdir -p "a="${avalue}
        date
        cp ${SCRATCH}/config.toml "a=${avalue}"/config.toml
        cp ${SCRATCH}/input.h5 "a=${avalue}"/input.h5
        sed -i "s/sigma\s*=\s.*/sigma = ${sigmavalue}/" "a="${avalue}/config.toml
        sed -i "s/^a\s*=\s.*/a = ${avalue}/" "a="${avalue}/config.toml
        cd "a="${avalue}
        srun --exclusive --ntasks ${MPI_NUM_RANKS}            \
             python3 ${SCRATCH}/hymd/main.py config.toml input.h5        \
             --logfile=log.txt --verbose 2 --seed 5           \
             --velocity-out
             #--destdir ${DEST}                               \
             #--double-precision
        cd ..
    done
    cd ..
done



wait
mkdir ${SLURM_SUBMIT_DIR}/out_${SLURM_JOB_ID}
cp -r ${SCRATCH}/* ${SLURM_SUBMIT_DIR}/out_${SLURM_JOB_ID}
