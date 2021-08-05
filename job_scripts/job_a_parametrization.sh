#for local computer
#  a series of NPT runs with water with varying a parameter

ERR_LOG="srun-${SLURM_JOB_ID}.err"
LOG_FILE="log.txt"
export OMP_NUM_THREADS=1
export DEST_DIR="/Users/samiransen23/hymdtest/test_aparametrization"
export HYMD_PATH="/Users/samiransen23/hymdtest/hymd/"
export INPUT_PATH="/Users/samiransen23/hymdtest/"

declare -a a
a=(9.15 9.25 9.30)
mkdir -p ${DEST_DIR}

for i in ${a[*]}
do
    echo $i
done
for i in ${a[*]}
do
    avalue=${i}
    mkdir -p ${DEST_DIR}/"a="${avalue}
    date
    cp ${INPUT_PATH}/examples/water/config.toml ${DEST_DIR}/"a=${avalue}"/config.toml
    gsed -i "s/^a\s*=\s.*/a = ${avalue}/g" ${DEST_DIR}/"a="${avalue}/config.toml
    mpirun -n 4 python3 hymd/main.py ${DEST_DIR}/"a=${avalue}"/config.toml ${INPUT_PATH}/examples/water/water.h5 --verbose --logfile ${LOG_FILE} --destdir ${DEST_DIR}/"a=${avalue}"
done
