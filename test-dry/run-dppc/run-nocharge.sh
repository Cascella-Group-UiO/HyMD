#conda activate py38

#PYPATH=/usr/local/opt/python@3.8/bin/
#mpirun -n 4 ${PYPATH}python3.8  hymd/main.py config.toml dppc.h5 --verbose -logfile log.txt  

#mpirun -n 4 python   ../../hymd/main.py config.toml dppc.h5 --verbose --logfile log.txt
mpirun -n 4 python   ../../hymd/main.py config-nocharge.toml dppc.h5 --verbose  --velocity-output  --logfile log-nocharge.txt


