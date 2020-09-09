#Example shell script of how to run the code

python3 make_input.py CONF.py
export OMP_NUM_THREADS=1
mpirun -n 2 python3 main.py CONF.py input.hdf5 --destdir=CONF --profile --verbose

#python3 main.py CONF_BINARY.py
