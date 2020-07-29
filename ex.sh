#Example shell script of how to run the code

python3 make_input.py CONF.py
mpirun python3 main.py CONF.py input.hdf5

#python3 main.py CONF_BINARY.py
