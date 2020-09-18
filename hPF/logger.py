import logging
from mpi4py import MPI


def clog(level, msg, *args, **kwargs):
    if 'comm' in kwargs:
        comm = kwargs.pop('comm', MPI.COMM_WORLD)
        if comm.rank == 0:
            logging.log(level, msg, *args, **kwargs)
    else:
        logging.log(level, msg, *args, **kwargs)
