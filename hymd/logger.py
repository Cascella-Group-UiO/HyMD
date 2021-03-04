import sys
import logging
from mpi4py import MPI


class MPIFilterRoot(logging.Filter):
    def filter(self, record):
        if record.funcName == '<module>':
            record.funcName = 'main'
        if MPI.COMM_WORLD.Get_rank() == 0:
            record.rank = MPI.COMM_WORLD.Get_rank()
            record.size = MPI.COMM_WORLD.Get_size()
            return True
        else:
            return False


class MPIFilterAll(logging.Filter):
    def filter(self, record):
        if record.funcName == '<module>':
            record.funcName = 'main'
        record.rank = MPI.COMM_WORLD.Get_rank()
        record.size = MPI.COMM_WORLD.Get_size()
        return True


class Logger:
    level = None
    log_file = None
    format = ' %(levelname)-8s [%(filename)s:%(lineno)d] <%(funcName)s> {rank %(rank)d/%(size)d} %(message)s'  # noqa: E501
    date_format = '%(asctime)s'
    formatter = logging.Formatter(fmt=date_format + format)
    rank0 = logging.getLogger('HyMD.rank_0')
    all_ranks = logging.getLogger('HyMD.all_ranks')

    @classmethod
    def setup(cls, default_level=logging.INFO, log_file=None,
              verbose=False):
        cls.level = default_level
        cls.log_file = log_file

        log_to_stdout = False
        if verbose:
            log_to_stdout = True

        level = default_level
        if not verbose:
            level = logging.WARNING
        cls.rank0.setLevel(level)
        cls.all_ranks.setLevel(level)

        cls.rank0.addFilter(MPIFilterRoot())
        cls.all_ranks.addFilter(MPIFilterAll())
        
        # Why don't add an else at the end instead of this?
        if (not log_file) and (not log_to_stdout):
            return
        if log_file:
            cls.log_file_handler = logging.FileHandler(log_file)
            cls.log_file_handler.setLevel(level)
            cls.log_file_handler.setFormatter(cls.formatter)

            cls.rank0.addHandler(cls.log_file_handler)
            cls.all_ranks.addHandler(cls.log_file_handler)

        if log_to_stdout:
            cls.log_to_stdout = True
            cls.stdout_handler = logging.StreamHandler()
            cls.stdout_handler.setLevel(level)
            cls.stdout_handler.setStream(sys.stdout)
            cls.stdout_handler.setFormatter(cls.formatter)

            cls.rank0.addHandler(cls.stdout_handler)
            cls.all_ranks.addHandler(cls.stdout_handler)
