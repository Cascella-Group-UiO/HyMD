import sys
import logging
from mpi4py import MPI


class Logger:
    def __init__(self, module_root, level, log_file=None, log_to_stdout=False):
        self.module_root = module_root
        self.level = level
        self.format = '%(levelname)-8s [%(filename)s:%(lineno)d] <%(funcName)s> %(message)s'  # noqa: E501
        self.date_format = '%(asctime)s,%(msecs)d'
        self.log_file = log_file
        self.log_to_stdout = log_to_stdout

        self.formatter = logging.Formatter(
            fmt=self.format,
            datefmt=self.date_format
        )
        self.logger = logging.getLogger(module_root)
        self.logger.setLevel(level)

        if (not log_file) and (not log_to_stdout):
            return
        if log_file:
            self.log_file_handler = logging.FileHandler(log_file, vel=level)
            self.log_file_handler.setLevel(level)
            self.log_file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.log_file_handler)
        if log_to_stdout:
            self.log_to_stdout = True
            self.stdout_handler = logging.StreamHandler()
            self.stdout_handler.setLevel(level)
            self.stdout_handler.setStream(sys.stdout)
            self.stdout_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.stdout_handler)

    def __call__(self, level, msg, *args, **kwargs):
        if 'comm' in kwargs:
            comm = kwargs.pop('comm', MPI.COMM_WORLD)
            if comm.rank == 0:
                self.logger.log(level, msg, *args, **kwargs)
        else:
            self.logger.log(level, msg, *args, **kwargs)


def clog(logger, level, msg, *args, **kwargs):
    logger(level, msg, *args, **kwargs)
