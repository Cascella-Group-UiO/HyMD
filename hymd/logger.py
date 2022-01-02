"""Handles event logging for simulation information, warnings, and errors.
"""
import sys
import logging
from mpi4py import MPI


class MPIFilterRoot(logging.Filter):
    """Log output Filter wrapper class for the root MPI rank log
    """
    def filter(self, record):
        """Log event message filter

        Parameters
        ----------
        record : logging.LogRecord
            LogRecord object corresponding to the log event.
        """
        if record.funcName == "<module>":
            record.funcName = "main"
        if MPI.COMM_WORLD.Get_rank() == 0:
            record.rank = MPI.COMM_WORLD.Get_rank()
            record.size = MPI.COMM_WORLD.Get_size()
            return True
        else:
            return False


class MPIFilterAll(logging.Filter):
    """Log output Filter wrapper class for the all-MPI-ranks log
    """
    def filter(self, record):
        """Log event message filter

        Parameters
        ----------
        record : logging.LogRecord
            LogRecord object corresponding to the log event.
        """
        if record.funcName == "<module>":
            record.funcName = "main"
        record.rank = MPI.COMM_WORLD.Get_rank()
        record.size = MPI.COMM_WORLD.Get_size()
        return True


class Logger:
    """Log output handler class

    Notes
    -----
    This wraps the default python library :code:`logging`, see
    `docs.python.org/3/library/logging.html`_.

    .. _`docs.python.org/3/library/logging.html`:
        https://docs.python.org/3/library/logging.html

    Attributes
    ----------
    level : int
        Determines the verbosity level of the log, corresponding to
        logging.level. Numerical values :code:`50` (:code:`logging.CRITICAL`),
        :code:`40` (:code:`logging.ERROR`), :code:`30`
        (:code:`logging.WARNING`), :code:`20` (:code:`logging.INFO`),
        :code:`10` (:code:`logging.DEBUG`), and :code:`0`
        (:code:`logging.UNSET`) are supported values. Any log event message
        less severe than the specified level is ignored. All other event
        messages are emitted.
    log_file : str
        Path to output log file.
    format : str
        Prepended dump string for each log event. Specifies the log event
        level, the module emitting the event, the code line, the enclosing
        function name, and the MPI rank writing the message.
    date_format : str
        Prepends the date before all log event messages.
    formatter : logging.Formatter
        Formatter handling the prepending of the information in `format` and
        `date_format` to each log event message. Used by default for all
        loggers.
    rank0 : logging.Logger
        Default logger object for the root MPI rank.
    all_ranks : logging.Logger
        Default logger object for messages being emitted from all MPI ranks
        simultaneously.
    """
    level = None
    log_file = None
    format = " %(levelname)-8s [%(filename)s:%(lineno)d] <%(funcName)s> {rank %(rank)d/%(size)d} %(message)s"  # noqa: E501
    date_format = "%(asctime)s"
    formatter = logging.Formatter(fmt=date_format + format)
    rank0 = logging.getLogger("HyMD.rank_0")
    all_ranks = logging.getLogger("HyMD.all_ranks")

    @classmethod
    def setup(cls, default_level=logging.INFO, log_file=None, verbose=False):
        """Sets up the logger object.

        If a :code:`log_file` path is provided, log event messages are output
        to it. Otherwise, the logging messages are emitted to stdout.

        Parameters
        ----------
        default_level : int, optional
            Default verbosity level of the logger. Unless specified, it is
            :code:`10` (:code:`logging.INFO`).
        log_file : str, optional
            Path to output log file. If `None` or not priovided, no log file is
            used and all logging is done to stdout.
        verbose : bool, optional
            Increases the logging level to :code:`30` (:code:`logging.WARNING`)
            if True, otherwise leaves the logging level unchanged.
        """
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


def format_timedelta(timedelta):
    days = timedelta.days
    hours, rem = divmod(timedelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    microseconds = timedelta.microseconds
    ret_str = ""
    if days != 0:
        ret_str += f"{days} days "
    ret_str += f"{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:06d}"
    return ret_str
