import pytest
import logging
import os
from mpi4py import MPI
from hymd.logger import Logger, get_version, print_header
from hymd.version import __version__

@pytest.mark.mpi()
def test_logger(tmp_path):
    logname = os.path.join(f"{tmp_path}","hymd_test.log")

    # broadcast filename to all ranks
    logname = MPI.COMM_WORLD.bcast(logname, root=0)

    # Setup logger
    Logger.setup(
        default_level=logging.INFO,
        log_file=logname,
        verbose=1
    )

    MPI.COMM_WORLD.Barrier()

    assert Logger.log_file == logname

    # test
    Logger.rank0.log(logging.INFO, "TEST INF RANK0")
    MPI.COMM_WORLD.Barrier()
    with open(logname, 'r') as f:
        logcontent = f.readlines()
        assert " INFO " in logcontent[-1]
        assert "test_logger.py" in logcontent[-1]
        assert "<test_logger>" in logcontent[-1]
        assert "rank 0" in logcontent[-1]
        assert "TEST INF RANK0"in logcontent[-1]

    Logger.all_ranks.log(logging.INFO, "TEST INF ALL_RANKS")
    MPI.COMM_WORLD.Barrier()
    with open(logname, 'r') as f:
        logcontent = f.readlines()
        assert "TEST INF ALL_RANKS" in logcontent[-1]

    Logger.rank0.log(logging.WARNING, "TEST WARN RANK0")
    MPI.COMM_WORLD.Barrier()
    with open(logname, 'r') as f:
        logcontent = f.readlines()
        assert " WARNING " in logcontent[-1]
        assert "TEST WARN RANK0" in logcontent[-1]

    Logger.rank0.log(logging.ERROR, "TEST ERR RANK0")
    MPI.COMM_WORLD.Barrier()
    with open(logname, 'r') as f:
        logcontent = f.readlines()
        assert " ERROR " in logcontent[-1]
        assert "TEST ERR RANK0" in logcontent[-1]


def test_version():
    try:
      import git
      foundgit = True
    except:
      foundgit = False

    version = get_version()

    if foundgit:
        assert 'branch commit' in version
    
    assert __version__ in version


def test_header():
    header = print_header()

    assert __version__ in header
