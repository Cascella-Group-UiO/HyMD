import pytest
from mpi4py import MPI
from hymd.logger import get_version, print_header
from hymd.version import __version__

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