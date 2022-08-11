import pytest
import logging
import os
from mpi4py import MPI
from argparse import Namespace
from hymd.configure_runtime import configure_runtime

def test_configure_runtime(h5py_molecules_file, config_toml,
                           change_tmp_dir, caplog):
    caplog.set_level(logging.ERROR)
    comm = MPI.COMM_WORLD
    config_toml_file, config_toml_str = config_toml

    basearg = [config_toml_file, h5py_molecules_file]

    parsed, _ = configure_runtime(basearg, comm)
    assert isinstance(parsed, Namespace)
    assert parsed.destdir == "."
    assert parsed.logfile == "sim.log"

    parsed, _ = configure_runtime(["-v", "2"]+basearg, comm)
    assert parsed.verbose == 2

    parsed, _ = configure_runtime(["--disable-field"]+basearg, comm)
    assert parsed.disable_field

    parsed, _ = configure_runtime(["--disable-bonds"]+basearg, comm)
    assert parsed.disable_bonds

    parsed, _ = configure_runtime(["--disable-angle-bonds"]+basearg, comm)
    assert parsed.disable_angle_bonds

    parsed, _ = configure_runtime(["--disable-dihedrals"]+basearg, comm)
    assert parsed.disable_dihedrals

    parsed, _ = configure_runtime(["--disable-dipole"]+basearg, comm)
    assert parsed.disable_dipole

    parsed, _ = configure_runtime(["--double-precision"]+basearg, comm)
    assert parsed.double_precision

    parsed, _ = configure_runtime(["--double-output"]+basearg, comm)
    assert parsed.double_output

    parsed, _ = configure_runtime(["--dump-per-particle"]+basearg, comm)
    assert parsed.dump_per_particle

    parsed, _ = configure_runtime(["--force-output"]+basearg, comm)
    assert parsed.force_output

    parsed, _ = configure_runtime(["--velocity-output"]+basearg, comm)
    assert parsed.velocity_output

    parsed, _ = configure_runtime(["--disable-mpio"]+basearg, comm)
    assert parsed.disable_mpio

    parsed, _ = configure_runtime(["--destdir", "testdir"]+basearg, comm)
    assert parsed.destdir == "testdir"

    parsed, _ = configure_runtime(["--seed", "54321"]+basearg, comm)
    assert parsed.seed == 54321

    parsed, _ = configure_runtime(["--logfile", "test.log"]+basearg, comm)
    assert parsed.logfile == "test.log"
