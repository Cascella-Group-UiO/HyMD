import pytest
import logging
import numpy as np
import os
import argparse
from mpi4py import MPI
from argparse import Namespace
from hymd.configure_runtime import configure_runtime, extant_file
from hymd.input_parser import Config


def test_extant_file(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(argparse.ArgumentTypeError) as recorded_error:
        _ = extant_file("inexistant_test_file.txt")
        log = caplog.text
        assert "does not exist" in log
    message = str(recorded_error.value)
    assert "does not exist" in message
    
    file_path = os.path.join(tmp_path, "test.txt")
    with open(file_path, 'w') as f:
        f.write("test")

    ret_path = extant_file(file_path)
    assert ret_path == file_path

    caplog.clear()


def test_configure_runtime(h5py_molecules_file, config_toml,
                           change_tmp_dir, caplog):
    caplog.set_level(logging.ERROR)
    comm = MPI.COMM_WORLD
    config_toml_file, _ = config_toml

    basearg = [config_toml_file, h5py_molecules_file]

    parsed, config, prng, topol = configure_runtime(basearg, comm)
    assert isinstance(parsed, Namespace)
    assert isinstance(config, Config)
    assert isinstance(prng, np.random.Generator)
    assert parsed.destdir == "."
    assert parsed.logfile == "sim.log"
    assert parsed.plumed_outfile == "plumed.out"

    parsed, _, _, _ = configure_runtime(["-v", "2"]+basearg, comm)
    assert parsed.verbose == 2

    parsed, _, _, _ = configure_runtime(["--disable-field"]+basearg, comm)
    assert parsed.disable_field

    parsed, _, _, _ = configure_runtime(["--disable-bonds"]+basearg, comm)
    assert parsed.disable_bonds

    parsed, _, _, _ = configure_runtime(["--disable-angle-bonds"]+basearg, comm)
    assert parsed.disable_angle_bonds

    parsed, _, _, _ = configure_runtime(["--disable-dihedrals"]+basearg, comm)
    assert parsed.disable_dihedrals

    parsed, _, _, _ = configure_runtime(["--disable-dipole"]+basearg, comm)
    assert parsed.disable_dipole

    parsed, _, _, _ = configure_runtime(["--double-precision"]+basearg, comm)
    assert parsed.double_precision

    parsed, _, _, _ = configure_runtime(["--double-output"]+basearg, comm)
    assert parsed.double_output

    parsed, _, _, _ = configure_runtime(["--dump-per-particle"]+basearg, comm)
    assert parsed.dump_per_particle

    parsed, _, _, _ = configure_runtime(["--force-output"]+basearg, comm)
    assert parsed.force_output

    parsed, _, _, _ = configure_runtime(["--velocity-output"]+basearg, comm)
    assert parsed.velocity_output

    parsed, _, _, _ = configure_runtime(["--disable-mpio"]+basearg, comm)
    assert parsed.disable_mpio

    parsed, _, _, _ = configure_runtime(["--destdir", "testdir"]+basearg, comm)
    assert parsed.destdir == "testdir"

    parsed, _, _, _ = configure_runtime(["--seed", "54321"]+basearg, comm)
    assert parsed.seed == 54321

    parsed, _, _, _ = configure_runtime(["--logfile", "test.log"]+basearg, comm)
    assert parsed.logfile == "test.log"

    parsed, _, _, _ = configure_runtime(["--plumed", "test.log"]+basearg, comm)
    assert parsed.plumed == "test.log"

    parsed, _, _, _ = configure_runtime(
                       ["--plumed-outfile", "test.plumed.out"]+basearg,
                       comm
                   )
    assert parsed.plumed_outfile == "test.plumed.out"

