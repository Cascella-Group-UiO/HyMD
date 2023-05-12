import pytest
import numpy as np
import logging
import filecmp
import os
import sys
from mpi4py import MPI
from hymd.input_parser import Config
from hymd.file_io import distribute_input

@pytest.mark.mpi(min_size=2)
def test_plumed_bias_obj(molecules_with_solvent, change_test_dir, tmp_path, 
                         caplog, monkeypatch):
    pytest.importorskip("plumed")
    from hymd.plumed import PlumedBias
    caplog.set_level(logging.INFO)
    indices, positions, molecules, velocities, bonds, names, types = molecules_with_solvent
    box_size = np.array([10, 10, 10], dtype=np.float64)
    config = Config(n_steps=1, time_step=0.03, box_size=box_size,
                    mesh_size=[5, 5, 5], sigma=0.5, kappa=0.05,
                    n_particles=len(indices), target_temperature=300.0)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # create PLUMED input with the correct tmp_path
    plumed_str = """d14: DISTANCE ATOMS=1,4

RESTRAINT ARG=d14 KAPPA=200.0 AT=2.0

DUMPATOMS ATOMS=1-45 FILE={}
PRINT ARG=d14 FILE={}""".format(os.path.join(tmp_path,"testdump.xyz"), 
                                os.path.join(tmp_path,"DIST"))
    with open(os.path.join(tmp_path,"plumed.dat"),"w") as f:
        f.write(plumed_str)

    # Test stub acting like a hdf5 file for distribute_input
    in_file = {'molecules': molecules, 'indices': indices}
    rank_range, molecules_flag = distribute_input(
        in_file, rank, size,  config.n_particles, 6, comm=comm
    )
    positions_ = positions[rank_range]
    molecules_ = molecules[rank_range]
    indices_ = indices[rank_range].astype(np.intc)
    velocities_ = velocities[rank_range]
    bonds_ = bonds[rank_range]
    types_ = types[rank_range]
    names_ = names[rank_range]

    # test working case
    plumed_obj = PlumedBias(
        config, 
        os.path.join(tmp_path,"plumed.dat"),
        os.path.join(tmp_path,"plumed.out"),
        comm=comm,
        verbose=2
    )

    if rank == 0:
        log = caplog.text
        print(log)
        cmp_strings = (
            "Attempting to read PLUMED input from",
            "Using PLUMED_KERNEL="
        )
        assert all([(s in log) for s in cmp_strings])
        caplog.clear()
    assert plumed_obj.api_version > 3

    forces_ = np.zeros_like(positions_)
    charges_ = np.zeros_like(indices_, dtype=np.double)

    needs_energy = plumed_obj.prepare(
        0,
        forces_,
        positions_,
        indices_,
        config,
        charges_
    )

    assert needs_energy == False
    if rank == 0:
        log = caplog.text
        cmp_strings = (
            "Setting PLUMED pointers for step",
        )
        assert all([(s in log) for s in cmp_strings])
        caplog.clear()

    plumed_forces, plumed_bias = plumed_obj.calc(forces_, 0.0)
    if rank == 0:
        log = caplog.text
        cmp_strings = (
            "Calculating PLUMED forces",
            "Done calculating PLUMED forces"
        )
        assert all([(s in log) for s in cmp_strings])
        caplog.clear()

    if rank == 0:
        ref_forces = np.loadtxt("refforces.txt")
        assert plumed_forces == pytest.approx(ref_forces, abs=1e-8)
        assert filecmp.cmp(os.path.join(tmp_path,"DIST"), "refDIST") == True
        assert filecmp.cmp(os.path.join(tmp_path,"testdump.xyz"), "reftestdump.xyz") == True

    assert plumed_bias == pytest.approx(635.5621691482, abs=1e-8)

    # test calc without prepare
    with pytest.raises(RuntimeError) as recorded_error:
        _, _ = plumed_obj.calc(forces_, 0.0)
        if rank == 0:
            log = caplog.text
            assert "without first calling prepare method" in log
    message = str(recorded_error.value)
    assert "without first calling prepare method" in message
    caplog.clear()

    # finalize PlumedBias object
    plumed_obj.finalize()


@pytest.mark.mpi()
@pytest.mark.skip(reason="Currently fails in CI due to environment variable")
def test_fail_plumed_bias_obj(monkeypatch):
    pytest.importorskip("plumed")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # create fake config
    box_size = np.array([10, 10, 10], dtype=np.float64)
    config = Config(n_steps=1, time_step=0.03, box_size=box_size,
                    mesh_size=[5, 5, 5], sigma=0.5, kappa=0.05,
                    n_particles=1000, target_temperature=300.0)

    # try to get rid of PLUMED_KERNEL path
    monkeypatch.delenv("PLUMED_KERNEL")
    found_plumed_in_syspath = False
    for libpath in sys.path:
        if "plumed2/" in libpath:
            found_plumed_in_syspath = True
            break

    import hymd.plumed
   
    with pytest.raises(RuntimeError) as recorded_error:
        with hymd.plumed.PlumedBias(
            config, 
            "test.in", 
            "test.out",
            comm=comm,
            verbose=2
        ) as _:
            if rank == 0:
                log = caplog.text
                cmp_strings = (
                    "The PLUMED_KERNEL environment variable is not set",
                )
                assert all([(s in log) for s in cmp_strings])

    if not found_plumed_in_syspath:
        cmp_strings = (
            "HyMD was not able to create a PLUMED object",
            "Maybe there is a problem with your PLUMED_KERNEL?"
        )
        message = str(recorded_error.value)
        assert all([(s in message) for s in cmp_strings])


def test_unavailable_plumed(hide_available_plumed):
    import hymd.plumed

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # create fake config
    box_size = np.array([10, 10, 10], dtype=np.float64)
    config = Config(n_steps=1, time_step=0.03, box_size=box_size,
                    mesh_size=[5, 5, 5], sigma=0.5, kappa=0.05,
                    n_particles=1000, target_temperature=300.0)

    with pytest.raises(ImportError) as recorded_error:
        with hymd.plumed.PlumedBias(
            config, 
            "test.in", 
            "test.out",
            comm=comm,
            verbose=2
        ) as _:
            if rank == 0:
                log = caplog.text
                cmp_strings = (
                    "You are trying to use PLUMED",
                    "but HyMD could not import py-plumed."
                )
                assert all([(s in log) for s in cmp_strings])
    cmp_strings = (
        "You are trying to use PLUMED",
        "but HyMD could not import py-plumed."
    )
    message = str(recorded_error.value)
    assert all([(s in message) for s in cmp_strings])
