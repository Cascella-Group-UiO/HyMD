import pytest
import numpy as np
import logging
import filecmp
import os
from mpi4py import MPI
from hymd.input_parser import Config
from hymd.file_io import distribute_input
from hymd.plumed import PlumedBias

@pytest.mark.mpi()
def test_plumed_bias_obj(molecules_with_solvent, change_test_dir, caplog):
    caplog.set_level(logging.INFO)
    indices, positions, molecules, velocities, bonds, names, types = molecules_with_solvent
    box_size = np.array([10, 10, 10], dtype=np.float64)
    config = Config(n_steps=1, time_step=0.03, box_size=box_size,
                    mesh_size=[5, 5, 5], sigma=0.5, kappa=0.05,
                    n_particles=len(indices), target_temperature=300.0)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
   
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

    plumed = PlumedBias(
      config, 
      "plumed.dat", 
      "plumed.out",
      comm
    )

    assert plumed.api_version > 3

    forces_ = np.zeros_like(positions_)
    charges_ = np.zeros_like(indices_, dtype=np.double)

    needs_energy = plumed.prepare(
        0,
        forces_,
        positions_,
        indices_,
        config,
        charges_
    )

    assert needs_energy == False

    plumed_forces, plumed_bias = plumed.calc(forces_, 0.0)

    if rank == 0:
        ref_forces = np.loadtxt("refforces.txt")
        assert np.allclose(plumed_forces, ref_forces, atol=1e-13)    
        assert filecmp.cmp("DIST", "refDIST") == True
        assert filecmp.cmp("testdump.xyz", "reftestdump.xyz") == True

    assert plumed_bias != 0.0
