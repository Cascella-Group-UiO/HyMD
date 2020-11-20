from mpi4py import MPI
import pytest
import warnings
import pmesh
import logging
import numpy as np
from input_parser import Config
from file_io import distribute_input
from field import domain_decomposition


@pytest.mark.mpi()
def test_domain_decomposition(molecules_with_solvent, caplog):
    caplog.set_level(logging.INFO)
    indices, positions, molecules, velocities = molecules_with_solvent
    box_size = np.array([10, 10, 10], dtype=np.float64)
    config = Config(n_steps=0, time_step=0.03, box_size=box_size,
                    mesh_size=[60, 60, 60], sigma=0.5, kappa=0.05,
                    n_particles=len(indices))
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore', category=np.VisibleDeprecationWarning,
            message=r'Creating an ndarray from ragged nested sequences'
        )
        pm = pmesh.ParticleMesh(config.mesh_size, BoxSize=config.box_size,
                                dtype='f4', comm=comm)

    # Test stub acting like a hdf5 file for distribute_input
    in_file = {'molecules': molecules, 'indices': indices}
    rank_range, molecules_flag = distribute_input(
        in_file, rank, size,  config.n_particles, 6, comm=comm
    )
    positions_ = positions[rank_range]
    molecules_ = molecules[rank_range]
    indices_ = indices[rank_range]
    velocities_ = velocities[rank_range]

    positions_, molecules_, indices_, velocities_ = domain_decomposition(
        positions_, molecules_, pm, indices_, velocities_, verbose=2, comm=comm
    )
    if rank == 0:
        assert 'DOMAIN_DECOMP' in caplog.text

    unique_molecules = np.unique(molecules_)
    receive_buffer = comm.gather(unique_molecules, root=0)
    unique_molecules = comm.bcast(receive_buffer, root=0)

    # Ensure no molecules are split across ranks
    for i, ranki in enumerate(unique_molecules):
        for mi in ranki:
            for j, rankj in enumerate(unique_molecules):
                if i != j:
                    assert mi not in rankj
