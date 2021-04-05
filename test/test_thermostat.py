import pytest
import numpy as np
import warnings
import pmesh
from mpi4py import MPI
from input_parser import Config
from thermostat import velocity_rescale
from field import domain_decomposition
from file_io import distribute_input


@pytest.mark.mpi()
def test_thermostat_coupling_groups(molecules_with_solvent):
    indices, positions, molecules, velocities, bonds, types = molecules_with_solvent
    n_particles = len(indices)
    box_size = np.array([10, 10, 10], dtype=np.float64)
    config = Config(n_steps=0, time_step=0.1, box_size=box_size, tau=0.1,
                    mesh_size=[5, 5, 5], sigma=0.5, kappa=0.05,
                    n_particles=len(indices), target_temperature=310.0,
                    thermostat_work=0.0, mass=72.0)
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
    bonds_ = bonds[rank_range]
    types_ = types[rank_range]

    positions_, indices_, velocities_, types_, bonds_, molecules_ = domain_decomposition(
        positions_, pm, indices_, velocities_, types_, molecules=molecules_,
        bonds=bonds_, verbose=2, comm=comm
    )

    kinetic_energy_A = 55.699870
    kinetic_energy_B = 59.560380
    kinetic_energy_C = 7.4953280
    kinetic_energy_D = 45.699978

    temperature_A = 318.8413449517521
    temperature_B = 318.2106216743212
    temperature_C = 600.6743926201775
    temperature_D = 244.1592650562905

    for t, T, K in zip(
        ("A", "B", "C", "D"),
        (temperature_A, temperature_B, temperature_C, temperature_D),
        (kinetic_energy_A, kinetic_energy_B, kinetic_energy_C, kinetic_energy_D)
    ):
        n_particles_ = comm.allreduce(len(np.where(types_ == np.string_(t))[0]),
                                      MPI.SUM)
        kinetic_energy = comm.allreduce(
            0.5 * 72.0 * np.sum(velocities_[types_ == np.string_(t)]**2),
            MPI.SUM
        )
        temperature = (2.0 / 3.0) * kinetic_energy / ((2.479 / 298.0) * n_particles_)
        assert K == pytest.approx(kinetic_energy, abs=1e-3)
        assert T == pytest.approx(temperature, abs=1e-3)

    velocities__ = velocities_.copy()
    velocities__ = velocity_rescale(velocities__, config, comm,
                                    R1=0.005, Ri2_sum=134.0026)

    kinetic_energy = comm.allreduce(
        0.5 * 72.0 * np.sum(velocities__**2),
        MPI.SUM
    )
    temperature = (2.0 / 3.0) * kinetic_energy / ((2.479 / 298.0) * n_particles)
    assert temperature == pytest.approx(305.0, abs=1e-2)
