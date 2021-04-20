import pytest
import numpy as np
import warnings
import pmesh
from mpi4py import MPI
from input_parser import Config, _find_unique_names
from thermostat import csvr_thermostat
from field import domain_decomposition
from file_io import distribute_input


@pytest.mark.mpi()
def test_thermostat_coupling_groups(molecules_with_solvent):

    class RandomMock:
        ind = 0

        def __init__(self, x):
            self.x = x

        def __call__(self, *args):
            self.ind += 1
            return self.x[self.ind - 1]

    def T_from_K(K, N):
        return 2 * K / (3 * (2.479 / 298.0) * N)

    def K_from_V(V, comm=MPI.COMM_WORLD, m=72.0):
        return 0.5 * m * comm.allreduce(np.sum(V**2), MPI.SUM)

    def K_from_T(T, N):
        return (2.479 / 298.0) * 3 * T * N / 2

    (indices_global, positions_global, molecules_global, velocities_global,
     bonds_global, names_global, types_global) = molecules_with_solvent

    n_particles = len(indices_global)
    box_size = np.array([10, 10, 10], dtype=np.float64)
    config = Config(
        n_steps=0, time_step=0.032958582578275, box_size=box_size,
        tau=0.925852989520023, mesh_size=[2, 2, 2], sigma=0.5, kappa=0.05,
        n_particles=len(indices_global), target_temperature=310.0,
        thermostat_work=0.0, mass=72.0
    )
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore', category=np.VisibleDeprecationWarning,
            message=r'Creating an ndarray from ragged nested sequences'
        )
        pm = pmesh.ParticleMesh(config.mesh_size, BoxSize=config.box_size,
                                dtype='f8', comm=comm)

    # Test stub acting like a hdf5 file for distribute_input
    in_file = {'molecules': molecules_global, 'indices': indices_global}
    rank_range, molecules_flag = distribute_input(
        in_file, rank, size,  config.n_particles, max_molecule_size=6,
        comm=comm,
    )
    positions = positions_global[rank_range]
    molecules = molecules_global[rank_range]
    indices = indices_global[rank_range]
    velocities = velocities_global[rank_range]
    bonds = bonds_global[rank_range]
    types = types_global[rank_range]
    names = names_global[rank_range]

    dd = domain_decomposition(
        positions, pm, indices, velocities, names, types, molecules=molecules,
        bonds=bonds, verbose=2, comm=comm,
    )
    positions, indices, velocities, names, types, bonds, molecules = dd

    # Regression testing to ensure the underlying system didn't change.
    kinetic_energy = 168.45555165866017
    temperature = 300.0000262607726
    kinetic_energy_species = [55.69986940118755, 59.56037533981311,
                              7.495327677224677, 45.699979240434836]
    temperature_species = [318.8413354377512, 318.2106019678958,
                           600.6743707981589, 244.15927235264365]

    total_kinetic_energy = K_from_V(velocities, comm=comm)
    total_temperature = T_from_K(total_kinetic_energy, n_particles)
    assert total_kinetic_energy == pytest.approx(kinetic_energy, abs=1e-12)
    assert total_temperature == pytest.approx(temperature, abs=1e-12)

    print("TOTAL NPARTICLES, KINETIC, TEMP", n_particles,
          total_kinetic_energy, total_temperature)

    for t, T, K in zip(("A", "B", "C", "D"), temperature_species,
                       kinetic_energy_species):
        n_particles_species = comm.allreduce(
            len(np.where(names == np.string_(t))[0]), MPI.SUM
        )
        K_species = K_from_V(velocities[names == np.string_(t)], comm=comm)
        T_species = T_from_K(K_species, n_particles_species)
        assert K == pytest.approx(K_species, abs=1e-13)
        assert T == pytest.approx(T_species, abs=1e-13)
        print("SPECIES, NPARTICLES, KINETIC, TEMP", t, n_particles_species,
              K_species, T_species)

    config = _find_unique_names(config, names, comm=comm)
    velocities_copy = velocities.copy()

    # Hijack the _random_gaussian and _random_chi_squared interal functions in
    # thermostat to control the exact numbers generated for testing.
    random_chi_squared = RandomMock([125.4595634810623])
    random_gaussian = RandomMock([0.5579657512081987])
    velocities_copy = csvr_thermostat(
        velocities_copy, names, config, comm=comm,
        random_gaussian=random_gaussian, random_chi_squared=random_chi_squared,
    )
    kinetic_energy_new = K_from_V(velocities_copy, comm=comm)
    assert kinetic_energy_new == pytest.approx(171.257138006274, abs=1e-13)
    assert config.thermostat_work == pytest.approx(2.80158634761387, abs=1e-13)

    config.thermostat_work = 0.0
    config.thermostat_coupling_groups = [
        ["A"],
        ["B"],
        ["C"],
        ["D"],
    ]
    velocities_copy = velocities.copy()
    random_chi_squared = RandomMock([35.43824087713971, 27.57022975113815,
                                     2.024725328228174, 36.50208472031436])
    random_gaussian = RandomMock([-1.752320325907187, 1.099694957420625,
                                  0.6113448515745533, -0.7183266831611322])
    velocities_copy = csvr_thermostat(
        velocities_copy, names, config, comm=comm,
        random_gaussian=random_gaussian, random_chi_squared=random_chi_squared,
    )

    K_species = np.array([
        K_from_V(velocities_copy[names == np.string_("A")], comm=comm),
        K_from_V(velocities_copy[names == np.string_("B")], comm=comm),
        K_from_V(velocities_copy[names == np.string_("C")], comm=comm),
        K_from_V(velocities_copy[names == np.string_("D")], comm=comm),
    ])
    K_expected = np.array([50.03215278458857, 62.31604075323946,
                           8.039648643032598, 43.74504593736311])

    assert np.allclose(K_species, K_expected, atol=1e-13, rtol=0.0)
    assert config.thermostat_work == pytest.approx(-4.322663540436441,
                                                   abs=1e-13)

    config.thermostat_work = 0.0
    config.thermostat_coupling_groups = [
        ["A", "B", "C"],
        ["D"],
    ]
    velocities_copy = velocities.copy()
    random_chi_squared = RandomMock([97.07130218590895, 33.06359496718739])
    random_gaussian = RandomMock([0.1661408606772054, -0.06216797747541603])
    velocities_copy = csvr_thermostat(
        velocities_copy, names, config, comm=comm,
        random_gaussian=random_gaussian, random_chi_squared=random_chi_squared,
    )

    inds_ABC = np.where(
        np.logical_or.reduce(
            (names == np.string_("A"), names == np.string_("B"),
             names == np.string_("C"))
        )
    )
    K_group_ABC = K_from_V(velocities_copy[inds_ABC], comm=comm)
    K_group_D = K_from_V(velocities_copy[names == np.string_("D")], comm=comm)
    assert K_group_ABC == pytest.approx(123.6090634948376, abs=1e-13)
    assert K_group_D == pytest.approx(45.41754237593906, abs=1e-13)
    assert config.thermostat_work == pytest.approx(0.5710542121164457,
                                                   abs=1e-13)
