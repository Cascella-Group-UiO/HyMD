import numba
import numpy as np
from mpi4py import MPI
from compute_gaussian_core import cgc as gaussian_core_kernel__fortran


def setup_chi_array(positions, types, config):
    N = len(positions)
    chi = np.empty(shape=(N, N), dtype=np.float32, order='F')

    chi_type = np.empty(shape=(5, 5), dtype=np.float32)
    for c in config.chi:
        t1 = config.name_to_type_map[c.atom_1]
        t2 = config.name_to_type_map[c.atom_2]
        e = c.interaction_energy
        chi_type[t1, t2] = e
        chi_type[t2, t1] = e

    @numba.jit(nopython=True)
    def chi_array_kernel(chi_type, chi):
        for i in range(N):
            for j in range(i + 1, N):
                chi[i, j] = chi_type[types[i], types[j]]
                chi[j, i] = chi[i, j]
        return chi

    chi = chi_array_kernel(chi_type, chi)
    return chi


@numba.jit(nopython=True, fastmath=True)
def gaussian_core_kernel__numba(r, chi, f, box_size, sigma, kappa):
    f.fill(0.0)
    denominator = 1 / (4 * sigma**2)
    V = np.prod(box_size)
    phi0 = len(r) / V
    factor = 1 / (16 * np.pi**(3 / 2) * kappa * sigma**5 * phi0)

    for i, ri in enumerate(r):
        for j in range(i + 1, len(r)):
            rj = r[j, :]
            rij = rj - ri

            # Apply periodic boundary conditions to the distance rij
            rij[0] -= box_size[0] * np.around(rij[0] / box_size[0])
            rij[1] -= box_size[1] * np.around(rij[1] / box_size[1])
            rij[2] -= box_size[2] * np.around(rij[2] / box_size[2])

            dr2 = np.dot(rij, rij)
            exp = np.exp(-dr2 * denominator)

            f_bond_vector = rij * exp * factor * (1 + kappa * chi[i, j])
            f[i, :] += f_bond_vector
            f[j, :] -= f_bond_vector


def gaussian_core_forces(positions, force, chi, config):
    energy = gaussian_core_kernel__fortran(
        positions, chi, force, np.asarray(config.box_size), config.sigma,
        config.kappa, len(positions)
    )
    return energy


def __setup_test_dppc_system(N_lipid, N_solvent, seed=None):
    from force import Chi
    from input_parser import (Config, _setup_type_to_name_map,
                              _find_unique_names)

    if seed is not None:
        np.random.seed(seed)
    N_lipid = 52
    N_solvent = 1400
    N = 12 * N_lipid + N_solvent
    positions = np.random.uniform(low=0.0, high=15.0, size=(N, 3))
    positions = np.asfortranarray(positions).astype(np.float32)
    positions[positions[:, 0] > 13.0, 0] -= 13.0
    positions[positions[:, 1] > 14.0, 0] -= 14.0
    forces = np.zeros_like(positions, dtype=np.float32)
    dppc_types = np.array([0, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.int32)
    dppc_names = np.array([np.string_(s) for s in ('N', 'P', 'G', 'G', 'C',
                                                   'C', 'C', 'C', 'C', 'C',
                                                   'C', 'C')], dtype='S1')
    names = np.concatenate((np.repeat(dppc_names, N_lipid),
                            np.full(N_solvent, np.string_('W'), dtype='S1')))
    types = np.concatenate((np.repeat(dppc_types, N_lipid),
                            np.full(N_solvent, 4, dtype=np.int32)))
    chi_list = [Chi(atom_1='C', atom_2='W', interaction_energy=42.24),
                Chi(atom_1='G', atom_2='C', interaction_energy=10.47),
                Chi(atom_1='N', atom_2='W', interaction_energy=-3.77),
                Chi(atom_1='G', atom_2='W', interaction_energy=4.53),
                Chi(atom_1='N', atom_2='P', interaction_energy=-9.34),
                Chi(atom_1='P', atom_2='G', interaction_energy=8.04),
                Chi(atom_1='N', atom_2='G', interaction_energy=1.97),
                Chi(atom_1='P', atom_2='C', interaction_energy=14.72),
                Chi(atom_1='P', atom_2='W', interaction_energy=-1.51),
                Chi(atom_1='N', atom_2='C', interaction_energy=13.56)]
    for n in ('N', 'P', 'G', 'C', 'W'):
        chi_list.append(Chi(atom_1=n, atom_2=n, interaction_energy=0.0))

    config = Config(n_steps=None, time_step=None, box_size=[13, 14, 15],
                    mesh_size=[3, 3, 3], sigma=0.5, kappa=0.05, chi=chi_list,
                    n_particles=N)
    config = _setup_type_to_name_map(config, names, types)
    config = _find_unique_names(config, names)
    return positions, forces, names, types, config


def __force_energy_grid_size_accuracy():
    import warnings
    import pmesh.pm as pmesh
    from file_io import distribute_input
    from field import update_field, compute_field_force
    from hamiltonian import DefaultWithChi

    r, f, names, types, config = __setup_test_dppc_system(52, 1400, seed=27654)
    N = config.n_particles
    chi = setup_chi_array(r, types, config)
    gaussian_core_energy = gaussian_core_forces(r, f, chi, config)

    in_file = {'positions': r, 'indices': np.arange(config.n_particles)}
    grab, _ = distribute_input(in_file, MPI.COMM_WORLD.Get_rank(),
                               MPI.COMM_WORLD.Get_size(), config.n_particles,
                               max_molecule_size=15, comm=MPI.COMM_WORLD)
    pos_ = r[grab, :]
    typ_ = types[grab]
    mesh = np.geomspace(3, 200, num=100, dtype=np.int32)
    mesh = np.unique(mesh)

    ind = np.random.randint(len(pos_))
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f' mesh{106*" "}relative energy\n size     {"force":>32s} '
              f'{"energy/N":>20s} {"relative force difference [%]":>43s} '
              f'    {"difference [%]":>12s}\n{126*"="}')
        print(f'     {f[ind, 0]:12.8f} {f[ind, 1]:12.8f} {f[ind, 2]:12.8f} '
              f'{gaussian_core_energy / N:20.10f}')

    for m in mesh:
        config.mesh_size = [m, m, m]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore', category=np.VisibleDeprecationWarning,
                message=r'Creating an ndarray from ragged nested sequences'
            )
            pm = pmesh.ParticleMesh(config.mesh_size, BoxSize=config.box_size,
                                    dtype='f4', comm=MPI.COMM_WORLD)
        phi = [pm.create('real', value=0.0) for _ in range(config.n_types)]
        phi_fourier = [pm.create('complex', value=0.0) for _ in
                       range(config.n_types)]
        force_on_grid = [[pm.create('real', value=0.0) for d in range(3)]
                         for _ in range(config.n_types)]
        v_ext_fourier = [pm.create('complex', value=0.0) for _ in range(4)]
        v_ext = [
            pm.create('real', value=0.0) for _ in range(config.n_types)
        ]
        layouts = [
            pm.decompose(pos_[typ_ == t]) for t in range(config.n_types)
        ]
        hamiltonian = DefaultWithChi(config, config.unique_names,
                                     config.type_to_name_map)
        update_field(phi, layouts, force_on_grid, hamiltonian, pm, pos_,
                     typ_, config, v_ext, phi_fourier, v_ext_fourier,
                     compute_potential=False)
        field_forces = np.zeros_like(pos_)
        compute_field_force(layouts, pos_, force_on_grid, field_forces, typ_,
                            config.n_types)
        V = np.prod(config.box_size)
        n_mesh__cells = np.prod(np.full(3, config.mesh_size))
        volume_per_cell = V / n_mesh__cells
        w = hamiltonian.w(phi) * volume_per_cell
        field_energy = w.csum()

        if MPI.COMM_WORLD.Get_rank() == 0:
            i = field_forces[field_forces > 0.01]

            ff = field_forces
            df = np.abs(ff[ind, :] - f[ind, :]) / np.abs(f[ind, :]) * 100.0
            de = (np.abs(gaussian_core_energy - field_energy)
                  / np.abs(gaussian_core_energy)) * 100.0
            print(f'{m:4} {ff[ind, 0]:12.8f} {ff[ind, 1]:12.8f} '
                  f'{ff[ind, 2]:12.8f} {field_energy / N:20.10f}     '
                  f'{df[0]:12.8f} {df[1]:12.8f} {df[2]:12.8f} {de:17.8f}')


if __name__ == '__main__':
    __force_energy_grid_size_accuracy()
