import numpy as np
import numba
import datetime
import sys
import os
import arrays_dppc_only
import functools

sys.path.append(os.path.join(os.pardir, 'hymd'))
from force import prepare_bonds  # noqa: E402
from input_parser import (parse_config_toml, read_config_toml,  # noqa: E402
                          check_config)  # noqa: E402


def timing(function=None, n_times=1, hot_start=False, print_return=False):
    assert callable(function) or function is None

    def _decorator(function_):
        @functools.wraps(function_)
        def wrapper(*args, **kwargs):
            if hot_start:
                # Let numba perform any needed compilation without it affecting
                # the timing.
                ret = function_(*args, **kwargs)
            t_start = datetime.datetime.now()
            ret = 0
            for _ in range(n_times):
                # Just to make sure this is not optimized away or something,
                # make it a sum.
                ret += function_(*args, **kwargs)
            ret /= n_times
            t_end = datetime.datetime.now()

            timedelta = t_end - t_start
            hours, rem = divmod(timedelta.seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            microseconds = timedelta.microseconds
            print(f'{function_.__name__} ({n_times} times):')
            print(f'{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds}')

            if print_return:
                print(f'E = {ret:30.15f}')
            print()

            return ret
        return wrapper
    return _decorator(function) if callable(function) else _decorator


@timing(n_times=1000, hot_start=False, print_return=True)
def compute_bond_forces__plain(f_bonds, r, bonds_2, box_size):
    f_bonds.fill(0.0)
    energy = 0.0

    for i, j, r0, k in bonds_2:
        k = np.float32(k)
        r0 = np.float32(r0)
        ri = r[i, :]
        rj = r[j, :]
        rij = rj - ri

        # Apply periodic boundary conditions to the distance rij
        for dim in range(len(rij)):
            rij[dim] -= box_size[dim] * np.around(rij[dim] / box_size[dim])
        dr = np.linalg.norm(rij)
        df = -k * (dr - r0)
        f_bond_vector = df * rij / dr
        f_bonds[i, :] -= f_bond_vector
        f_bonds[j, :] += f_bond_vector

        energy += 0.5 * k * (dr - r0)**2
    return energy


@timing(n_times=1000, hot_start=True, print_return=True)
@numba.jit(nopython=True, fastmath=False)
def compute_bond_forces__numba(f_bonds, r, box_size, bonds_2_atom1,
                               bonds_2_atom2, bonds_2_equilibrium,
                               bonds_2_stength):
    f_bonds.fill(0.0)
    energy = 0.0

    for ind in range(len(bonds_2_atom1)):
        i = bonds_2_atom1[ind]
        j = bonds_2_atom2[ind]
        r0 = bonds_2_equilibrium[ind]
        k = bonds_2_stength[ind]
        ri = r[i, :]
        rj = r[j, :]
        rij = rj - ri

        # Apply periodic boundary conditions to the distance rij
        rij[0] -= box_size[0] * np.around(rij[0] / box_size[0])
        rij[1] -= box_size[1] * np.around(rij[1] / box_size[1])
        rij[2] -= box_size[2] * np.around(rij[2] / box_size[2])

        dr = np.linalg.norm(rij)
        df = -k * (dr - r0)
        f_bond_vector = df * rij / dr
        f_bonds[i, :] -= f_bond_vector
        f_bonds[j, :] += f_bond_vector

        energy += 0.5 * k * (dr - r0)**2
    return energy


@timing(n_times=1000, hot_start=True, print_return=True)
@numba.jit(nopython=True, fastmath=True)
def compute_bond_forces__numba_fastmath(f_bonds, r, box_size, bonds_2_atom1,
                                        bonds_2_atom2, bonds_2_equilibrium,
                                        bonds_2_stength):
    f_bonds.fill(0.0)
    energy = 0.0

    for ind in range(len(bonds_2_atom1)):
        i = bonds_2_atom1[ind]
        j = bonds_2_atom2[ind]
        r0 = bonds_2_equilibrium[ind]
        k = bonds_2_stength[ind]
        ri = r[i, :]
        rj = r[j, :]
        rij = rj - ri

        # Apply periodic boundary conditions to the distance rij
        rij[0] -= box_size[0] * np.around(rij[0] / box_size[0])
        rij[1] -= box_size[1] * np.around(rij[1] / box_size[1])
        rij[2] -= box_size[2] * np.around(rij[2] / box_size[2])

        dr = np.linalg.norm(rij)
        df = -k * (dr - r0)
        f_bond_vector = df * rij / dr
        f_bonds[i, :] -= f_bond_vector
        f_bonds[j, :] += f_bond_vector

        energy += 0.5 * k * (dr - r0)**2
    return energy


if __name__ == '__main__':
    """
    compute_bond_forces__plain (1000 times):
    00:03:06.434376
    E =           7770.150099065346694

    compute_bond_forces__numba (1000 times):
    00:00:05.570653
    E =           7770.150099065346694

    compute_bond_forces__numba_fastmath (1000 times):
    00:00:05.850952
    E =           7770.150099065346694
    """
    indices, names, bonds, positions, types, molecules = (
        arrays_dppc_only.arrays()
    )
    f_bonds = np.empty_like(positions)

    config_file_path = os.path.join(os.pardir, 'examples', 'config.toml')
    toml_config = read_config_toml(config_file_path)
    config = parse_config_toml(toml_config, file_path=config_file_path)
    config.n_particles = 6337
    box_size = np.array(config.box_size)
    config = check_config(config, indices, names, types)

    bonds_2, bonds_3 = prepare_bonds(molecules, names, bonds, indices,
                                     config)

    E = compute_bond_forces__plain(f_bonds, positions, bonds_2, box_size)

    bonds_2_atom1 = np.empty(len(bonds_2), dtype=int)
    bonds_2_atom2 = np.empty(len(bonds_2), dtype=int)
    bonds_2_equilibrium = np.empty(len(bonds_2), dtype=np.float32)
    bonds_2_stength = np.empty(len(bonds_2), dtype=np.float32)
    for i, b in enumerate(bonds_2):
        bonds_2_atom1[i] = b[0]
        bonds_2_atom2[i] = b[1]
        bonds_2_equilibrium[i] = b[2]
        bonds_2_stength[i] = b[3]

    E = compute_bond_forces__numba(
        f_bonds, positions, box_size, bonds_2_atom1, bonds_2_atom2,
        bonds_2_equilibrium, bonds_2_stength
    )

    E = compute_bond_forces__numba_fastmath(
        f_bonds, positions, box_size, bonds_2_atom1, bonds_2_atom2,
        bonds_2_equilibrium, bonds_2_stength
    )
