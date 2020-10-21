import numpy as np
import numba
import sys
import os
import arrays_dppc_only
from benchmark_compute_bond_forces import timing
from _compute_angle_forces import caf

sys.path.append(os.path.join(os.pardir, 'hymd'))
from force import prepare_bonds  # noqa: E402
from input_parser import (parse_config_toml, read_config_toml,  # noqa: E402
                          check_config)  # noqa: E402


@timing(n_times=1, hot_start=False, print_return=True)
def compute_angle_forces__plain(f_angles, r, bonds_3, box_size):
    f_angles.fill(0.0)
    energy = 0.0

    for a, b, c, theta0, k in bonds_3:
        ra = r[a, :] - r[b, :]
        rc = r[c, :] - r[b, :]

        for dim in range(len(ra)):
            ra[dim] -= box_size[dim] * np.around(ra[dim] / box_size[dim])
            rc[dim] -= box_size[dim] * np.around(rc[dim] / box_size[dim])

        xra = 1.0 / np.sqrt(np.dot(ra, ra))
        xrc = 1.0 / np.sqrt(np.dot(rc, rc))
        ea = ra * xra
        ec = rc * xrc

        cosphi = np.dot(ea, ec)
        theta = np.arccos(cosphi)
        xsinph = 1.0 / np.sqrt(1.0 - cosphi**2)

        d = theta - theta0
        f = - k * d

        xrasin = xra * xsinph * f
        xrcsin = xrc * xsinph * f

        fa = (ea * cosphi - ec) * xrasin
        fc = (ec * cosphi - ea) * xrcsin

        f_angles[a, :] += fa
        f_angles[c, :] += fc
        f_angles[b, :] += -(fa + fc)

        energy -= 0.5 * f * d

    return energy


@timing(n_times=1, hot_start=False, print_return=True)
@numba.jit(nopython=True, fastmath=False)
def compute_angle_forces__numba(f_angles, r, box_size, bonds_3_atom1,
                                bonds_3_atom2, bonds_3_equilibrium,
                                bonds_3_stength):
    f_angles.fill(0.0)
    energy = 0.0

    for ind in range(len(bonds_3_atom1)):
        a = bonds_3_atom1[ind]
        b = bonds_3_atom2[ind]
        c = bonds_3_atom3[ind]
        theta0 = bonds_3_equilibrium[ind]
        k = bonds_3_stength[ind]

        ra = r[a, :] - r[b, :]
        rc = r[c, :] - r[b, :]

        ra[0] -= box_size[0] * np.around(ra[0] / box_size[0])
        ra[1] -= box_size[1] * np.around(ra[1] / box_size[1])
        ra[2] -= box_size[2] * np.around(ra[2] / box_size[2])

        rc[0] -= box_size[0] * np.around(rc[0] / box_size[0])
        rc[1] -= box_size[1] * np.around(rc[1] / box_size[1])
        rc[2] -= box_size[2] * np.around(rc[2] / box_size[2])

        xra = 1.0 / np.sqrt(np.dot(ra, ra))
        xrc = 1.0 / np.sqrt(np.dot(rc, rc))
        ea = ra * xra
        ec = rc * xrc

        cosphi = np.dot(ea, ec)
        theta = np.arccos(cosphi)
        xsinph = 1.0 / np.sqrt(1.0 - cosphi**2)

        d = theta - theta0
        f = - k * d

        xrasin = xra * xsinph * f
        xrcsin = xrc * xsinph * f

        fa = (ea * cosphi - ec) * xrasin
        fc = (ec * cosphi - ea) * xrcsin

        f_angles[a, :] += fa
        f_angles[c, :] += fc
        f_angles[b, :] += -(fa + fc)

        energy -= 0.5 * f * d

    return energy


@timing(n_times=1, hot_start=False, print_return=True)
@numba.jit(nopython=True, fastmath=True)
def compute_angle_forces__numba_fastmath(f_angles, r, box_size, bonds_3_atom1,
                                         bonds_3_atom2, bonds_3_equilibrium,
                                         bonds_3_stength):
    f_angles.fill(0.0)
    energy = 0.0

    for ind in range(len(bonds_3_atom1)):
        a = bonds_3_atom1[ind]
        b = bonds_3_atom2[ind]
        c = bonds_3_atom3[ind]
        theta0 = bonds_3_equilibrium[ind]
        k = bonds_3_stength[ind]

        ra = r[a, :] - r[b, :]
        rc = r[c, :] - r[b, :]

        ra[0] -= box_size[0] * np.around(ra[0] / box_size[0])
        ra[1] -= box_size[1] * np.around(ra[1] / box_size[1])
        ra[2] -= box_size[2] * np.around(ra[2] / box_size[2])

        rc[0] -= box_size[0] * np.around(rc[0] / box_size[0])
        rc[1] -= box_size[1] * np.around(rc[1] / box_size[1])
        rc[2] -= box_size[2] * np.around(rc[2] / box_size[2])

        xra = 1.0 / np.sqrt(np.dot(ra, ra))
        xrc = 1.0 / np.sqrt(np.dot(rc, rc))
        ea = ra * xra
        ec = rc * xrc

        cosphi = np.dot(ea, ec)
        theta = np.arccos(cosphi)
        xsinph = 1.0 / np.sqrt(1.0 - cosphi**2)

        d = theta - theta0
        f = - k * d

        xrasin = xra * xsinph * f
        xrcsin = xrc * xsinph * f

        fa = (ea * cosphi - ec) * xrasin
        fc = (ec * cosphi - ea) * xrcsin

        f_angles[a, :] += fa
        f_angles[c, :] += fc
        f_angles[b, :] += -(fa + fc)

        energy -= 0.5 * f * d

    return energy


@timing(n_times=1000, hot_start=False, print_return=True)
def compute_bond_forces__fortran(f_bonds, r, box_size, bonds_3_atom1,
                                 bonds_3_atom2, bonds_3_atom3,
                                 bonds_3_equilibrium, bonds_3_stength):
    return caf(f_bonds, r, box_size, bonds_3_atom1, bonds_3_atom2,
               bonds_3_atom3, bonds_3_equilibrium, bonds_3_stength)


if __name__ == '__main__':
    """
    Fortran code roughly 770 times faster than plain python.
    Numba code roughly 30 times faster than plain python.
    Fortran code rougly 25 times faster than numba implementation.

    - fastmath has no impact on numba execution time.
    - using single precision has no impact on fortran excution time.
    ============================================================================
    compute_angle_forces__plain (1000 times):
    00:03:23.192013
    E =          23644.298796080365719

    compute_angle_forces__numba (1000 times):
    00:00:06.716348
    E =          23644.298796080365719

    compute_angle_forces__numba_fastmath (1000 times):
    00:00:06.468394
    E =          23644.298796080365719

    compute_bond_forces__fortran_float64 (1000 times):
    00:00:00.262911
    E =          23644.298796080365719

    compute_bond_forces__fortran_float32 (1000 times):
    00:00:00.297183
    E =          23644.298797234576341
    """
    indices, names, bonds, positions, types, molecules = (
        arrays_dppc_only.arrays()
    )
    f_angles = np.empty_like(positions)
    config_file_path = os.path.join(os.pardir, 'examples', 'config.toml')
    toml_config = read_config_toml(config_file_path)
    config = parse_config_toml(toml_config, file_path=config_file_path)
    config.n_particles = 6337
    config.box_size = [13.0, 13.0, 14.0]
    box_size = np.array(config.box_size)
    config = check_config(config, indices, names, types)

    bonds_2, bonds_3 = prepare_bonds(molecules, names, bonds, indices,
                                     config)
    """
    bonds_3_ = []
    for i, b in enumerate(bonds_3):
        if i > 0:
            break
        bonds_3_.append(b)
    bonds_3 = bonds_3_
    """

    bonds_3_atom1 = np.empty(len(bonds_3), dtype=int)
    bonds_3_atom2 = np.empty(len(bonds_3), dtype=int)
    bonds_3_atom3 = np.empty(len(bonds_3), dtype=int)
    bonds_3_equilibrium = np.empty(len(bonds_3), dtype=np.float64)
    bonds_3_stength = np.empty(len(bonds_3), dtype=np.float64)
    for i, b in enumerate(bonds_3):
        bonds_3_atom1[i] = b[0]
        bonds_3_atom2[i] = b[1]
        bonds_3_atom3[i] = b[2]
        bonds_3_equilibrium[i] = b[3]
        bonds_3_stength[i] = b[4]
    E = compute_angle_forces__plain(f_angles, positions, bonds_3, box_size)

    E = compute_angle_forces__numba(
        f_angles, positions, box_size, bonds_3_atom1, bonds_3_atom2,
        bonds_3_equilibrium, bonds_3_stength
    )

    E = compute_angle_forces__numba_fastmath(
        f_angles, positions, box_size, bonds_3_atom1, bonds_3_atom2,
        bonds_3_equilibrium, bonds_3_stength
    )

    f_angles_fortran = np.asfortranarray(f_angles, dtype=np.float32)
    r_fortran = np.asfortranarray(positions, dtype=np.float32)
    E = compute_bond_forces__fortran(
        f_angles_fortran, r_fortran, box_size, bonds_3_atom1, bonds_3_atom2,
        bonds_3_atom3, bonds_3_equilibrium, bonds_3_stength
    )
