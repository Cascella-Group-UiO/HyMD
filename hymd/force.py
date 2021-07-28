import numba
import numpy as np
import networkx as nx
from dataclasses import dataclass
from compute_bond_forces import cbf as compute_bond_forces__fortran  # noqa: F401, E501
from compute_angle_forces import (
    caf as compute_angle_forces__fortran,
)  # noqa: F401, E501
from compute_dihedral_forces import (
    cdf as compute_dihedral_forces__fortran,
)  # noqa: F401, E501
from compute_bond_forces__double import (
    cbf as compute_bond_forces__fortran__double,
)  # noqa: F401, E501
from compute_angle_forces__double import (
    caf as compute_angle_forces__fortran__double,
)  # noqa: F401, E501
from compute_dihedral_forces__double import (
    cdf as compute_dihedral_forces__fortran__double,
)


@dataclass
class Bond:
    atom_1: str
    atom_2: str
    equilibrium: float
    strength: float


@dataclass
class Angle(Bond):
    atom_3: str
    # Specify angle type in config.toml
    # Set to 1 if backbone angle
    # Probably useless since we will most likely use
    # the CBT potential
    # That means the type should be set for the dihedrals
    # and no angle (3-body interactions) should be set for BB atoms.
    # Type would then also be used for improper dihedrals.
    type: int


# 2- Harmonic potential for impropers
# 1- Fourier series
@dataclass
class Dihedral:
    atom_1: str
    atom_2: str
    atom_3: str
    atom_4: str
    coeff: list
    phase: list
    # type: (0) Fourier or (1) CBT, (2) impropers
    type: int


# 3- Combined bending-torsional potential
@dataclass
class CBT_potential(Dihedral):
    coeff_k: list
    phase_k: list


@dataclass
class Chi:
    atom_1: str
    atom_2: str
    interaction_energy: float


def prepare_bonds_old(molecules, names, bonds, indices, config):
    bonds_2 = []
    bonds_3 = []
    bonds_4 = []
    different_molecules = np.unique(molecules)
    for mol in different_molecules:
        bond_graph = nx.Graph()

        # Won't this be super slow for big systems?
        for local_index, global_index in enumerate(indices):
            if molecules[local_index] != mol:
                continue

            bond_graph.add_node(
                global_index,
                name=names[local_index].decode("UTF-8"),
                local_index=local_index,
            )
            for bond in [b for b in bonds[local_index] if b != -1]:
                bond_graph.add_edge(global_index, bond)

        connectivity = nx.all_pairs_shortest_path(bond_graph)
        for c in connectivity:
            i = c[0]
            connected = c[1]
            for j, path in connected.items():
                if len(path) == 2 and path[-1] > path[0]:
                    name_i = bond_graph.nodes()[i]["name"]
                    name_j = bond_graph.nodes()[j]["name"]

                    for b in config.bonds:
                        match_forward = name_i == b.atom_1 and name_j == b.atom_2
                        match_backward = name_j == b.atom_2 and name_i == b.atom_1
                        if match_forward or match_backward:
                            bonds_2.append(
                                [
                                    bond_graph.nodes()[i]["local_index"],
                                    bond_graph.nodes()[j]["local_index"],
                                    b.equilibrium,
                                    b.strength,
                                ]
                            )

                if len(path) == 3 and path[-1] > path[0]:
                    name_i = bond_graph.nodes()[i]["name"]
                    name_mid = bond_graph.nodes()[path[1]]["name"]
                    name_j = bond_graph.nodes()[j]["name"]

                    for a in config.angle_bonds:
                        match_forward = (
                            name_i == a.atom_1
                            and name_mid == a.atom_2
                            and name_j == a.atom_3
                        )
                        match_backward = (
                            name_i == a.atom_3
                            and name_mid == a.atom_2
                            and name_j == a.atom_1
                        )
                        if match_forward or match_backward:
                            bonds_3.append(
                                [
                                    bond_graph.nodes()[i]["local_index"],
                                    bond_graph.nodes()[path[1]]["local_index"],
                                    bond_graph.nodes()[j]["local_index"],
                                    np.radians(a.equilibrium),
                                    a.strength,
                                    a.type,
                                ]
                            )
                if len(path) == 4 and path[-1] > path[0]:
                    name_i = bond_graph.nodes()[i]["name"]
                    name_mid_1 = bond_graph.nodes()[path[1]]["name"]
                    name_mid_2 = bond_graph.nodes()[path[2]]["name"]
                    name_j = bond_graph.nodes()[j]["name"]

                    for a in config.dihedrals:
                        match_forward = (
                            name_i == a.atom_1
                            and name_mid_1 == a.atom_2
                            and name_mid_2 == a.atom_3
                            and name_j == a.atom_4
                        )
                        match_backward = (
                            name_i == a.atom_4
                            and name_mid_1 == a.atom_3
                            and name_mid_2 == a.atom_2
                            and name_j == a.atom_1
                        )
                        if match_forward or match_backward:
                            bonds_4.append(
                                [
                                    bond_graph.nodes()[i]["local_index"],
                                    bond_graph.nodes()[path[1]]["local_index"],
                                    bond_graph.nodes()[path[2]]["local_index"],
                                    bond_graph.nodes()[j]["local_index"],
                                    a.coeff,
                                    a.phase,
                                ]
                            )

    return bonds_2, bonds_3, bonds_4


def prepare_bonds(molecules, names, bonds, indices, config):
    bonds_2, bonds_3, bonds_4 = prepare_bonds_old(
        molecules, names, bonds, indices, config
    )
    # Bonds
    bonds_2_atom1 = np.empty(len(bonds_2), dtype=int)
    bonds_2_atom2 = np.empty(len(bonds_2), dtype=int)
    bonds_2_equilibrium = np.empty(len(bonds_2), dtype=np.float64)
    bonds_2_stength = np.empty(len(bonds_2), dtype=np.float64)
    for i, b in enumerate(bonds_2):
        bonds_2_atom1[i] = b[0]
        bonds_2_atom2[i] = b[1]
        bonds_2_equilibrium[i] = b[2]
        bonds_2_stength[i] = b[3]
    # Angles
    bonds_3_atom1 = np.empty(len(bonds_3), dtype=int)
    bonds_3_atom2 = np.empty(len(bonds_3), dtype=int)
    bonds_3_atom3 = np.empty(len(bonds_3), dtype=int)
    bonds_3_equilibrium = np.empty(len(bonds_3), dtype=np.float64)
    bonds_3_stength = np.empty(len(bonds_3), dtype=np.float64)
    bonds_3_type = np.empty(len(bonds_3), dtype=np.float64)
    for i, b in enumerate(bonds_3):
        bonds_3_atom1[i] = b[0]
        bonds_3_atom2[i] = b[1]
        bonds_3_atom3[i] = b[2]
        bonds_3_equilibrium[i] = b[3]
        bonds_3_stength[i] = b[4]
        bonds_3_type[i] = b[5]
    # Dihedrals
    bonds_4_atom1 = np.empty(len(bonds_4), dtype=int)
    bonds_4_atom2 = np.empty(len(bonds_4), dtype=int)
    bonds_4_atom3 = np.empty(len(bonds_4), dtype=int)
    bonds_4_atom4 = np.empty(len(bonds_4), dtype=int)
    bonds_4_coeff = np.empty((len(bonds_4), 5), dtype=np.float64)
    bonds_4_phase = np.empty((len(bonds_4), 5), dtype=np.float64)
    for i, b in enumerate(bonds_4):
        bonds_4_atom1[i] = b[0]
        bonds_4_atom2[i] = b[1]
        bonds_4_atom3[i] = b[2]
        bonds_4_atom4[i] = b[3]
        bonds_4_coeff[i] = b[4]
        bonds_4_phase[i] = b[5]

    return (
        bonds_2_atom1,
        bonds_2_atom2,
        bonds_2_equilibrium,
        bonds_2_stength,
        bonds_3_atom1,
        bonds_3_atom2,
        bonds_3_atom3,
        bonds_3_equilibrium,
        bonds_3_stength,
        bonds_3_type,
        bonds_4_atom1,
        bonds_4_atom2,
        bonds_4_atom3,
        bonds_4_atom4,
        bonds_4_coeff,
        bonds_4_phase,
    )


@numba.jit(nopython=True, fastmath=True)
def compute_bond_forces__numba(
    f_bonds,
    r,
    box_size,
    bonds_2_atom1,
    bonds_2_atom2,
    bonds_2_equilibrium,
    bonds_2_stength,
):
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

        energy += 0.5 * k * (dr - r0) ** 2
    return energy


def compute_bond_forces__plain(f_bonds, r, bonds_2, box_size):
    f_bonds.fill(0.0)
    energy = 0.0

    for i, j, r0, k in bonds_2:
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

        energy += 0.5 * k * (dr - r0) ** 2
    return energy


@numba.jit(nopython=True, fastmath=True)
def compute_angle_forces__numba(
    f_angles,
    r,
    box_size,
    bonds_3_atom1,
    bonds_3_atom2,
    bonds_3_atom3,
    bonds_3_equilibrium,
    bonds_3_stength,
):
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
        xsinph = 1.0 / np.sqrt(1.0 - cosphi ** 2)

        d = theta - theta0
        f = -k * d

        xrasin = xra * xsinph * f
        xrcsin = xrc * xsinph * f

        fa = (ea * cosphi - ec) * xrasin
        fc = (ec * cosphi - ea) * xrcsin

        f_angles[a, :] += fa
        f_angles[c, :] += fc
        f_angles[b, :] += -(fa + fc)

        energy -= 0.5 * f * d

    return energy


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
        xsinph = 1.0 / np.sqrt(1.0 - cosphi ** 2)

        d = theta - theta0
        f = -k * d

        xrasin = xra * xsinph * f
        xrcsin = xrc * xsinph * f

        fa = (ea * cosphi - ec) * xrasin
        fc = (ec * cosphi - ea) * xrcsin

        f_angles[a, :] += fa
        f_angles[c, :] += fc
        f_angles[b, :] -= fa + fc
        # f_angles[b, :] += -(fa + fc)

        energy -= 0.5 * f * d

    return energy


def compute_dihedral_forces__plain(f_dihedrals, r, bonds_4, box_size):
    f_dihedrals.fill(0.0)
    energy = 0.0

    for a, b, c, d, coeff, phase in bonds_4:
        f = r[a, :] - r[b, :]
        g = r[b, :] - r[c, :]
        h = r[d, :] - r[c, :]

        for dim in range(3):
            f -= box_size[dim] * np.around(f[dim] / box_size[dim])
            g -= box_size[dim] * np.around(g[dim] / box_size[dim])
            h -= box_size[dim] * np.around(h[dim] / box_size[dim])

        v = np.cross(f, g)
        w = np.cross(h, g)
        vv = np.dot(v, v)
        ww = np.dot(w, w)
        gn = np.linalg.norm(g)

        cosphi = np.dot(v, w)
        sinphi = np.dot(np.cross(v, w), g) / gn
        phi = np.arctan2(sinphi, cosphi)

        fg = np.dot(f, g)
        hg = np.dot(h, g)
        sc = v * fg / (vv * gn) - w * hg / (ww * gn)

        df = 0

        for m in range(len(coeff)):
            energy += coeff[m] * (1 + np.cos(m * phi + phase[m]))
            df += m * coeff[m] * np.sin(m * phi + phase[m])

        force_on_a = df * gn * v / vv
        force_on_d = df * gn * w / ww

        f_dihedrals[a, :] -= force_on_a
        f_dihedrals[b, :] += df * sc + force_on_a
        f_dihedrals[c, :] -= df * sc + force_on_d
        f_dihedrals[d, :] += force_on_d
    return energy


def dipole_forces_redistribution(f_dipoles, f_elec, trans, a, b, c, angle_type):
    """Redistribute electrostatic forces calculated from ghost dipole point charges
    to the backcone atoms of the protein."""

    pairs = [f_elec[i : i + 2] for i in range(0, len(f_elec), 2)]
    matrices = [trans[i : i + 3] for i in range(0, len(trans), 3)]

    for i, j, k, force, matrix, type in zip(a, b, c, pairs, matrices, angle_type):
        if type == 1:
            # Positive dipole charge
            f_dipoles[i] += matrix[0] @ force[0]  # Atom A
            f_dipoles[j] += matrix[1] @ force[0]  # Atom B
            f_dipoles[k] += matrix[2] @ force[0]  # Atom C

            # Negative dipole charge
            f_dipoles[i] += matrix[0] @ force[1]  # Atom A
            f_dipoles[j] += matrix[1] @ force[1]  # Atom B
            f_dipoles[k] += matrix[2] @ force[1]  # Atom C
