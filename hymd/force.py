import numpy as np
import networkx as nx
from dataclasses import dataclass

# Imported here so we can call from force import compute_bond_forces__fortran
from force_kernels import (  # noqa: F401
    cbf as compute_bond_forces__fortran
)
from force_kernels import (  # noqa: F401
    caf as compute_angle_forces__fortran,
)
from force_kernels import (  # noqa: F401
    cdf as compute_dihedral_forces__fortran,
)
from force_kernels import (  # noqa: F401
    cbf_d as compute_bond_forces__fortran__double,
)
from force_kernels import (  # noqa: F401
    caf_d as compute_angle_forces__fortran__double,
)
from force_kernels import (  # noqa: F401
    cdf_d as compute_dihedral_forces__fortran__double,
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


@dataclass
class Dihedral:
    atom_1: str
    atom_2: str
    atom_3: str
    atom_4: str
    coeffs: np.ndarray
    dih_type: int


@dataclass
class Chi:
    atom_1: str
    atom_2: str
    interaction_energy: float


def findPathsNoLC(G, u, n):
    if n == 0:
        return [[u]]
    paths = []
    for neighbor in G.neighbors(u):
        for path in findPathsNoLC(G, neighbor, n - 1):
            if u not in path:
                paths.append([u] + path)
    return paths


def prepare_bonds_old(molecules, names, bonds, indices, config):
    bonds_2 = []
    bonds_3 = []
    bonds_4 = []
    bb_index = []

    different_molecules = np.unique(molecules)
    for mol in different_molecules:
        bb_dihedral = 0
        bond_graph = nx.Graph()

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
                        match_forward = (
                            name_i == b.atom_1 and name_j == b.atom_2
                        )
                        match_backward = (
                            name_i == b.atom_2 and name_j == b.atom_1
                        )
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
                                ]
                            )

            all_paths_len_four = findPathsNoLC(bond_graph, i, 3)
            for p in all_paths_len_four:
                name_i = bond_graph.nodes()[i]["name"]
                name_mid_1 = bond_graph.nodes()[p[1]]["name"]
                name_mid_2 = bond_graph.nodes()[p[2]]["name"]
                name_j = bond_graph.nodes()[p[3]]["name"]

                for a in config.dihedrals:
                    match_forward = (
                        name_i == a.atom_1
                        and name_mid_1 == a.atom_2
                        and name_mid_2 == a.atom_3
                        and name_j == a.atom_4
                    )
                    if (
                        match_forward
                        and [
                            bond_graph.nodes()[p[3]]["local_index"],
                            bond_graph.nodes()[p[2]]["local_index"],
                            bond_graph.nodes()[p[1]]["local_index"],
                            bond_graph.nodes()[i]["local_index"],
                            a.coeffs,
                            a.dih_type,
                        ]
                        not in bonds_4
                    ):
                        bonds_4.append(
                            [
                                bond_graph.nodes()[i]["local_index"],
                                bond_graph.nodes()[p[1]]["local_index"],
                                bond_graph.nodes()[p[2]]["local_index"],
                                bond_graph.nodes()[p[3]]["local_index"],
                                a.coeffs,
                                a.dih_type,
                            ]
                        )
                        if a.dih_type == 1:
                            bb_dihedral = len(bonds_4)

        if bb_dihedral:
            bb_index.append(bb_dihedral - 1)

    return bonds_2, bonds_3, bonds_4, bb_index


def prepare_bonds(molecules, names, bonds, indices, config):
    bonds_2, bonds_3, bonds_4, bb_index = prepare_bonds_old(
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
    for i, b in enumerate(bonds_3):
        bonds_3_atom1[i] = b[0]
        bonds_3_atom2[i] = b[1]
        bonds_3_atom3[i] = b[2]
        bonds_3_equilibrium[i] = b[3]
        bonds_3_stength[i] = b[4]

    # Dihedrals
    bonds_4_atom1 = np.empty(len(bonds_4), dtype=int)
    bonds_4_atom2 = np.empty(len(bonds_4), dtype=int)
    bonds_4_atom3 = np.empty(len(bonds_4), dtype=int)
    bonds_4_atom4 = np.empty(len(bonds_4), dtype=int)
    number_of_coeff = 6
    len_of_coeff = 5
    bonds_4_coeff = np.empty(
        (len(bonds_4), number_of_coeff, len_of_coeff), dtype=np.float64
    )
    bonds_4_type = np.empty(len(bonds_4), dtype=int)
    bonds_4_last = np.zeros(len(bonds_4), dtype=int)
    for i, b in enumerate(bonds_4):
        bonds_4_atom1[i] = b[0]
        bonds_4_atom2[i] = b[1]
        bonds_4_atom3[i] = b[2]
        bonds_4_atom4[i] = b[3]
        bonds_4_coeff[i] = np.resize(b[4], (number_of_coeff, len_of_coeff))
        bonds_4_type[i] = b[5]
    bonds_4_last[bb_index] = 1

    return (
        bonds_2_atom1, bonds_2_atom2, bonds_2_equilibrium, bonds_2_stength,
        bonds_3_atom1, bonds_3_atom2, bonds_3_atom3, bonds_3_equilibrium, bonds_3_stength,  # noqa: E501
        bonds_4_atom1, bonds_4_atom2, bonds_4_atom3, bonds_4_atom4, bonds_4_coeff, bonds_4_type, bonds_4_last,  # noqa: E501
    )


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

    for a, b, c, d, coeffs, phase in bonds_4:
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

        for m in range(len(coeffs[0])):
            energy += coeffs[0][m] * (1 + np.cos(m * phi - coeffs[1][m]))
            df += m * coeffs[0][m] * np.sin(m * phi - coeffs[1][m])

        force_on_a = df * gn * v / vv
        force_on_d = df * gn * w / ww

        f_dihedrals[a, :] -= force_on_a
        f_dihedrals[b, :] += df * sc + force_on_a
        f_dihedrals[c, :] -= df * sc + force_on_d
        f_dihedrals[d, :] += force_on_d
    return energy


def dipole_forces_redistribution(
    f_on_bead, f_dipoles, trans_matrices, a, b, c, d, type_array, last_bb
):
    """Redistribute electrostatic forces calculated from ghost dipole point
    charges to the backcone atoms of the protein."""

    f_on_bead.fill(0.0)
    for i, j, k, l, fd, matrix, dih_type, is_last in zip(
        a, b, c, d, f_dipoles, trans_matrices, type_array, last_bb
    ):
        if dih_type == 1:
            tot_force = fd[0] + fd[1]
            f_on_bead[i] += matrix[0] @ tot_force  # Atom A
            f_on_bead[j] += matrix[1] @ tot_force + 0.5 * tot_force  # Atom B
            f_on_bead[k] += matrix[2] @ tot_force + 0.5 * tot_force  # Atom C

            if is_last == 1:
                tot_force = fd[2] + fd[3]

                # Atom B
                f_on_bead[j] += matrix[3] @ tot_force

                # Atom C
                f_on_bead[k] += matrix[4] @ tot_force + 0.5 * tot_force

                # Atom D
                f_on_bead[l] += matrix[5] @ tot_force + 0.5 * tot_force
