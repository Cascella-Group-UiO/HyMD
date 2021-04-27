import numba
import numpy as np
import networkx as nx
from dataclasses import dataclass
from compute_bond_forces import cbf as compute_bond_forces__fortran  # noqa: F401, E501
from compute_angle_forces import (
    caf as compute_angle_forces__fortran,
)  # noqa: F401, E501
from compute_bond_forces__double import (
    cbf as compute_bond_forces__fortran__double,
)  # noqa: F401, E501
from compute_angle_forces__double import (
    caf as compute_angle_forces__fortran__double,
)  # noqa: F401, E501


@dataclass
class Bond:
    atom_1: str
    atom_2: str
    equilibrium: float
    strength: float


@dataclass
class Angle(Bond):
    atom_3: str


# 1- Sinmple fourier function potential
# 2- Harmonic potential for impropers
# 3- Combined bending-torsional potential
@dataclass
class Dihedral:
    atom_1: str
    atom_2: str
    atom_3: str
    atom_4: str
    cn: list
    dn: list


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
                                    a.cn,
                                    a.dn,
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
    bonds_4_cn = np.empty(len(bonds_4), dtype=np.float64)
    bonds_4_dn = np.empty(len(bonds_4), dtype=np.float64)
    for i, b in enumerate(bonds_4):
        bonds_4_atom1[i] = b[0]
        bonds_4_atom2[i] = b[1]
        bonds_4_atom3[i] = b[2]
        bonds_4_atom4[i] = b[3]
        bonds_4_cn[i] = b[4]
        bonds_4_dn[i] = b[5]

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
        bonds_4_atom1,
        bonds_4_atom2,
        bonds_4_atom3,
        bonds_4_atom4,
        bonds_4_cn,
        bonds_4_dn,
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

    for a, b, c, d, cm, dm in bonds_4:
        f = r[a, :] - r[b, :]
        g = r[b, :] - r[c, :]
        h = r[d, :] - r[c, :]

        for dim in range(3):
            f -= box_size[dim] * np.around(f[dim] / box_size[dim])
            g -= box_size[dim] * np.around(g[dim] / box_size[dim])
            h -= box_size[dim] * np.around(h[dim] / box_size[dim])

        gn = np.linalg.norm(g)
        gv = g / gn

        v = f - np.dot(f, gv) * gv
        w = h - np.dot(h, gv) * gv

        cos_phi = np.dot(v, w)
        sin_phi = np.dot(np.cross(gv, v), w)
        phi = np.arctan2(sin_phi, cos_phi)

        # Let's reuse the same variables we don't needa anymore after
        # getting the angle (don't think this is good practice though)
        v = np.cross(f, g)
        w = np.cross(h, g)

        vv = np.dot(a, a)
        ww = np.dot(b, b)

        fg = np.dot(f, g)
        hg = np.dot(h, g)
        s = a * fg / (vv * gn) - w * hg / (ww * gn)

        df = 0
        # Use 8 terms => len(cn) == len(dn) == 8
        for m in range(8):
            energy += cm[m] * (1 + np.cos(m * phi + dm[m]))
            df += m * cm[m] * np.sin(m * phi + dm[m])

        # force_on_a = -df * gn * v / v2
        # f_dihedrals[a, :] += force_on_a
        # f_dihedrals[b, :] += df * s - force_on_a
        # f_dihedrals[c, :] += -(df * s + force_on_d)

        force_on_a = df * gn * v / vv
        force_on_d = df * gn * w / ww

        f_dihedrals[a, :] -= force_on_a
        f_dihedrals[b, :] += df * s + force_on_a
        f_dihedrals[c, :] -= df * s + force_on_d
        f_dihedrals[d, :] += force_on_d

    return energy
