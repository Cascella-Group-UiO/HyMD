import numpy as np
import networkx as nx
from dataclasses import dataclass


@dataclass
class Bond:
    atom_1: str
    atom_2: str
    equilibrium: float
    strenght: float


@dataclass
class Angle(Bond):
    atom_3: str


@dataclass
class Chi:
    atom_1: str
    atom_2: str
    interaction_energy: float


def prepare_bonds(molecules, names, bonds, indices, config):
    bonds_2 = []
    bonds_3 = []
    different_molecules = np.unique(molecules)
    for mol in different_molecules:
        bond_graph = nx.Graph()
        for local_index, global_index in enumerate(indices):
            if molecules[local_index] != mol:
                continue

            bond_graph.add_node(global_index,
                                name=names[local_index].decode('UTF-8'),
                                local_index=local_index)
            for bond in [b for b in bonds[local_index] if b != -1]:
                bond_graph.add_edge(global_index, bond)

        connectivity = nx.all_pairs_shortest_path(bond_graph)
        for c in connectivity:
            i = c[0]
            connected = c[1]
            for j, path in connected.items():
                if len(path) == 2 and path[-1] > path[0]:
                    name_i = bond_graph.nodes()[i]['name']
                    name_j = bond_graph.nodes()[j]['name']

                    for b in config.bonds:
                        match_forward = (name_i == b.atom_1 and
                                         name_j == b.atom_2)
                        match_backward = (name_j == b.atom_2 and
                                          name_i == b.atom_1)
                        if match_forward or match_backward:
                            bonds_2.append([
                                bond_graph.nodes()[i]['local_index'],
                                bond_graph.nodes()[j]['local_index'],
                                b.equilibrium,
                                b.strenght
                            ])

                if len(path) == 3 and path[-1] > path[0]:
                    name_i = bond_graph.nodes()[i]['name']
                    name_mid = bond_graph.nodes()[path[1]]['name']
                    name_j = bond_graph.nodes()[j]['name']

                    for a in config.angle_bonds:
                        match_forward = (name_i == a.atom_1 and
                                         name_mid == a.atom_2 and
                                         name_j == a.atom_3)
                        match_backward = (name_i == a.atom_3 and
                                          name_mid == a.atom_2 and
                                          name_j == a.atom_1)
                        if match_forward or match_backward:
                            bonds_3.append([
                                bond_graph.nodes()[i]['local_index'],
                                bond_graph.nodes()[path[1]]['local_index'],
                                bond_graph.nodes()[j]['local_index'],
                                np.radians(a.equilibrium),
                                a.strenght
                            ])
    return bonds_2, bonds_3


def compute_bond_forces(f_bonds, r, bonds_2, box_size, comm):
    f_bonds.fill(0.0)
    energy = 0.0

    for i, j, r0, k in bonds_2:
        ri = r[i, :]
        rj = r[j, :]
        rij = rj - ri
        rij = np.squeeze(rij)

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


def compute_angle_forces(f_angles, r, bonds_3, box_size, comm):
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
