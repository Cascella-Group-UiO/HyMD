"""Calculates intramolecular forces between bonded particles in molecules
"""
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
    """Dataclass representing a single two-particle bond type

    A `bond type` is a bond strength and equilibrium distance associated with
    any bond between particles of specific types :code:`A` and :code:`B`
    (where :code:`A` and :code:`B` may be the same or different). Harmonic
    two-particle bonds in HyMD take the form

    .. math::

        V_2(\\mathbf{r}_1, \\mathbf{r}_2) =
            \\frac{1}{2}k
            \\left(
                \\vert \\mathbf{r}_1 - \\mathbf{r}_2 \\vert - r_0
            \\right)^2,

    where :math:`k` is the bond strength (spring constant) and :math:`r_0` is
    the equilibrium bond length (at which the energy is zero).

    Attributes
    ----------
    atom_1 : str
        Type name of particle 1.
    atom_2 : str
        Type name of particle 2.
    equilibrium : float
        Equilibrium distance at which the energy associated with the bond
        vanishes and the resulting force is zero.
    strength : float
        Harmonic bond strength coefficient (spring constant).
    """
    atom_1: str
    atom_2: str
    equilibrium: float
    strength: float


@dataclass
class Angle(Bond):
    """Dataclass representing a single three-particle bond type

    A `bond type` is a bond strength and equilibrium angle associated with
    any three-particle bond between particles of specific types :code:`A`,
    :code:`B`, and :code:`C` (where :code:`A`, :code:`B`, and :code:`C` may be
    different or the same). Harmonic angular three-particle bonds in HyMD take
    the form

    .. math::

        V_3(\\mathbf{r}_1, \\mathbf{r}_2, \\mathbf{r}_3) =
            \\frac{1}{2}k
            \\left(
                \\cos^{-1}
                \\left[
                    \\frac{
                        (\\mathbf{r}_1-\\mathbf{r}_2)
                        \\cdot
                        (\\mathbf{r}_3-\\mathbf{r}_2)
                    }
                    {
                        \\vert\\mathbf{r}_1-\\mathbf{r}_2\\vert
                        \\vert\\mathbf{r}_3-\\mathbf{r}_2\\vert
                    }
                \\right] - \\theta_0
            \\right)^2

    Attributes
    ----------
    atom_1 : str
        Type name of particle 1.
    atom_2 : str
        Type name of particle 2.
    atom_3 : str
        Type name of particle 3.
    equilibrium : float
        Equilibrium angle at which the energy associated with the
        three-particle angular bond vanishes and the resulting force is zero.
    strength : float
        Harmonic bond strength coefficient.

    See also
    --------
    Bond :
        Two-particle bond type dataclass
    """
    atom_3: str


@dataclass
class Dihedral:
    """Dataclass representing a single four-particle bond type

    A `bond type` is a bond strength and equilibrium angle associated with
    any four-particle torsional bond between particles of specific types
    :code:`A`, :code:`B`, :code:`C`, and :code:`D` (where :code:`A`, :code:`B`,
    :code:`C`, and :code:`D` may be different or the same). Dihedral
    four-particle bonds in HyMD take different forms depending on the
    :code:`dih_type` parameter.

    In the following, let :math:`\\phi` denote the angle between the planes
    spanned by the relative positions of atoms :code:`A`-:code:`B`-:code:`C`
    and :code:`B`-:code:`C`-:code:`D`. If :code:`dih_type = 0`, then

    .. math::

        V_4(\\phi) = \\sum_n c_n
            \\left(
                1 + \\cos\\left[
                    n\\phi - d_n
                \\right]
            \\right),

    where :math:`c_n` are energy coefficients and :math:`d_n` are propensity
    phases. By default, the cosine sum is truncated at five terms. If
    :code:`coeffs` provides only a single float, this is used as the coiling
    propensity parameter :math:`\\lambda`. In this case, :math:`c_n` and
    :math:`d_n` are automatically set to values which promote alpha helical
    (:math:`\\lambda=-1`), beta sheet (:math:`\\lambda=1`), or a mixed
    (:math:`1>\\lambda>-1`) structure (using values provided by Bore et al.
    (2018)). In this case, the full potential takes the form

    .. math::

        V_4(\\phi;\\lambda) =
            \\frac{1}{2}(1-\\lambda) V_{\\text{prop},\\alpha}(\\phi)
            +
            \\frac{1}{2}(1+\\lambda) V_{\\text{prop},\\beta}(\\phi)
            +
            (1-\\vert\\lambda\\vert) V_{\\text{prop}, \\text{coil}}(\\phi),

    with each of the :math:`V_{\\mathrm{prop}, X}` being fixed cosine series
    potentials with pre-set :math:`c_n` -s and :math:`d_n` -s.

    If :code:`dih_type = 1`, then a combined bending torsional (CBT) potential
    is employed,

    .. math::

        V_4(\\phi,\\gamma;\\lambda) =
            V_4(\\phi;\\lambda)
            +
            \\frac{1}{2}K(\\phi)(\\gamma - \\gamma_0)^2,

    where :math:`V_4(\\phi;\\lambda)` specifies the potential as given by the
    coiling propensity parameter above, :math:`K(\\phi)` is a separate cosine
    series potential acting as the angular three-particle bond strength, and
    :math:`\\gamma_0` is the three-particle bond equilibrium angle. In this
    case, the :code:`coeffs` parameter must specify *both* a :math:`\\lambda`
    value, in additon to the energy and phases dictating the :math:`K(\\phi)`
    potential.

    Attributes
    ----------
    atom_1 : str
        Type name of particle 1.
    atom_2 : str
        Type name of particle 2.
    atom_3 : str
        Type name of particle 3.
    atom_4 : str
        Type name of particle 4.
    coeffs : list[list[float]] or list[float] or numpy.ndarray
        Dihedral coefficients defining the series expansion of the dihedral
        energy.
    dih_type : int
        Specifies the type of dihedral used; If :code:`0`, then :code:`coeffs`
        must contain **either** two lists of five energy coefficients
        :math:`c_n` and five propensity phases :math:`d_n` **or** a single
        floating point number defining the :math:`\\lambda` coiling propensity
        parameter. If :code:`1`, the combined bending-torsional potential is
        used, and :code:`coeffs` must specify :math:`\\lambda` *and* two lists
        containing energy coefficients and propensity phases for the
        :math:`V_\\text{prop}` propensity potential, defined in terms of cosine
        series.

    References
    ----------
    Bore et al. J. Chem. Theory Comput., 14(2): 1120–1130, 2018.
    """
    atom_1: str
    atom_2: str
    atom_3: str
    atom_4: str
    coeffs: np.ndarray
    dih_type: int


@dataclass
class Chi:
    """Dataclass representing a single :math:`\\chi` mixing interaction type

    An `interaction mixing energy type` is a mixing energy associated with
    density overlap between species of types :code:`A` and :code:`B`
    (specified as inputs :code:`atom_1` and :code:`atom_2`). A positive mixing
    energy promotes phase separation, a negative mixing energy promotes mixing.
    The interaction energy density (provided the :code:`DefaultWithChi`
    Hamiltonian is used) takes the form

    .. math::

        w[\\tilde\\phi] = \\frac{1}{2\\kappa}
            \\sum_{k,l}\\chi_{k,l} \\tilde\\phi_k\\tilde\\phi_l,

    where :math:`\\chi_{k,l}` denotes the mixing energy between species
    :math:`k` and :math:`l`, with :math:`\\kappa` being the incompressibility.
    The value of the interaction mixing energy parameter may be extracted from
    `Flory-Huggins theory`_.

    .. _`Flory-Huggins theory`:
        https://en.wikipedia.org/wiki/Flory%E2%80%93Huggins_solution_theory

    Attributes
    ----------
    atom_1 : str
        Type name of particle 1.
    atom_2 : str
        Type name of particle 2.
    interaction_energy : float
        Interaction mixing energy.

    See also
    --------
    hymd.hamiltonian.DefaultWithChi :
        Interaction energy functional using :math:`\\chi`-interactions.
    """
    atom_1: str
    atom_2: str
    interaction_energy: float


def find_all_paths(G, u, n):
    """Helper function that recursively finds all paths of given lenght 'n + 1' inside a network 'G'.
    Adapted from https://stackoverflow.com/a/28103735."""
    if n == 0:
        return [[u]]
    paths = []
    for neighbor in G.neighbors(u):
        for path in find_all_paths(G, neighbor, n - 1):
            if u not in path:
                paths.append([u] + path)
    return paths


def prepare_bonds_old(molecules, names, bonds, indices, config):
    """Find bonded interactions from connectivity and bond types information

    .. deprecated:: 1.0.0
        :code:`prepare_bonds_old` was replaced by :code:`prepare_bonds` for
        use with compiled Fortran kernels prior to 1.0.0 release.

    Prepares the necessary equilibrium and bond strength information needed by
    the intramolecular interaction functions. This is performed locally on each
    MPI rank, as the domain decomposition ensures that for all molecules *all*
    consituent particles are always contained on *the same* MPI rankself.

    This function traverses the bond connectivity information provided in the
    structure/topology input file and indentifies any two-, three-, or
    four-particle potential bonds. For each connected chain of two, three, or
    four particles, a matching to bond types is attempted. If the corresponding
    names match, a bond object is initialized.

    In order to investigate the connectivity, a graph is created using
    networkx functionality.

    Parameters
    ----------
    molecules : (N,) numpy.ndarray
        Array of integer molecule affiliation for each of :code:`N` particles.
        Global (across all MPI ranks) or local (local indices on this MPI rank
        only) may be used, both, without affecting the result.
    names : (N,) numpy.ndarray
        Array of type names for each of :code:`N` particles.
    bonds : (N,M) numpy.ndarray
        Array of :code:`M` bonds originating from each of :code:`N` particles.
    indices : (N,) numpy.ndarray
        Array of integer indices for each of :code:`N` particles. Global
        (across all MPI ranks) or local (local indices on this MPI rank only)
        may be used, both, without affecting the result.
    config : Config
        Configuration object.

    Returns
    -------
    bonds_2 : list
        List of lists containing *local* particle indices, equilibrium
        distance, and bond strength coefficient for each reconstructed
        two-particle bond.
    bonds_3 :
        List of lists containing *local* particle indices, equilibrium angle,
        and bond strength coefficient for each reconstructed three-particle
        bond.
    bonds_4 :
        List of lists containing *local* particle indices, dihedral type index,
        and dihedral coefficients for each reconstructed four-particle
        torsional bond.
    bb_index : list
        List indicating the dihedral type of each four-particle bond in
        :code:`bonds_4`.

    See also
    --------
    Bond :
        Two-particle bond type dataclass.
    Angle :
        Three-particle angular bond type dataclass.
    Dihedral :
        Four-particle torsional bond type dataclass.
    hymd.input_parser.Config
        Configuration dataclass handler.
    """
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

            all_paths_len_four = find_all_paths(bond_graph, i, 3)
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
    """Rearrange the bond information for usage in compiled Fortran kernels

    Restructures the lists resulting from the execution of
    :code:`prepare_bonds_old` into numpy arrays suitable for calls to optimized
    Fortran code calculating bonded forces and energiesself.

    Parameters
    ----------
    molecules : (N,) numpy.ndarray
        Array of integer molecule affiliation for each of :code:`N` particles.
        Global (across all MPI ranks) or local (local indices on this MPI rank
        only) may be used, both, without affecting the result.
    names : (N,) numpy.ndarray
        Array of type names for each of :code:`N` particles.
    bonds : (N,M) numpy.ndarray
        Array of :code:`M` bonds originating from each of :code:`N` particles.
    indices : (N,) numpy.ndarray
        Array of integer indices for each of :code:`N` particles. Global
        (across all MPI ranks) or local (local indices on this MPI rank only)
        may be used, both, without affecting the result.
    config : Config
        Configuration object.

    Returns
    -------
    bonds_2_atom_1 : (B,) numpy.ndarray
        Local index of particle 1 for each of :code:`B` constructed
        two-particle bonds.
    bonds_2_atom_2 : (B,) numpy.ndarray
        Local index of particle 2 for each of :code:`B` constructed
        two-particle bonds.
    bonds_2_equilibrium : (B,) numpy.ndarray
        Equilibrium distance for each of :code:`B` constructed two-particle
        bonds.
    bonds_2_strength : (B,) numpy.ndarray
        Bond strength for each of :code:`B` constructed two-particle
        bonds.
    bonds_3_atom_1 : (A,) numpy.ndarray
        Local index of particle 1 for each of :code:`A` constructed
        three-particle bonds.
    bonds_3_atom_2 : (A,) numpy.ndarray
        Local index of particle 2 for each of :code:`A` constructed
        three-particle bonds.
    bonds_3_atom_3 : (A,) numpy.ndarray
        Local index of particle 3 for each of :code:`A` constructed
        three-particle bonds.
    bonds_3_equilibrium : (A,) numpy.ndarray
        Equilibrium angle for each of :code:`A` constructed three-particle
        bonds.
    bonds_3_strength : (A,) numpy.ndarray
        Bond strength for each of :code:`A` constructed three-particle
        bonds.
    bonds_4_atom_1 : (D,) numpy.ndarray
        Local index of particle 1 for each of :code:`D` constructed
        four-particle bonds.
    bonds_4_atom_2 : (D,) numpy.ndarray
        Local index of particle 2 for each of :code:`D` constructed
        four-particle bonds.
    bonds_4_atom_3 : (D,) numpy.ndarray
        Local index of particle 3 for each of :code:`D` constructed
        four-particle bonds.
    bonds_4_atom_4 : (D,) numpy.ndarray
        Local index of particle 4 for each of :code:`D` constructed
        four-particle bonds.
    bonds_4_coeff : (D,) numpy.ndarray
        Cosine series coefficients for each of :code:`D` constructed
        four-particle bonds.
    bonds_4_type : (D,) numpy.ndarray
        Dihedral type for each of :code:`D` constructed four-particle
        bonds.
    bonds_4_last : (D,) numpy.ndarray
        Flags indicating if :code:`dih_type` is :code:`1` for each of :code:`D`
        constructed four-particle bonds.

    See also
    --------
    prepare_bonds_old :
        Used internally to reconstruct the bonded interactions types from the
        connectivity information in the structure/topology input file and the
        bonded types specified in the configuration file.
    """
    bonds_2, bonds_3, bonds_4, bb_index = prepare_bonds_old(
        molecules, names, bonds, indices, config
    )

    # Bonds
    bonds_2_atom1 = np.empty(len(bonds_2), dtype=int)
    bonds_2_atom2 = np.empty(len(bonds_2), dtype=int)
    bonds_2_equilibrium = np.empty(len(bonds_2), dtype=np.float64)
    bonds_2_strength = np.empty(len(bonds_2), dtype=np.float64)
    for i, b in enumerate(bonds_2):
        bonds_2_atom1[i] = b[0]
        bonds_2_atom2[i] = b[1]
        bonds_2_equilibrium[i] = b[2]
        bonds_2_strength[i] = b[3]

    # Angles
    bonds_3_atom1 = np.empty(len(bonds_3), dtype=int)
    bonds_3_atom2 = np.empty(len(bonds_3), dtype=int)
    bonds_3_atom3 = np.empty(len(bonds_3), dtype=int)
    bonds_3_equilibrium = np.empty(len(bonds_3), dtype=np.float64)
    bonds_3_strength = np.empty(len(bonds_3), dtype=np.float64)
    for i, b in enumerate(bonds_3):
        bonds_3_atom1[i] = b[0]
        bonds_3_atom2[i] = b[1]
        bonds_3_atom3[i] = b[2]
        bonds_3_equilibrium[i] = b[3]
        bonds_3_strength[i] = b[4]

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
        bonds_2_atom1, bonds_2_atom2, bonds_2_equilibrium, bonds_2_strength,
        bonds_3_atom1, bonds_3_atom2, bonds_3_atom3, bonds_3_equilibrium, bonds_3_strength,  # noqa: E501
        bonds_4_atom1, bonds_4_atom2, bonds_4_atom3, bonds_4_atom4, bonds_4_coeff, bonds_4_type, bonds_4_last,  # noqa: E501
    )


def compute_bond_forces__plain(f_bonds, r, bonds_2, box_size):
    """Computes forces resulting from bonded interactions

    .. deprecated:: 1.0.0
        :code:`compute_bond_forces__plain` was replaced by compiled Fortran
        code prior to 1.0.0 release.
    """
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
    """Computes forces resulting from angular interactions

    .. deprecated:: 1.0.0
        :code:`compute_angle_forces__plain` was replaced by compiled Fortran
        code prior to 1.0.0 release.
    """
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
    """Computes forces resulting from dihedral interactions

    .. deprecated:: 1.0.0
        :code:`compute_dihedral_forces__plain` was replaced by compiled Fortran
        code prior to 1.0.0 release.
    """
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
    """Redistribute electrostatic forces calculated from topologically 
    reconstructed ghost dipole point charges to the backcone atoms of the protein.
    """
    f_on_bead.fill(0.0)
    for i, j, k, l, fd, matrix, dih_type, is_last in zip(
        a, b, c, d, f_dipoles, trans_matrices, type_array, last_bb
    ):
        if dih_type ==1:
            sum_force = fd[0] + fd[1]
            diff_force = fd[0] - fd[1]
            f_on_bead[i] += matrix[0] @ diff_force  # Atom A
            f_on_bead[j] += matrix[1] @ diff_force + 0.5 * sum_force  # Atom B
            f_on_bead[k] += matrix[2] @ diff_force + 0.5 * sum_force  # Atom C

            if is_last == 1:
                sum_force = fd[2] + fd[3]
                diff_force = fd[2] - fd[3]
                f_on_bead[j] += matrix[3] @ diff_force
                f_on_bead[k] += matrix[4] @ diff_force + 0.5 * sum_force
                f_on_bead[l] += matrix[5] @ diff_force + 0.5 * sum_force
