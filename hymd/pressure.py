import logging
import numpy as np
from mpi4py import MPI
from .logger import Logger
import sympy
from .field import comp_laplacian


def comp_pressure(
    phi,
    phi_q,
    psi,
    hamiltonian,
    velocities,
    config,
    phi_fourier,
    phi_laplacian,
    phi_transfer,
    positions,
    bond_pr,
    angle_pr,
    comm=MPI.COMM_WORLD,
):
    """
    Computes total internal pressure of the system.
    Kinetic pressure is trivially calculated from the kinetic energy.
    Already computed bond and angle pressure terms are inserted into the
    total internal pressure.
    The field pressure equation is implemented.

    Parameters
    ----------
    phi : list[pmesh.pm.RealField], (M,)
        Pmesh :code:`RealField` objects containing discretized particle number
        density values on the computational grid; one for each particle type
        :code:`M`. Pre-allocated, but empty; any values in this field are discarded.
        Changed in-place. Local for each MPI rank--the full computaional grid
        is represented by the collective fields of all MPI ranks.
    hamiltonian : Hamiltonian
        Particle-field interaction energy handler object. Defines the
        grid-independent filtering function, :math:`H`.
    velocities : (N, D) numpy.ndarray
        Array of velocities of N particles in D dimensions.
    config : Config
        Configuration dataclass containing simulation metadata and parameters.
    phi_fourier : list[pmesh.pm.ComplexField], (M,)
        Pmesh :code:`ComplexField` objects containing discretized particle
        number density values in reciprocal space on the computational grid;
        one for each particle type. Pre-allocated, but empty; any values in
        this field are discarded Changed in-place. Local for each MPI rank--the
        full computaional grid is represented by the collective fields of all
        MPI ranks.
    phi_laplacian : list[pmesh.pm.RealField], (M, 3)
        Like phi, but containing the laplacian of particle number densities.
    phi_transfer : list[pmesh.pm.ComplexField], (3,)
        Like phi_fourier, used as an intermediary to perform FFT operations
        to obtain the gradient or laplacian of particle number densities.
    positions : (N,D) numpy.ndarray
        Array of positions for :code:`N` particles in :code:`D` dimensions.
        Local for each MPI rank.
    bond_pr : (3,) numpy.ndarray
        Total bond pressure due all two-particle bonds.
    angle_pr : (3,) numpy.ndarray
        Total angle pressure due all three-particle bonds
    comm : MPI.Intracomm, optional
        MPI communicator to use for rank commuication. Defaults to
        MPI.COMM_WORLD.

    Returns
    -------
    pressure : (18,) numpy.ndarray
        Pressure contributions from various energy terms.
        0: due to kinetic energy
        1-5: due to field energy
        6-8: due to two-particle bonded terms
        9-11: due to three-particle bonded terms (called angle terms)
        12-14: due to four-particle bonded terms (called dihedral terms)
        (defaults to 0 currently. Yet to be implemented)
        15-17: total pressure in x,y,z directions.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells

    w_0 = hamiltonian.w_0(phi) * volume_per_cell
    w_elec = 0.0
    if (config.coulombtype == "PIC_Spectral_GPE") or (
        config.coulombtype == "PIC_Spectral"
    ):
        w_elec = hamiltonian.w_elec([phi_q, psi]) * volume_per_cell
    w = w_0 + w_elec

    # Kinetic term
    kinetic_energy = 0.5 * config.mass * np.sum(velocities**2)
    p_kin = 2 / (3 * V) * kinetic_energy

    # Term 1
    p0 = -1 / V * np.sum(w)

    # Term 2
    if psi is not None: # if using electrostatics
        V_bar = np.array([hamiltonian.V_bar[k]([phi, psi]) for k in range(config.n_types)])
    else:
        V_bar = np.array([hamiltonian.V_bar_0[k](phi) for k in range(config.n_types)])

    p1 = np.sum((volume_per_cell / V) * V_bar * phi)

    # Term 3
    comp_laplacian(
        phi_fourier,
        phi_transfer,
        phi_laplacian,
        hamiltonian,
        config,
    )

    p2 = np.sum(
        volume_per_cell
        / V
        * config.sigma**2
        * np.repeat(V_bar[:, np.newaxis, :, :, :], 3, axis=1)
        * phi_laplacian,
        axis=(0, 2, 3, 4),
    )

    # Bonded force term: linking 2 particles
    p_bond = bond_pr / V

    # Angle force term: linking 3 particles
    p_angle = angle_pr / V

    # TODO: Dihedral angle force term: linking 4 atoms
    p_dihedral = np.zeros(3)

    # Add formal parameter dihedral_forces as: comp_pressure(..., dihedral_forces)
    # Define dictionary:
    # forces = {
    #          'x': dihedral_forces[:,0],
    #          'y': dihedral_forces[:,1],
    #          'z': dihedral_forces[:,2]
    #           }
    # Compute the pressure due to dihedrals as:
    # p_dihedral = {
    #          'x': np.sum( np.multiply(forces['x'],positions[:,0]) )*(1/V),
    #          'y': np.sum( np.multiply(forces['y'],positions[:,1]) )*(1/V),
    #          'z': np.sum( np.multiply(forces['z'],positions[:,2]) )*(1/V)
    #              }

    # Total pressure in x, y, z
    p_tot = p_kin + p0 + p1 + p2 + p_bond + p_angle + p_dihedral

    return_value = np.array(
        [
            p_kin,
            p0,
            p1,
            p2[0],
            p2[1],
            p2[2],
            p_bond[0],
            p_bond[1],
            p_bond[2],
            p_angle[0],
            p_angle[1],
            p_angle[2],
            p_dihedral[0],
            p_dihedral[1],
            p_dihedral[2],
            p_tot[0],
            p_tot[1],
            p_tot[2],
        ]
    )
    return_value = comm.allreduce(return_value, MPI.SUM)

    return return_value
