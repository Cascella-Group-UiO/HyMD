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
        phi_gradient,
        hamiltonian,
        velocities,
        config,
        phi_fourier,
        phi_laplacian,
        phi_transfer,
        phi_grad_lap_fourier,
        phi_grad_lap,
        positions,
        bond_pr,
        angle_pr,
        comm=MPI.COMM_WORLD
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
    phi_gradient : list[pmesh.pm.RealField], (M, 3)
        Like phi, but containing the gradient of particle number densities.
        Needed only for vestigial squaregradient term.
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
    phi_grad_lap_fourier : list[pmesh.pm.ComplexField], (3,)
        Like phi_fourier, used as a second intermediary after phi_transfer
        to perform FFT operations to obtain gradient of laplacian of particle
        number densities. Needed only for vestigial squaregradient term.
    phi_grad_lap : list[pmesh.pm.RealField], (M, 3, 3)
        Like phi, to obtain the gradient of laplacian in all 3x3 directions.
        Needed only for vestigial squaregradient term.
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
    pressure : (25,) numpy.ndarray
        Pressure contributions from various energy terms.
        0: due to kinetic energy
        1-5: due to field energy
        6-12: due to vestigial squaregradient (defaults to 0)
        13-15: due to two-particle bonded terms
        16-18: due to three-particle bonded terms (called angle terms)
        19-21: due to four-particle bonded terms (called dihedral terms)
        (defaults to 0 currently. Yet to be implemented)
        22-24: total pressure in x,y,z directions.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells

    w_0 = hamiltonian.w_0(phi) * volume_per_cell
    w_elec = 0.
    if ((config.coulombtype == 'PIC_Spectral_GPE')
        or (config.coulombtype == 'PIC_Spectral')):
        w_elec = hamiltonian.w_elec([phi_q,psi]) * volume_per_cell  
    w_1 = 0.
    if config.squaregradient:
        w_1 = hamiltonian.w_1(phi_gradient) * volume_per_cell
    w = w_0 + w_elec

    #Kinetic term
    kinetic_energy = 0.5 * config.mass * np.sum(velocities ** 2)
    p_kin = 2/(3*V)*kinetic_energy

    #Term 1
    p0 = -1/V * np.sum(w)

    #Term 2
    V_bar_tuple = [
        hamiltonian.V_bar[k]([phi,psi]) for k in range(config.n_types)
    ]

    V_bar = [sum(list(V_bar_tuple[i])) for i in range(len(V_bar_tuple))]

    p1 = [
        1/V
        * V_bar[i]
        * phi[i] * volume_per_cell for i in range(config.n_types)
    ]
    p1 = [np.sum(p1[i].value) for i in range(config.n_types)]
    p1 = np.sum(p1)

    #Term 3
    comp_laplacian(
            phi_fourier,
            phi_transfer,
            phi_laplacian,
            phi_grad_lap_fourier,
            phi_grad_lap,
            hamiltonian,
            config,
            )

    p2x = [
        1/V * config.sigma**2 * V_bar[i] * phi_laplacian[i][0] * volume_per_cell for i in range(config.n_types)
    ]
    p2y = [
        1/V * config.sigma**2 * V_bar[i] * phi_laplacian[i][1] * volume_per_cell for i in range(config.n_types)
    ]
    p2z = [
        1/V * config.sigma**2 * V_bar[i] * phi_laplacian[i][2] * volume_per_cell for i in range(config.n_types)
    ]

    #square gradient: vesitigial now: config.squaregradient = False always
    p_w1_0 = 0.0
    p_w1_1 = [0.0, 0.0, 0.0]
    p_w1_2 = [0.0, 0.0, 0.0]
    if config.squaregradient:
        p_w1_0 = 1/V * np.sum(w_1)
        #p_w1_1 and p_w1_2
        config.K_coupl_type_dictionary = {
            tuple(sorted([c.atom_1, c.atom_2])): c.squaregradient_energy
            for c in config.K_coupl
        }
        type_to_name_map=config.type_to_name_map
        for d in range(3):
            for i in range(config.n_types):
                for j in range(config.n_types):
                    ni = type_to_name_map[i]
                    nj = type_to_name_map[j]
                    names = sorted([ni, nj])
                    if ni!=nj:
                        c = config.K_coupl_type_dictionary[tuple(names)]
                    else:
                        c = 0

                    p_w1_1[d] += 1/V * c / config.rho_0   \
                                 * phi_gradient[i][d] * phi_gradient[j][d] * volume_per_cell
                    for d_dot in range(3):
                        #This contains a gradient of the laplacian <- anisotropic also
                        p_w1_2[d] += 1/V * config.sigma**2 * c / config.rho_0    \
                                     * phi_gradient[i][d_dot] * phi_grad_lap[j][d][d_dot] * volume_per_cell
        for d in range(3):
            p_w1_1[d] = np.sum(p_w1_1[d])
            p_w1_2[d] = np.sum(p_w1_2[d])

    p2x = [np.sum(p2x[i].value) for i in range(config.n_types)]
    p2y = [np.sum(p2y[i].value) for i in range(config.n_types)]
    p2z = [np.sum(p2z[i].value) for i in range(config.n_types)]
    p2x = np.sum(p2x)
    p2y = np.sum(p2y)
    p2z = np.sum(p2z)

    #Bonded force term: linking 2 particles
    p_bond  = {
             'x': bond_pr[0]/V,
             'y': bond_pr[1]/V,
             'z': bond_pr[2]/V
             }

    #Angle force term: linking 3 particles
    p_angle =  {
             'x': angle_pr[0]/V,
             'y': angle_pr[1]/V,
             'z': angle_pr[2]/V
             }

    #Dihedral angle force term: linking 4 atoms
    p_dihedral = {
              'x': 0.0,
              'y': 0.0,
              'z': 0.0
                  }

    #Add formal parameter dihedral_forces as: comp_pressure(..., dihedral_forces)
    #Define dictionary:
    #forces = {
    #          'x': dihedral_forces[:,0],
    #          'y': dihedral_forces[:,1],
    #          'z': dihedral_forces[:,2]
    #           }
    # Compute the pressure due to dihedrals as:
    #p_dihedral = {
    #          'x': np.sum( np.multiply(forces['x'],positions[:,0]) )*(1/V),
    #          'y': np.sum( np.multiply(forces['y'],positions[:,1]) )*(1/V),
    #          'z': np.sum( np.multiply(forces['z'],positions[:,2]) )*(1/V)
    #              }



    #Total pressure in x, y, z
    p_tot = {
        'x': p_kin + p0 + p1 + p2x + p_w1_0 + p_w1_1[0] + p_w1_2[0] + p_bond['x'] + p_angle['x'] + p_dihedral['x'],
        'y': p_kin + p0 + p1 + p2y + p_w1_0 + p_w1_1[1] + p_w1_2[1] + p_bond['y'] + p_angle['y'] + p_dihedral['y'],
        'z': p_kin + p0 + p1 + p2z + p_w1_0 + p_w1_1[2] + p_w1_2[2] + p_bond['z'] + p_angle['z'] + p_dihedral['z']
            }

    return_value = [
                p_kin,p0,p1,                                                  #0-2
                p2x,p2y,p2z,                                                  #3-5
                p_w1_0,                                                       #6
                p_w1_1[0], p_w1_2[0], p_w1_1[1], p_w1_2[1], p_w1_1[2], p_w1_2[2],#7-12
                p_bond['x'], p_bond['y'], p_bond['z'],                        #13-15
                p_angle['x'], p_angle['y'], p_angle['z'],                     #16-18
                p_dihedral['x'], p_dihedral['y'], p_dihedral['z'],            #19-21
                p_tot['x'], p_tot['y'], p_tot['z']                            #22-24
    ]
    return_value = [comm.allreduce(_, MPI.SUM) for _ in return_value]
    #Total pressure across all ranks

    return return_value
