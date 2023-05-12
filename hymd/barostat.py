"""Implements the Berendsen barostat.
Scales the box and particle positions during simulation to
simulate coupling to an external pressure bath set at a
target pressure.

It calculates the scaling factor according to:
.. math::

    \\alpha_{L,N} = 1 - \\frac{dt n_b}{\\tau_p}\\β(P_{L,N}^t - P_{L,N})

where :math:`dt` is the outer rRESPA time-step, :math:`n_b` is the frequency
of barostat calls, :math:`\\tau_p` is the pressure coupling time constant,
:math:`\\beta` is the isothermal compressibility, :math:`P_{L,N}^t` and 
:math:`P_{L,N}` is the target and instantaneous internal pressure in the 
lateral (L) and normal (N) directions respectively. Convention: Cartesian
z-direction is considered normal.

The box and particle positions are scaled in the L and N directions according
to the nature of the barostat (see functions `isotropic` and `semiisotropic`
below by an amount :math:`\α^{\\frac{1}{3}}`.

The updated system information is passed on to the pmesh objects.

References
----------
H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren,
A. DiNola, and J. R. Haak , "Molecular dynamics with coupling
to an external bath", J. Chem. Phys. 81, 3684-3690 (1984)
"""
import numpy as np
from mpi4py import MPI
from dataclasses import dataclass
from typing import Union
from .pressure import comp_pressure
from .field import initialize_pm


@dataclass
class Target_pressure:
    P_L: Union[bool, float]
    P_N: Union[bool, float]


def isotropic(
    pmesh,
    pm_stuff,
    phi,
    phi_q,
    psi,
    hamiltonian,
    positions,
    velocities,
    config,
    phi_fft,
    phi_laplacian,
    phi_transfer,
    bond_pr,
    angle_pr,
    step,
    prng,
    comm=MPI.COMM_WORLD,
):
    """
    Implements an isotropic Berendsen barostat.
    The box and particle positions are scaled uniformly
    in the L and N directions.

    Parameters
    ----------
    pmesh : module 'pmesh.pm'
    pm_stuff : list[Union(pmesh.pm.RealField, pmesh.pm.ComplexField]
        List of pmesh objects.
    phi : list[pmesh.pm.RealField], (M,)
        Pmesh :code:`RealField` objects containing discretized particle number
        density values on the computational grid; one for each particle type
        :code:`M`. Pre-allocated, but empty; any values in this field are discarded.
        Changed in-place. Local for each MPI rank--the full computaional grid
        is represented by the collective fields of all MPI ranks.
    hamiltonian : Hamiltonian
        Particle-field interaction energy handler object. Defines the
        grid-independent filtering function, :math:`H`.
    positions : (N,D) numpy.ndarray
        Array of positions for :code:`N` particles in :code:`D` dimensions.
        Local for each MPI rank.
    velocities : (N, D) numpy.ndarray
        Array of velocities of N particles in D dimensions.
    config : Config
        Configuration dataclass containing simulation metadata and parameters.
    phi_fft : list[pmesh.pm.ComplexField], (M,)
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
    bond_pr : (3,) numpy.ndarray
        Total bond pressure due all two-particle bonds.
    angle_pr : (3,) numpy.ndarray
        Total angle pressure due all three-particle bonds.
    step : integer
        MD step number
    prng : np.random.Generator
        Numpy object that provides a stream of random bits
    comm : MPI.Intracomm, optional
        MPI communicator to use for rank commuication. Defaults to
        MPI.COMM_WORLD.

    Returns
    -------
    pm_stuff : list[Union(pmesh.pm.RealField, pmesh.pm.ComplexField]
        List of modified/unmodified pmesh objects.
    change : Boolean
        Indicates whether or not any pmesh objects were reinitialized.
    """
    rank = comm.Get_rank()
    beta = 4.6 * 10 ** (-5)  # bar^(-1) #isothermal compressibility of water
    change = False

    if np.mod(step, config.n_b) == 0:
        change = True
        # compute pressure
        pressure = comp_pressure(
            phi,
            phi_q,
            psi,
            hamiltonian,
            velocities,
            config,
            phi_fft,
            phi_laplacian,
            phi_transfer,
            positions,
            bond_pr,
            angle_pr,
            comm=comm,
        )

        # Total pressure across all ranks
        P = np.average(pressure[-3:-1])  # kJ/(mol nm^3)
        P = P * 16.61  # bar

        # scaling factor
        alpha = (
            1.0
            - (config.time_step * config.respa_inner)
            * config.n_b
            / config.tau_p
            * beta
            * (config.target_pressure.P_L - P)
        ) ** (1 / 3)

        # length scaling
        config.box_size *= alpha

        # position coordinates scaling
        positions *= alpha

        # pmesh re-initialize
        pm_stuff = initialize_pm(pmesh, config, comm)
    return (pm_stuff, change)


def semiisotropic(
    pmesh,
    pm_stuff,
    phi,
    phi_q,
    psi,
    hamiltonian,
    positions,
    velocities,
    config,
    phi_fft,
    phi_laplacian,
    phi_transfer,
    bond_pr,
    angle_pr,
    step,
    prng,
    comm=MPI.COMM_WORLD,
):
    """
    Implements a semiisotropic Berendsen barostat.
    The box and particle positions are scaled by :math:`\\alpha_L^{\\frac{1}{3}}`
    in the L direction and by :math:`\\alpha_N^{\\frac{1}{3}}` in the N direction.

    Parameters
    ----------
    pmesh : module 'pmesh.pm'
    pm_stuff : list[Union(pmesh.pm.RealField, pmesh.pm.ComplexField]
        List of pmesh objects.
    phi : list[pmesh.pm.RealField], (M,)
        Pmesh :code:`RealField` objects containing discretized particle number
        density values on the computational grid; one for each particle type
        :code:`M`. Pre-allocated, but empty; any values in this field are discarded.
        Changed in-place. Local for each MPI rank--the full computaional grid
        is represented by the collective fields of all MPI ranks.
    hamiltonian : Hamiltonian
        Particle-field interaction energy handler object. Defines the
        grid-independent filtering function, :math:`H`.
    positions : (N,D) numpy.ndarray
        Array of positions for :code:`N` particles in :code:`D` dimensions.
        Local for each MPI rank.
    velocities : (N, D) numpy.ndarray
        Array of velocities of N particles in D dimensions.
    config : Config
        Configuration dataclass containing simulation metadata and parameters.
    phi_fft : list[pmesh.pm.ComplexField], (M,)
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
    bond_pr : (3,) numpy.ndarray
        Total bond pressure due all two-particle bonds.
    angle_pr : (3,) numpy.ndarray
        Total angle pressure due all three-particle bonds.
    step : integer
        MD step number
    prng : np.random.Generator
        Numpy object that provides a stream of random bits
    comm : MPI.Intracomm, optional
        MPI communicator to use for rank commuication. Defaults to
        MPI.COMM_WORLD.

    Returns
    -------
    pm_stuff : list[Union(pmesh.pm.RealField, pmesh.pm.ComplexField]
        List of modified/unmodified pmesh objects.
    change : Boolean
        Indicates whether or not any pmesh objects were reinitialized.
    """
    rank = comm.Get_rank()
    beta = 4.6 * 10 ** (-5)  # bar^(-1) #isothermal compressibility of water
    change = False
    if np.mod(step, config.n_b) == 0:
        change = True

        # compute pressure
        pressure = comp_pressure(
            phi,
            phi_q,
            psi,
            hamiltonian,
            velocities,
            config,
            phi_fft,
            phi_laplacian,
            phi_transfer,
            positions,
            bond_pr,
            angle_pr,
            comm=comm,
        )

        # Total pressure across all ranks
        # L: Lateral; N: Normal
        [PL, PN] = [0, 0]
        PL = (pressure[-3] + pressure[-2]) / 2  # kJ/(mol nm^3)
        PN = pressure[-1]  # kJ/(mol nm^3)
        PL = PL * 16.61  # bar
        PN = PN * 16.61  # bar
        alphaL = 1.0
        alphaN = 1.0

        if config.target_pressure.P_L:
            # scaling factor
            alphaL = (
                1.0
                - (config.time_step * config.respa_inner)
                * config.n_b
                / config.tau_p
                * beta
                * (config.target_pressure.P_L - PL)
            ) ** (1 / 3)
            # length scaling
            config.box_size[0:2] *= alphaL

            positions[:][0:2] *= alphaL

        if config.target_pressure.P_N:
            # scaling factor
            alphaN = (
                1.0
                - (config.time_step * config.respa_inner)
                * config.n_b
                / config.tau_p
                * beta
                * (config.target_pressure.P_N - PN)
            ) ** (1 / 3)
            # length scaling
            config.box_size[2] *= alphaN

            positions[:][2] *= alphaN

        # pmesh re-initialize
        pm_stuff = initialize_pm(pmesh, config, comm)
    return (pm_stuff, change)
