"""Implements the stochastic cell rescaling (SCR) barostat.
This is a first-order barostat that samples the correct
volume fluctuations through a suitable noise term.
Scales the box and particle positions by a factor (see functions
`isotropic` and `semiisotropic` for details) during simulation to
simulate coupling to an external pressure bath set at a
target pressure.

The box and particle positions are scaled in the L and N directions according
to the nature of the barostat (see functions `isotropic` and `semiisotropic`
below by an amount :math:`\\alpha`.

The updated system information is passed on to the pmesh objects.

References
----------
Mattia Bernetti and Giovanni Bussi , "Pressure control using stochastic
cell rescaling", J. Chem. Phys. 153, 114107 (2020)
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
    It calculates the scaling factor according to

    .. math::

        \\log \\alpha' = - \\frac{n_b\,dt\,\\beta}{\\tau_p}(P^t - P) + \\sqrt{\\frac{2n_b^2k_BT\\beta\,dt}{\mathcal{V}\\tau_p}}dW

    .. math::

        \\alpha = \\exp{\\frac{1}{3}\\log \\alpha'}

    where :math:`dt` is the outer rRESPA time-step, :math:`n_b` is the frequency
    of barostat calls, :math:`\\tau_p` is the pressure coupling time constant,
    :math:`\\beta` is the isothermal compressibility, :math:`P_{L,N}^t` and
    :math:`P_{L,N}` is the target and instantaneous internal pressure in the
    lateral (L) and normal (N) directions respectively. Convention: Cartesian
    z-direction is considered normal.
    """
    rank = comm.Get_rank()
    beta = 7.6 * 10 ** (-4)  # bar^(-1) #isothermal compressibility of water
    change = False

    if np.mod(step, config.n_b) == 0:

        R = prng.normal()

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

        V = np.prod(config.box_size)
        noise_term = (
            np.sqrt(
                2.0
                * config.n_b
                * config.gas_constant
                * config.target_temperature
                * beta
                * config.time_step * config.respa_inner
                * config.n_b
                / (V * config.tau_p)
            )
            * R
        )
        log_alpha = (
            -config.n_b
            * config.time_step * config.respa_inner
            * beta
            / config.tau_p
            * (config.target_pressure.P_L - P)
        )
        log_alpha = log_alpha + noise_term
        alpha = np.exp(log_alpha / 3.0)

        config.box_size *= alpha

        # position coordinates scaling
        positions *= alpha

        # pmesh re-initialize
        pm_stuff = initialize_pm(pmesh, config, comm)
    return (pm_stuff, False)


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
    It calculates the scaling factor according to

    .. math::

        \\log \\alpha'_{L} = - \\frac{2n_b\,dt\,\\beta}{3\\tau_p}(P_L^t - P_L - \\frac{\\gamma}{L_z}) + \\sqrt{\\frac{4n_b^2k_BT\\beta\,dt}{3\mathcal{V}\\tau_p}}dW_L

    .. math::

        \\alpha_L = \\exp{\\frac{1}{2}\\log \\alpha'_L}

    .. math::

        \\log \\alpha'_{N} = - \\frac{n_b\,dt\,\\beta}{3\\tau_p}(P_N^t - P_N) + \\sqrt{\\frac{2n_b^2k_BT\\beta\,dt}{3\mathcal{V}\\tau_p}}dW_N

    .. math::

        \\alpha_L = \\log \\alpha'_N

    where :math:`dt` is the outer rRESPA time-step, :math:`n_b` is the frequency
    of barostat calls, :math:`\\tau_p` is the pressure coupling time constant,
    :math:`\\beta` is the isothermal compressibility, :math:`P_{L,N}^t` and
    :math:`P_{L,N}` is the target and instantaneous internal pressure in the
    lateral (L) and normal (N) directions respectively, :math:`\\gamma` is the
    surface tension. Convention: Cartesian z-direction is considered normal.
    """
    rank = comm.Get_rank()
    beta = 7.6 * 10 ** (-4)  # isothermal compressibility of water
    change = False
    if np.mod(step, config.n_b) == 0:
        Rxy = prng.normal()
        Rz = prng.normal()
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
        config.surface_tension = config.box_size[2] / 2 * (PN - PL)  # bar nm

        if config.target_pressure.P_L:
            V = np.prod(config.box_size)
            noise_term = (
                np.sqrt(
                    4.0
                    * config.n_b
                    * config.gas_constant
                    * config.target_temperature
                    * beta
                    * config.time_step * config.respa_inner
                    * config.n_b
                    / (3 * V * config.tau_p)
                )
                * Rxy
            )
            log_alpha = (
                -2.0
                * config.n_b
                * config.time_step * config.respa_inner
                * beta
                / (3 * config.tau_p)
                * (
                    config.target_pressure.P_L
                    - PL
                    - config.surface_tension / config.box_size[2]
                )
            )
            log_alpha = log_alpha + noise_term
            alpha = np.exp(
                log_alpha / 2.0
            )  # not 100% sure about this factor 2, have to check it out <<< TODO

            config.box_size[0:2] *= alpha

            positions[:][0:2] *= alpha

        if config.target_pressure.P_N:
            V = np.prod(config.box_size)
            noise_term = (
                np.sqrt(
                    2.0
                    * config.n_b
                    * config.gas_constant
                    * config.target_temperature
                    * beta
                    * config.time_step * config.respa_inner
                    * config.n_b
                    / (3 * V * config.tau_p)
                )
                * Rz
            )
            log_alpha = (
                -config.n_b
                * config.time_step * config.respa_inner
                * beta
                / (3 * config.tau_p)
                * (config.target_pressure.P_N - PN)
            )
            log_alpha = log_alpha + noise_term
            alpha = np.exp(
                log_alpha / 1.0
            )  # not 100% sure about this factor 2, have to check it out <<< TODO

            config.box_size[2] *= alpha

            positions[:][2] *= alpha

        # pmesh re-initialize
        pm_stuff = initialize_pm(pmesh, config, comm)
    return (pm_stuff, change)
