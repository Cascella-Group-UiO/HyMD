# Bussy Barostat
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
        phi_gradient,
        hamiltonian,
        positions,
        velocities,
        config,
        phi_fft,
        phi_laplacian,
        phi_transfer,
        phi_grad_lap_fourier,
        phi_grad_lap,
        bond_forces,
        angle_forces,
        args,
        bond_pr,
        angle_pr,
        step,
        comm=MPI.COMM_WORLD
    ):
    rank = comm.Get_rank()
    beta = 7.6 * 10**(-4) #bar^(-1) #isothermal compressibility of water
    change = False

    if(np.mod(step, config.n_b)==0):

        if comm.Get_rank() == 0:
            R = np.random.normal()
        R = comm.bcast(R, root=0)

        change = True
        #compute pressure
        pressure = comp_pressure(
                phi,
                phi_gradient,
                hamiltonian,
                velocities,
                config,
                phi_fft,
                phi_laplacian,
                phi_transfer,
                phi_grad_lap_fourier,
                phi_grad_lap,
                args,
                bond_forces,
                angle_forces,
                positions,
                bond_pr,
                angle_pr,
                comm=comm
        )

        #Total pressure across all ranks
        P = np.average(pressure[-3:-1]) #kJ/(mol nm^3)
        P = P * 16.61 #bar

        V = np.prod(config.box_size)
        noise_term = np.sqrt(2 * config.n_b * config.gas_constant * config.target_temperature * beta * config.time_step * config.n_b / (V * config.tau_p)) * R
        log_alpha = - config.n_b * config.time_step * beta / config.tau_p * (config.target_pressure.P_L - P)
        log_alpha = log_alpha + noise_term
        alpha = np.exp(log_alpha / 3.0)

        L0 = alpha * config.box_size[0]
        L1 = alpha * config.box_size[1]
        L2 = alpha * config.box_size[2]

        config.box_size[0] = L0
        config.box_size[1] = L1
        config.box_size[2] = L2

        #position coordinates scaling
        positions[:, :] = alpha * positions

        #pmesh re-initialize
        pm_stuff  = initialize_pm(pmesh, config, comm)
    return (pm_stuff, False)

def semiisotropic(
        pmesh,
        pm_stuff,
        phi,
        phi_gradient,
        hamiltonian,
        positions,
        velocities,
        config,
        phi_fft,
        phi_laplacian,
        phi_transfer,
        phi_grad_lap_fourier,
        phi_grad_lap,
        bond_forces,
        angle_forces,
        args,
        bond_pr,
        angle_pr,
        step,
        comm=MPI.COMM_WORLD
    ):
    rank = comm.Get_rank()
    beta = 7.6 * 10**(-4)  # isothermal compressibility of water
    change = False
    Rxy = Rz = None
    if(np.mod(step, config.n_b)==0):
        if comm.Get_rank() == 0:
            Rxy = np.random.normal()
            Rz = np.random.normal()
        Rxy = comm.bcast(Rxy, root=0)
        Rz = comm.bcast(Rz, root=0)
        change = True
        #compute pressure
        pressure = comp_pressure(
                phi,
                phi_gradient,
                hamiltonian,
                velocities,
                config,
                phi_fft,
                phi_laplacian,
                phi_transfer,
                phi_grad_lap_fourier,
                phi_grad_lap,
                args,
                bond_forces,
                angle_forces,
                positions,
                bond_pr,
                angle_pr,
                comm=comm
        )

        #Total pressure across all ranks
        #L: Lateral; N: Normal
        [PL, PN] = [0, 0]
        PL = (pressure[-3] + pressure[-2])/2 #kJ/(mol nm^3)
        PN = pressure[-1] #kJ/(mol nm^3)
        PL = PL * 16.61 #bar
        PN = PN * 16.61 #bar
        alphaL = 1.0
        alphaN = 1.0
        config.surface_tension = config.box_size[2]/2 * (PN - PL) #bar nm

        if config.target_pressure.P_L:
            V = np.prod(config.box_size)
            noise_term = np.sqrt(4 * config.n_b * config.gas_constant * config.target_temperature * beta * config.time_step * config.n_b / (3 * V * config.tau_p)) * Rxy
            log_alpha = - 2 * config.n_b * config.time_step * beta / (3 * config.tau_p) * (config.target_pressure.P_L - PL - config.surface_tension / config.box_size[2])
            log_alpha = log_alpha + noise_term
            alpha = np.exp(log_alpha / 2.0)  # not 100% sure about this factor 2, have to check it out <<< TODO

            L0 = alpha * config.box_size[0]
            L1 = alpha * config.box_size[1]

            config.box_size[0] = L0
            config.box_size[1] = L1

            for i in range(len(positions)):
                positions[i][0:2] = alpha * positions[i][0:2]

        if config.target_pressure.P_N:
            V = np.prod(config.box_size)
            noise_term = np.sqrt(2 * config.n_b * config.gas_constant * config.target_temperature * beta * config.time_step * config.n_b / (3 * V * config.tau_p)) * Rz
            log_alpha = - config.n_b * config.time_step * beta / (3 * config.tau_p) * (config.target_pressure.P_N - PN)
            log_alpha = log_alpha + noise_term
            alpha = np.exp(log_alpha / 1.0)  # not 100% sure about this factor 2, have to check it out <<< TODO

            L2 = alpha * config.box_size[2]

            config.box_size[2] = L2

            for i in range(len(positions)):
                positions[i][2] = alpha * positions[i][2]

        #pmesh re-initialize
        pm_stuff  = initialize_pm(pmesh, config, comm)
    return (pm_stuff, change)
