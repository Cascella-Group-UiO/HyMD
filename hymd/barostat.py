# Berendsen Barostat
import numpy as np
from mpi4py import MPI
from dataclasses import dataclass
from typing import Union
from pressure import comp_pressure
from field import initialize_pm

@dataclass
class Target_pressure:
    P_L: Union[bool, float]
    P_N: Union[bool, float]

def isotropic(
        pmesh,
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
        comm=MPI.COMM_WORLD
    ):
    beta = 4.6 * 10**(-5) #bar^(-1) #isothermal compressibility of water
    rank = comm.Get_rank()

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
    P = np.average(pressure[-3:-1])

    #scaling factor                                                                                        
    alpha = 1 - config.time_step / config.tau_p * beta * (config.target_pressure.P_L - P)

    #length scaling
    L0 = alpha**(1/3) * config.box_size[0]
    L1 = alpha**(1/3) * config.box_size[1]
    L2 = alpha**(1/3) * config.box_size[2]
    config.box_size[0] = L0
    config.box_size[1] = L1
    config.box_size[2] = L2

    #position coordinates scaling
    positions[:] = alpha**(1/3) * positions

    #pmesh re-initialize
    pm_stuff  = initialize_pm(pmesh, config, comm)
    #(pm, phi, phi_fourier, force_on_grid, v_ext_fourier, v_ext, phi_transfer, phi_laplacian,
    #field_list) = pm_stuff
    return pm_stuff

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
        comm=MPI.COMM_WORLD
    ):
    rank = comm.Get_rank()
    beta = 4.6 * 10**(-5) #bar^(-1) #isothermal compressibility of water
    eps_alpha = abs(1 - config.alpha_0)

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
    PL = (pressure[-3] + pressure[-2])/2
    PN = pressure[-1]
    alphaL = 1.0
    alphaN = 1.0

    if config.target_pressure.P_L:
        #scaling factor                                                                                        
        alphaL = 1 - config.time_step / config.tau_p * beta * (config.target_pressure.P_L - PL)
        if(abs(1-alphaL) > eps_alpha):
            #length scaling
            config.box_size[0] = alphaL**(1/3) * config.box_size[0]
            config.box_size[1] = alphaL**(1/3) * config.box_size[1]
            for i in range(len(positions)):
                positions[i][0:2] = alphaL**(1/3) * positions[i][0:2]
    if config.target_pressure.P_N:
        #scaling factor                                                                                        
        alphaN = 1 - config.time_step / config.tau_p * beta * (config.target_pressure.P_N - PN)
        if(abs(1-alphaN) > eps_alpha):
            #length scaling
            config.box_size[2] = alphaN**(1/3) * config.box_size[2]
            for i in range(len(positions)):
                positions[i][2] = alphaN**(1/3) * positions[i][2]
    if(abs(1-alphaL) > eps_alpha or abs(1-alphaN) > eps_alpha):
        #pmesh re-initialize
        pm_stuff  = initialize_pm(pmesh, config, comm)


#    #scaling factor                                                                                        
#    alphaL = 1 - config.time_step / config.tau_p * beta * (config.target_pressure - PL)
#    alphaN = 1 - config.time_step / config.tau_p * beta * (config.target_pressure - PN)
#    if(abs(1-alphaL) > eps_alpha or abs(1-alphaN) > eps_alpha):
#        #length scaling
#        L0 = alphaL**(1/3) * config.box_size[0]
#        L1 = alphaL**(1/3) * config.box_size[1]
#        L2 = alphaN**(1/3) * config.box_size[2]
#        config.box_size[0] = L0
#        config.box_size[1] = L1
#        config.box_size[2] = L2
#    
#        for i in range(len(positions)):
#            positions[i][0:2] = alphaL**(1/3) * positions[i][0:2]
#            positions[i][2] = alphaN**(1/3) * positions[i][2]
#    
#        #pmesh re-initialize
#        pm_stuff  = initialize_pm(pmesh, config, comm)
    return pm_stuff
