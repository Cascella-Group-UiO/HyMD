# Berendsen Barostat
import numpy as np
from pressure import comp_pressure

# Isotropic barostat from HyMD-2020
#def BERENDSEN_BAROSTAT(tau_p, P):
#    beta = 4.6 * 10**(-5) #bar^(-1) #isothermal compressibility of water
#    P_target = CONF['P_target'] #bar
#    dt = CONF['dt']
#
#    #scaling factor
#    alpha = 1 - dt/tau_p*beta*(P_target - P)
#    #print("scaling factor: alpha:",alpha)
#   
#    #length scaling
#    L0 = alpha**(1/3) * CONF['L'][0]
#    L1 = alpha**(1/3) * CONF['L'][1]
#    L2 = alpha**(1/3) * CONF['L'][2]
#
#    #volume scaling
#    V  = L0 * L1 * L2
#    CONF['L'][0] = L0
#    CONF['L'][1] = L1
#    CONF['L'][2] = L2
#    CONF['V'] = V
#    CONF['dV']     = CONF['V']/(CONF['Nv']**3)
#    volumes.append(CONF['V'])
#    
#    #coordinates scaling
#    for i in range(len(r)):
#        r[i]  = alpha**(1/3) * r[i]
#
#    return



def isotropic(
        phi,
        hamiltonian,
        positions,
        velocities,
        config,
        phi_fft,
        phi_laplacian,
        phi_new,
        comm,
        bond_forces,
        angle_forces,
        args,
        bond_pr,
        angle_pr
    ):
    beta = 4.6 * 10**(-5) #bar^(-1) #isothermal compressibility of water

    #compute pressure
    pressure = comp_pressure(
            phi,
            hamiltonian,
            velocities,
            config,
            phi_fft,
            phi_laplacian,
            phi_new,
            args,
            bond_forces,
            angle_forces,
            positions,
            bond_pr,
            angle_pr
    )

    #Total pressure across all ranks
    P = np.average(pressure[-3:-1])

    #scaling factor                                                                                        
    alpha = 1 - config.time_step / config.tau_p * beta * (config.target_pressure - P)

    #length scaling
    L0 = alpha**(1/3) * config.box_size[0]
    L1 = alpha**(1/3) * config.box_size[1]
    L2 = alpha**(1/3) * config.box_size[2]
    config.box_size[0] = L0
    config.box_size[1] = L1
    config.box_size[2] = L2

    #position coordinates scaling
    positions = alpha**(1/3) * positions

    return positions

def semiisotropic(
        phi,
        hamiltonian,
        positions,
        velocities,
        config,
        phi_fft,
        phi_laplacian,
        phi_new,
        comm,
        bond_forces,
        angle_forces,
        args,
        bond_pr,
        angle_pr
    ):

    beta = 4.6 * 10**(-5) #bar^(-1) #isothermal compressibility of water

    #compute pressure
    pressure = comp_pressure(
            phi,
            hamiltonian,
            velocities,
            config,
            phi_fft,
            phi_laplacian,
            phi_new,
            args,
            bond_forces,
            angle_forces,
            positions,
            bond_pr,
            angle_pr
    )

    #Total pressure across all ranks
    #L: Lateral; N: Normal
    [PL, PN] = [0, 0]
    PL = (pressure[-3] + pressure[-2])/2
    PN = pressure[-1]

    #scaling factor                                                                                        
    alphaL = 1 - config.time_step / config.tau_p * beta * (config.target_pressure - PL)
    alphaN = 1 - config.time_step / config.tau_p * beta * (config.target_pressure - PN)

    #length scaling
    L0 = alphaL**(1/3) * config.box_size[0]
    L1 = alphaL**(1/3) * config.box_size[1]
    L2 = alphaN**(1/3) * config.box_size[2]
    config.box_size[0] = L0
    config.box_size[1] = L1
    config.box_size[2] = L2

    #position coordinates scaling
    #print('positions[0] before:',positions[0])
    for i in range(len(positions)):
        positions[i][0:2] = alphaL**(1/3) * positions[i][0:2]
        positions[i][2] = alphaN**(1/3) * positions[i][2]
    #print('positions[0] after:',positions[0])

    return positions
