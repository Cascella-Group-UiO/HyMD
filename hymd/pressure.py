import logging
import numpy as np
from mpi4py import MPI
from logger import Logger
import sympy
from decimal import Decimal

def comp_pressure(
        phi,
        hamiltonian,
        velocities,
        config,
        phi_fourier,
        phi_laplacian,
        lap_transfer,
        args,
        bond_forces,
        angle_forces,
        positions,
        bond_pr,
        angle_pr,
        comm=MPI.COMM_WORLD
):
    rank = comm.Get_rank()
    size = comm.Get_size()

    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells
    w = hamiltonian.w(phi) * volume_per_cell

    #Kinetic term
    kinetic_energy = 0.5 * config.mass * np.sum(velocities ** 2)
    p_kin = 2/(3*V)*kinetic_energy

    #Term 1
    p0 = -1/V * np.sum(w)
  
    #Term 2
    V_bar_tuple = [
        hamiltonian.V_bar[k](phi) for k in range(config.n_types)
    ]
    V_bar = [sum(list(V_bar_tuple[i])) for i in range(len(V_bar_tuple))]

    p1 = [
        1/V
        * V_bar[i]
        * phi[i] * volume_per_cell for i in range(config.n_types)
    ]
    p1 = [np.sum(p1[i].value) for i in range(config.n_types)]
    p1 = np.sum(p1)
    
    #numericallap(phi, hamiltonian, config, V_bar, volume_per_cell)

    #Term 3
    for t in range(config.n_types):
        #print('Inside comp_pressure, phi_fourier for t= ',t, 'is ',phi_fourier[0].value[0][0][0:2])
        np.copyto(
            lap_transfer[0].value, phi_fourier[t].value, casting="no", where=True
        )
        np.copyto(
            lap_transfer[1].value, phi_fourier[t].value, casting="no", where=True
        )
        np.copyto(
            lap_transfer[2].value, phi_fourier[t].value, casting="no", where=True
        )

        # Evaluate laplacian of phi in fourier space
        for d in range(3):

            def laplacian_transfer(k, v, d=d):
                return -k[d]**2 * v
               # return -k.normp(p=2,zeromode=1) * v
            def gradient_transfer(k, v, d=d):
                return 1j * k * v

            lap_transfer[d].apply(laplacian_transfer, out=Ellipsis)
            lap_transfer[d].c2r(out=phi_laplacian[t][d])
            #print('Inside comp_pressure, phi_fourier.apply(lap) for t=',t,' d= ',d, 'is ',phi_fourier[d].value[0][0][0:2])
            #phi_fourier[d].apply(gradient_transfer, out=Ellipsis).c2r(out=phi_gradient[t][d])
    #print('phi_fourier[2].value[0][0]:',phi_fourier[2].value[0][0])

    p2x = [
        1/V * config.sigma**2 * V_bar[i] * phi_laplacian[i][0] * volume_per_cell for i in range(config.n_types)
    ]
    p2y = [
        1/V * config.sigma**2 * V_bar[i] * phi_laplacian[i][1] * volume_per_cell for i in range(config.n_types)
    ]
    p2z = [
        1/V * config.sigma**2 * V_bar[i] * phi_laplacian[i][2] * volume_per_cell for i in range(config.n_types)
    ]

    p2x = [np.sum(p2x[i].value) for i in range(config.n_types)]
    p2y = [np.sum(p2y[i].value) for i in range(config.n_types)]
    p2z = [np.sum(p2z[i].value) for i in range(config.n_types)]
    p2x = np.sum(p2x)
    p2y = np.sum(p2y)
    p2z = np.sum(p2z)

    #Bonded force term: linking 2 particles
    #forces = {
    #        'x': bond_forces[:,0],
    #        'y': bond_forces[:,1],
    #        'z': bond_forces[:,2]
    #         }

    #p_bond = {
    #        'x': np.sum( np.multiply(forces['x'],positions[:,0]) )*(1/V),
    #        'y': np.sum( np.multiply(forces['y'],positions[:,1]) )*(1/V),
    #        'z': np.sum( np.multiply(forces['z'],positions[:,2]) )*(1/V)
    #          }
    p_bond  = {
             'x': bond_pr[0]/V,
             'y': bond_pr[1]/V,
             'z': bond_pr[2]/V
             }

    #Angle force term: linking 3 particles
    #forces = {
    #        'x': angle_forces[:,0],
    #        'y': angle_forces[:,1],
    #        'z': angle_forces[:,2]
    #         }

    #p_angle = {
    #        'x': np.sum( np.multiply(forces['x'],positions[:,0]) )*(1/V),
    #        'y': np.sum( np.multiply(forces['y'],positions[:,1]) )*(1/V),
    #        'z': np.sum( np.multiply(forces['z'],positions[:,2]) )*(1/V)
    #           }
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



    #PLOTS
    if(config.plot):
        plot(
                'phi',phi, hamiltonian, config, phi_laplacian, V_bar_tuple, 'pmesh',
                'x','y'
        ) 
        plot(
              'V_bar*lap',phi, hamiltonian, config, phi_laplacian, V_bar_tuple, 'pmesh',
              'x','y'
        )
        #plot(
        #    'V_bar',phi, hamiltonian, config, phi_laplacian, V_bar_tuple, 'pmesh',
        #    'x','y'
        #)

    #Total pressure in x, y, z
    p_tot = {
        'x': p_kin + p0 + p1 + p2x + p_bond['x'] + p_angle['x'] + p_dihedral['x'],
        'y': p_kin + p0 + p1 + p2y + p_bond['y'] + p_angle['y'] + p_dihedral['y'],
        'z': p_kin + p0 + p1 + p2z + p_bond['z'] + p_angle['z'] + p_dihedral['z']
            }


    return_value = [
                p_kin,p0,p1,
                p2x,p2y,p2z,
                p_bond['x'], p_bond['y'], p_bond['z'],
                p_angle['x'], p_angle['y'], p_angle['z'],
                p_dihedral['x'], p_dihedral['y'], p_dihedral['z'],
                p_tot['x'], p_tot['y'], p_tot['z']
    ]

    return_value = [comm.allreduce(_, MPI.SUM) for _ in return_value]
    #Total pressure across all ranks

    if config.barostat:
        beta = 4.6 * 10**(-5) #bar^(-1) #isothermal compressibility of water
        if config.barostat == 'semiisotropic':
            #L: Lateral; N: Normal
            [PL, PN] = [0, 0]
            PL = (return_value[-3] + return_value[-2])/2
            PN = return_value[-1]
            alphaL = 1 - config.time_step / config.tau_p * beta * (config.target_pressure - PL)
            alphaN = 1 - config.time_step / config.tau_p * beta * (config.target_pressure - PN)
        elif config.barostat == 'isotropic':
            P = np.average(return_value[-3:-1])
            alpha = 1 - config.time_step / config.tau_p * beta * (config.target_pressure - P)
    return return_value
