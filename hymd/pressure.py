import logging
import numpy as np
from mpi4py import MPI
from logger import Logger
import sympy
from field import comp_laplacian

def bin_pr(pr, r, d, config, comm):
    rank = comm.Get_rank()
    bond_pr_binned = np.zeros(tuple(config.mesh_size))
    #binning z axis
    z = r[:,2]
    n_bins = config.mesh_size[2]
    bins = np.linspace(0, config.box_size[d], num=n_bins)
    indices = np.digitize(z, bins)
    hist = np.zeros(n_bins)
    for i in range(len(indices)):
        idx = indices[i]
        hist[idx-1] += pr[i,d]
    bond_pr_binned[0,0,:] = hist
    #print('n_bins:',n_bins)
    #print('pr:',pr.shape,pr)
    #print('sum of pr:',np.sum(pr))
    #print('r:',r.shape,r)
    #print('max_pos:',max_pos)
    #print('min_pos:',min_pos)
    #print('indices:',indices.shape,indices)
    #print('hist:',hist.shape)
    #print(hist)
    #with open('bond_pr_hist.dat','w') as f1:
    #    f1.write('index\tbin\tbond_pr_hist')
    #    for i in range(len(bins)):
    #        f1.write(str(i)+'\t'+str(bins[i])+'\t'+str(hist[i])+'\n')
    return bond_pr_binned

def comp_pressure(
        phi,
        phi_gradient,
        hamiltonian,
        velocities,
        config,
        phi_fourier,
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
        comm=MPI.COMM_WORLD
):
    rank = comm.Get_rank()
    size = comm.Get_size()

    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells
    w = hamiltonian.w(phi) * volume_per_cell
    pr_arr = np.zeros([25, w.shape[0], w.shape[1], w.shape[2]])
    w1 = 0.0
    if config.squaregradient:
        w1 = hamiltonian.w1(phi_gradient) * volume_per_cell


    #Kinetic term
    kinetic_energy = 0.5 * config.mass * np.sum(velocities ** 2)
    p_kin = 2/(3*V)*kinetic_energy
    if config.pr_arr:
        pr_arr[0][:] = p_kin / np.prod(w.shape)

    #Term 1
    p0 = -1/V * w
    if config.pr_arr:
        pr_arr[1] = p0
    p0 = np.sum(p0)
  
    #Term 2
    V_bar_tuple = [
        hamiltonian.V_bar[k](phi) for k in range(config.n_types)
    ]
    V_bar = [sum(list(V_bar_tuple[i])) for i in range(len(V_bar_tuple))]
    p1 = [
        1/V
        * V_bar[i].value
        * phi[i].value * volume_per_cell for i in range(config.n_types)
    ]
    p1 = np.sum(p1, axis=0)
    if config.pr_arr:
        pr_arr[2] = p1
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
        1/V * config.sigma**2 * V_bar[i].value * phi_laplacian[i][0].value * volume_per_cell for i in range(config.n_types)
    ]
    p2y = [
        1/V * config.sigma**2 * V_bar[i].value * phi_laplacian[i][1].value * volume_per_cell for i in range(config.n_types)
    ]
    p2z = [
        1/V * config.sigma**2 * V_bar[i].value * phi_laplacian[i][2].value * volume_per_cell for i in range(config.n_types)
    ]

    #square gradient
    p_w1_0 = 0.0
    p_w1_1 = [0.0, 0.0, 0.0]
    p_w1_2 = [0.0, 0.0, 0.0]
    if config.squaregradient:
        p_w1_0 = 1/V * w1
        if config.pr_arr:
            pr_arr[6] = p_w1_0
        p_w1_0 = np.sum(p_w1_0)

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
                                 * phi_gradient[i][d].value * phi_gradient[j][d].value * volume_per_cell
                    for d_dot in range(3):
                        #This contains a gradient of the laplacian <- anisotropic also
                        p_w1_2[d] += 1/V * config.sigma**2 * c / config.rho_0    \
                                     * phi_gradient[i][d_dot].value * phi_grad_lap[j][d][d_dot].value * volume_per_cell
        if config.pr_arr:
            pr_arr[7] = p_w1_1[0]
            pr_arr[8] = p_w1_2[0]
            pr_arr[9] = p_w1_1[1]
            pr_arr[10] = p_w1_2[1]
            pr_arr[11] = p_w1_1[2]
            pr_arr[12] = p_w1_2[2]
        for d in range(3):
            p_w1_1[d] = np.sum(p_w1_1[d])
            p_w1_2[d] = np.sum(p_w1_2[d])

    p2x = np.sum(p2x, axis=0)
    p2y = np.sum(p2y, axis=0)
    p2z = np.sum(p2z, axis=0)
    if config.pr_arr:
        pr_arr[3] = p2x
        pr_arr[4] = p2y
        pr_arr[5] = p2z
    p2x = np.sum(p2x)
    p2y = np.sum(p2y)
    p2z = np.sum(p2z)

    bond_pr = bond_pr/V
    angle_pr = angle_pr/V
    if config.pr_arr:
        for d in range(3):
            pr_arr[13+d] = bin_pr(bond_pr, positions, d, config, comm)
            pr_arr[16+d] = bin_pr(angle_pr, positions, d, config, comm)

    #Bonded force term: linking 2 particles
    bond_pr = np.sum(bond_pr, axis=0)
    p_bond  = {
             'x': bond_pr[0],
             'y': bond_pr[1],
             'z': bond_pr[2]
             }

    #Angle force term: linking 3 particles
    angle_pr = np.sum(angle_pr, axis=0)
    p_angle =  {
             'x': angle_pr[0],
             'y': angle_pr[1],
             'z': angle_pr[2]
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

    if config.pr_arr:
        return_value = pr_arr
    else:
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
    return return_value
