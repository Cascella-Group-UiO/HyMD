import logging
import numpy as np
from mpi4py import MPI
from logger import Logger
import sympy
from decimal import Decimal

def plot(
        function, phi, hamiltonian, config, phi_laplacian, V_bar_tuple, title, *args
):
    if(title == 'numerical'):
        lapfactor = config.mesh_size[0]
    elif(title == 'pmesh'):
        lapfactor = 1
    axis = 0
    Y = []
    V_bar = [sum(list(V_bar_tuple[i])) for i in range(len(V_bar_tuple))]
    V = np.prod(config.box_size)

    #PLOTS
    grid1d = np.array(np.arange(0,config.mesh_size[0],1))
    if(function == 'phi'):
        #phi plots
        phimod = 1/(config.kappa * config.rho_0)*sum(phi)
        Y.clear()
        grid1d = np.array(np.arange(0,config.mesh_size[0],1))
        for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(config.mesh_size[0])]:
            y = phimod[i][j]
            Y.append(y)
        y = np.average(np.asarray(Y), axis=0)
        plt.plot(grid1d, y, label='k*(phi[0]+phi[1]) in z')
        plt.title(title); plt.legend()

        if(args):
            if('y' in args):
                Y.clear()
                phimod_tr = np.transpose(phimod, axes=[0,2,1])
                for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(config.mesh_size[0])]:
                    y = phimod_tr[i][j]
                    Y.append(y)
                y = np.average(np.asarray(Y), axis=0)
                plt.plot(grid1d, y, label='k*(phi1+phi2) in y')
                plt.title(title); plt.legend()
            if('x' in args):
                Y.clear()
                phimod_tr = np.transpose(phimod, axes=[1,2,0])
                for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(config.mesh_size[0])]:
                    y = phimod_tr[i][j]
                    Y.append(y)
                y = np.average(np.asarray(Y), axis=0)
                plt.plot(grid1d, y, label='k*(phi1+phi2) in x')
                plt.title(title); plt.legend()
        plt.show()

        #phi_laplacian plots
        Y.clear()
        for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(config.mesh_size[0])]:
            y = phi_laplacian[0][2][i][j] * lapfactor
            Y.append(y)
        y = np.average(np.asarray(Y), axis=0)
        plt.plot(grid1d, y, label='lap[0] in z')
        plt.title(title); plt.legend()
        if(args):
            if('y' in args):
                Y.clear()
                phi_laplacian_tr = np.transpose(phi_laplacian[0][1], axes=[0,2,1])
                for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(round(config.mesh_size[0]*6/10),round(9/10*config.mesh_size[0]))]:
                    y = phi_laplacian_tr[i][j]
                    Y.append(y)
                y = np.average(np.asarray(Y), axis=0)
                plt.plot(grid1d, y, label='lap[0] in y lower half')
                plt.title(title); plt.legend()
            if('x' in args):
                Y.clear()
                phi_laplacian_tr = np.transpose(phi_laplacian[0][0], axes=[1,2,0])
                for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(round(config.mesh_size[0]*6/10),round(9/10*config.mesh_size[0]))]:
                    y = phi_laplacian_tr[i][j]
                    Y.append(y)
                y = np.average(np.asarray(Y), axis=0)
                plt.plot(grid1d, y, label='lap[0] in x lower half')
                plt.title(title); plt.legend()
        plt.show()

    if(function == 'V_bar' or function == 'V_bar*lap'):
        #V_bar plots
        Y_int = [] ; Y_inc = []
        Y.clear()
        V_interaction = [_[0] for _ in V_bar_tuple]
        V_incompressibility = [_[1] for _ in V_bar_tuple]
        for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(config.mesh_size[0])]:
            # Total V_bar for type = 0 and direction = z
            y = V_bar[0][i][j]
            #y_int = V_interaction[0][i][j]
            y_inc = V_incompressibility[0][i][j]
            Y.append(y)
            #Y_int.append(y_int)
            Y_inc.append(y_inc)
        y = np.average(np.asarray(Y), axis=0)
        #y_int = np.average(np.asarray(Y_int), axis=0)
        y_inc = np.average(np.asarray(Y_inc), axis=0)
        #plt.plot(grid1d, y_int, label='V_interaction[0]')
        plt.title(title); plt.legend()
        plt.plot(grid1d, y_inc, label='V_incompressibility[0]')
        plt.title(title); plt.legend()
        plt.show()
        plt.plot(grid1d, y, label='V_bar[0] in z')
        plt.title(title); plt.legend()

        if(args):
            if('y' in args):
                Y.clear()
                V_bar_tr = np.transpose(V_bar[0], axes=[0,2,1])
                #for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(config.mesh_size[0])]:
                for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(round(config.mesh_size[0]/10),round(4/10*config.mesh_size[0]))]:
                    y = V_bar_tr[i][j]
                    Y.append(y)
                y = np.average(np.asarray(Y), axis=0)
                plt.plot(grid1d, y, label='V_bar[0] in y lower half')
                plt.title(title); plt.legend()

                Y.clear()
                for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(round(6/10*config.mesh_size[0]),round(9/10*config.mesh_size[0]))]:
                    y = V_bar_tr[i][j]
                    Y.append(y)
                y = np.average(np.asarray(Y), axis=0)
                plt.plot(grid1d, y, label='V_bar[0] in y higher half')
                plt.title(title); plt.legend()
            if('x' in args):
                Y.clear()
                V_bar_tr = np.transpose(V_bar[0], axes=[1,2,0])
                for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(round(config.mesh_size[0]/10),round(4/10*config.mesh_size[0]))]:
                    y = V_bar_tr[i][j]
                    Y.append(y)
                y = np.average(np.asarray(Y), axis=0)
                plt.plot(grid1d, y, label='V_bar[0] in x lower half')
                plt.title(title); plt.legend()

                Y.clear()
                for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(round(6/10*config.mesh_size[0]),round(9/10*config.mesh_size[0]))]:
                    y = V_bar_tr[i][j]
                    Y.append(y)
                y = np.average(np.asarray(Y), axis=0)
                plt.plot(grid1d, y, label='V_bar[0] in x higher half')
                plt.title(title); plt.legend()
        plt.show()

        #V_bar * lap plots
        Y.clear()
        for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(config.mesh_size[0])]:
            y = V_bar[0][i][j] * phi_laplacian[0][2][i][j]
            Y.append(y)
        y = np.average(np.asarray(Y), axis=0)
        plt.plot(grid1d, y, label='V_bar[0]*lap[0] in z direction')
        plt.title(title); plt.legend()
        if(args):
            if('y' in args):
                Y.clear()
                V_bar_tr = np.transpose(V_bar[0], axes=[0,2,1])
                phi_laplacian_tr = np.transpose(phi_laplacian[0][1], axes=[0,2,1])
                for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(round(config.mesh_size[0]/10),round(4/10*config.mesh_size[0]))]:
                    y = V_bar_tr[i][j] * phi_laplacian_tr[i][j]
                    Y.append(y)
                y = np.average(np.asarray(Y), axis=0)
                plt.plot(grid1d, y, label='V_bar[0]*lap[0] in y lower half')
                plt.title(title); plt.legend()
            if('x' in args):
                Y.clear()
                V_bar_tr = np.transpose(V_bar[0], axes=[1,2,0])
                phi_laplacian_tr = np.transpose(phi_laplacian[0][1], axes=[1,2,0])
                for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(round(config.mesh_size[0]/10),round(4/10*config.mesh_size[0]))]:
                    y = V_bar_tr[i][j] * phi_laplacian_tr[i][j]
                    Y.append(y)
                y = np.average(np.asarray(Y), axis=0)
                plt.plot(grid1d, y, label='V_bar[0]*lap[0] in x lower half')
                plt.title(title); plt.legend(loc='lower right')
        plt.show()



def numericallap(phi, hamiltonian, config, V_bar, volume_per_cell
):
    #print('np.asarray(phi[:][:]).shape:',np.asarray(phi[:][:]).shape)
    V = np.prod(config.box_size)
    gradtot = [] #gradtot[type][x][y][z]
    laptot = []
    #Calculate laplacian for phi[0][:] and phi[1][:] and add
    for t in range(config.n_types):
        grad = np.gradient(phi[t][:], axis=(0,1,2))
        #print('np.asarray(grad1).shape',np.asarray(grad).shape)
        lap = [np.gradient(grad[i], axis = i) for i in range(3)]
        #print('np.asarray(grad2).shape',np.asarray(grad2).shape)
        gradtot.append(grad) 
        laptot.append(lap)
    #totallap = np.sum(lap, axis=0)
    #print('np.asarray(totallap).shape',np.asarray(totallap).shape)
    #print('np.asarray(totallap).shape',np.asarray(totallap).shape)
    p2x = [
          config.sigma**2 * V_bar[i] * lap[i][0] * volume_per_cell for i in range(config.n_types)
          #lap[i][0] for i in range(config.n_types)
      ]
    p2x = np.sum(p2x, axis=0)
    p2x = 1/V * np.cumsum(p2x)[-1]
    print('numerical p2x:',p2x)
    p2y = [
          config.sigma**2 * V_bar[i] * lap[i][1] * volume_per_cell for i in range(config.n_types)
      ]
    p2y = np.sum(p2y, axis=0)
    p2y = 1/V * np.cumsum(p2y)[-1]
    print('numerical p2y:',p2y)
    p2z = [
          config.sigma**2 * V_bar[i] * lap[i][2] * volume_per_cell for i in range(config.n_types)
      ]
    p2z = np.sum(p2z, axis=0)
    p2z = 1/V * np.cumsum(p2z)[-1]
    print('numerical p2z:',p2z)

    #PLOTS
    #plot(
    #    'phi', phi, hamiltonian, config, gradtot, laptot, 'numerical'
    #)
    #grid1d = np.array(np.arange(0,config.mesh_size[0],1))
    #grid1d = np.array(np.arange(13,36,1))
    #y = phi[0][10][10]
    #plt.plot(grid1d, y, label='phi[0]')
    #y = phi[1][10][10]
    #plt.plot(grid1d, y, label='phi[1]')
    #plt.legend()
    #plt.show()
    #y = gradtot[0][2][10][10]
    #plt.plot(grid1d, y, label='grad[0]')
    #y = gradtot[1][2][10][10]
    #plt.plot(grid1d, y, label='grad[1]')
    #plt.legend()
    #plt.show()
    #y = laptot[0][2][10][10]
    #plt.plot(grid1d, y, label='lap[0]')
    #y = laptot[1][2][10][10]
    #plt.plot(grid1d, y, label='lap[1]')
    #plt.legend()
    #plt.show()

    #y = phi[0][10][10]
    #plt.plot(grid1d, y, label='phi[0]')
    #y = gradtot[0][2][10][10]
    #plt.plot(grid1d, y, label='grad[0]')
    #y = laptot[0][2][10][10]
    #plt.plot(grid1d, y, label='lap[0]')
    #plt.legend()
    #plt.show()


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
        eps,
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
        #L: Lateral; N: Normal
        [PL, PN] = [0, 0]
        PL = (return_value[-3] + return_value[-2])/2
        PN = return_value[-1]
        beta = 4.6 * 10**(-5) #bar^(-1) #isothermal compressibility of water
        eps_alpha = Decimal('%.0E'%abs(1 - config.alpha_0))
        alphaL = 1 - config.time_step / config.tau_p * beta * (config.target_pressure - PL)
        alphaN = 1 - config.time_step / config.tau_p * beta * (config.target_pressure - PN)
        eps[0] = abs(1-alphaL); eps[1] = abs(1-alphaN)

    return return_value
