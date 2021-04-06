import logging
import numpy as np
from mpi4py import MPI
from logger import Logger
import sympy
import matplotlib.pyplot as plt

def plot(
        function, phi, hamiltonian, config, phi_gradient, phi_laplacian, title
):
    if(title == 'numerical'):
        lapfactor = config.mesh_size[0]
    elif(title == 'pmesh'):
        lapfactor = 1
    axis = 0
    Y = []
    #PLOTS
    grid1d = np.array(np.arange(0,config.mesh_size[0],1))
    if(function == 'phi'):
        phimod = 1/(config.kappa * config.rho_0)*sum(phi)
        Y.clear()
        grid1d = np.array(np.arange(0,config.mesh_size[0],1))
        for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(config.mesh_size[0])]:
            y = phimod[i][j]
            Y.append(y)
        y = np.average(np.asarray(Y), axis=0)
        plt.plot(grid1d, y, label='1/(config.kappa * config.rho_0)*(phi[0]+phi[1])')
        plt.title(title); plt.legend()
        plt.show()

        #Y.clear()
        #for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(config.mesh_size[0])]:
        #    y = phi_gradient[0][2][i][j]/config.mesh_size[0]§
        #    Y.append(y)
        #    plt.plot(grid1d, y, label='grad[0]')
        #y = np.average(np.asarray(Y), axis=0)
        #plt.plot(grid1d, y, label='grad[0]')

        Y.clear()
        for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(config.mesh_size[0])]:
            y = phi_laplacian[0][2][i][j] * lapfactor
            Y.append(y)
        y = np.average(np.asarray(Y), axis=0)
        plt.plot(grid1d, y, label='lap[0]')
        plt.title(title); plt.legend()
        plt.show()

    if(function == 'V_bar'):
        V_bar = [
          hamiltonian.V_bar[k](phi) for k in range(config.n_types)
        ]
        Y.clear()
        for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(config.mesh_size[0])]:
            y = V_bar[0][i][j]
            Y.append(y)
        y = np.average(np.asarray(Y), axis=0)
        plt.plot(grid1d, y, label='V_bar[0]')
        plt.title(title); plt.legend()
        plt.show()

    if(function == 'V_bar*lap'):
        V_bar = [
          hamiltonian.V_bar[k](phi) for k in range(config.n_types)
        ]
        Y.clear()
        for (i,j) in [(i,j) for i in range(config.mesh_size[0]) for j in range(config.mesh_size[0])]:
            y = V_bar[0][i][j] * phi_laplacian[0][2][i][j]
            Y.append(y)
        y = np.average(np.asarray(Y), axis=0)
        plt.plot(grid1d, y, label='V_bar[0]*lap[0]')
        plt.title(title); plt.legend()
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
        config,
        phi_fourier,
        phi_laplacian,
        phi_new
):
    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells
    w = hamiltonian.w(phi) * volume_per_cell

    #Term 1
    p0 = -1/V * w.csum()
    print('p0:',p0)
  
    #Term 2
    V_bar = [
        hamiltonian.V_bar[k](phi) for k in range(config.n_types)
    ]

    #plot(
    #    'V_bar',phi, hamiltonian, config, phi_gradient, phi_laplacian, 'pmesh'
    #)

    #print('V_bar:',np.asarray(V_bar).shape)
    p1 = [
        1/V #* hamiltonian.V_bar[i](phi)
        * V_bar[i]
        * phi[i] * volume_per_cell for i in range(config.n_types)
    ]
    p1 = np.sum([
        p1[i].csum() for i in range(config.n_types)
    ])
    print('p1:',p1)
    
    #numericallap(phi, hamiltonian, config, V_bar, volume_per_cell)

    #Term 3
    for t in range(config.n_types):
#        print('np.sum(phi[t]):', np.sum(phi[t][:]))
        phi[t].r2c(out=phi_fourier[0])
#        print("\n*****************")
#        print('np.sum(phi_fourier[0]):', np.sum(phi_fourier[0]))
#        print("*****************\n")
        phi_fourier[0].c2r(out = phi_new[t])
#        print('np.sum(phi_new[t])',np.sum(phi_new[t]))
        np.copyto(
            phi_fourier[1].value, phi_fourier[0].value, casting="no", where=True
        )
        np.copyto(
            phi_fourier[2].value, phi_fourier[0].value, casting="no", where=True
        )

        # Evaluate laplacian of phi in fourier space
        for d in range(3):

            def laplacian_transfer(k, v, d=d):
                return -k[d]**2 * v
               # return -k.normp(p=2,zeromode=1) * v
            def gradient_transfer(k, v, d=d):
                return 1j * k * v

            phi_fourier[d].apply(laplacian_transfer, out=Ellipsis)
            phi_fourier[d].c2r(out=phi_laplacian[t][d])
            #phi_fourier[d].apply(gradient_transfer, out=Ellipsis).c2r(out=phi_gradient[t][d])

    p2x = [
        1/V * config.sigma**2 * V_bar[i] * phi_laplacian[i][0] * volume_per_cell for i in range(config.n_types)
    ]
    p2y = [
        1/V * config.sigma**2 * V_bar[i] * phi_laplacian[i][1] * volume_per_cell for i in range(config.n_types)
    ]
    p2z = [
        1/V * config.sigma**2 * V_bar[i] * phi_laplacian[i][2] * volume_per_cell for i in range(config.n_types)
    ]
    p2x = np.cumsum(p2x)
    p2y = np.cumsum(p2y)
    p2z = np.cumsum(p2z)
    print('p2x:',p2x[-1])
    print('p2y:',p2y[-1])
    print('p2z:',p2z[-1])

    #PLOTS
    #plot(
    #    'phi',phi, hamiltonian, config, phi_gradient, phi_laplacian, 'pmesh'
    #) 
    #plot(
    #      'V_bar*lap',phi, hamiltonian, config, phi_gradient, phi_laplacian, 'pmesh'
    #  )

#    #TESTING
#    # phi_t_fft ========= phi_fourier[0] (after applying r2c)
#    phi_t_fft = np.fft.fftn(phi[0][:], norm='forward')
#    print("\n*****************")
#    print('np.sum(phi_t_fft)',np.sum(phi_t_fft))
#    print("*****************\n")
#
#    # k        ========= k[0] 
#    freq = np.fft.fftfreq(config.mesh_size[0], d=1)
#    k    = 2*np.pi * freq
#    
#    # laplaced_x_fft === phi_fourier[0] (after applying laplacian_transfer)
#    laplaced_x_fft = -1 * k**2 * phi_t_fft 
#    laplaced_y_fft = -1 * k**2 * phi_t_fft 
#    laplaced_z_fft = -1 * k**2 * phi_t_fft 
#
#    # laplaced_wf_x ==== phi_lap
#    laplaced_wf_x = np.fft.ifftn(laplaced_x_fft).real
#    laplaced_wf_y = np.fft.ifftn(laplaced_y_fft).real
#    laplaced_wf_z = np.fft.ifftn(laplaced_z_fft).real
#
#    # ptest2x   ======== p2x
#    ptest2x = config.sigma**2 * V_bar[0] * laplaced_wf_x
#    ptest2x = [el*volume_per_cell/V for el in ptest2x]
#    ptest2x = np.sum(ptest2x)
#    print('ptest2x:',ptest2x)
#    ptest2y = config.sigma**2 * V_bar[0] * laplaced_wf_y
#    ptest2y = [el*volume_per_cell/V for el in ptest2y]
#    ptest2x = np.sum(ptest2y)
#    print('ptest2y:',ptest2x)
#    ptest2z = config.sigma**2 * V_bar[0] * laplaced_wf_z
#    ptest2z = [el*volume_per_cell/V for el in ptest2z]
#    ptest2z = np.sum(ptest2z)
#    print('ptest2z:',ptest2z)
    return 1.0
