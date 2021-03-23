import logging
import numpy as np
from mpi4py import MPI
from logger import Logger
import sympy

def comp_pressure(
        phi,
        hamiltonian,
        config,
        phi_fourier,
        phi_laplacian
):
    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells
    w = hamiltonian.w(phi) * volume_per_cell

    #Term 1
    p0 = -1/V * w.csum()
    #print('p0:',p0)
  
    #Term 2
    V_bar = [
        hamiltonian.V_bar[k](phi) for k in range(config.n_types)
    ]
    p1 = [
        1/V * #hamiltonian.V_bar[i](phi)
        V_bar[i] * phi[i] * volume_per_cell for i in range(config.n_types)
    ]
    p1 = np.sum([
        p1[i].csum() for i in range(config.n_types)
    ])
    #print('p1:',p1)

    #Term 3
    for t in range(config.n_types):
        phi[t].r2c(out=phi_fourier[0])
        print('np.sum(phi[0]):', np.sum(phi[0][:]))
        print('np.sum(phi_fourier[0]):', np.sum(phi_fourier[0]))
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

            phi_fourier[d].apply(laplacian_transfer, out=Ellipsis)
            phi_fourier[d].c2r(out=phi_laplacian[t][d])
    p2x = [
        config.sigma**2 * V_bar[i] * phi_laplacian[i][0] * volume_per_cell for i in range(config.n_types)
    ]
    p2x = 1/V * np.cumsum(p2x[0])
    print('p2x:',p2x[-1])
    #print('np.asarray(phi_laplacian[0][2]).shape:',sum(sum(sum(phi_laplacian[0][2]))))


    #TESTING
    # phi_t_fft ========= phi_fourier[0] (after applying r2c)
    phi_t_fft = np.fft.fftn(phi[0][:])
    print('np.sum(phi_t_fft)',np.sum(phi_t_fft))

    # k        ========= k[0] 
    freq = np.fft.fftfreq(24, d=1)
    k    = 2*np.pi * freq
    
    # laplaced_x_fft === phi_fourier[0] (after applying laplacian_transfer)
    laplaced_x_fft = -1 * k**2 * phi_t_fft 

    # laplaced_wf_x ==== phi_lap
    laplaced_wf_x = np.fft.ifftn(laplaced_x_fft).real

    # ptest2x   ======== p2x
    ptest2x = config.sigma**2 * V_bar[0] * laplaced_wf_x
    ptest2x = [el*volume_per_cell/V for el in ptest2x]
    ptest2x = np.sum(ptest2x)
    print('ptest2x:',ptest2x)
    return 1.0
