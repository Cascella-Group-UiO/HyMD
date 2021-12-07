import logging
import numpy as np
from mpi4py import MPI
from logger import Logger


def compute_field_force(layouts, r, force_mesh, force, types, config):
    for t in range(config.n_types):
        ind = types == t
        for d in range(3):
            force[ind, d] = force_mesh[t][d].readout(r[ind], layout=layouts[t])

def compute_field_force_1d_with_potential(layouts, r, force_mesh, force, types, config):
    for t in range(config.n_types):
        ind = types == t

        #for d in range(3):
        #    force[ind, d] = force_mesh[t][d].readout(r[ind], layout=layouts[t])
        ##--> only x dimension 
        d = 0 
        force[ind, d] = force_mesh[t][d].readout(r[ind], layout=layouts[t])
        #print(force[ind, d])
        ############## simple homonic potential 
        ### 1 particle with potential at  v= 100(r-ro)^2
        #r0 =  2.0
        #rx = r[ind][0][0] 
        ##force[ind, d] += -10.0*(rx-r0)#--->   
        ## apply pbc to the distance rx-r0
        #dr = (rx-r0) - config.box_size[0] * np.around( (rx-r0) / config.box_size[0])
        #df = -10.0 * dr
        #force[ind, d] += df
        ###############

        ############## two gaussain potential  
        def gaussian(a, x, mu, sig):
            return a*np.exp(- (x - mu)**2 / (2 * sig**2) )
        def gaussian_dev(a, x, mu, sig):
            return a*np.exp(- (x - mu)**2 / (2 * sig**2) )* -1.0*(x-mu)/sig**2
        def gaussian_f(a, x, mu, sig):
            return a*np.exp(- (x - mu)**2 / (2 * sig**2) )* (x-mu)/sig**2
        
        ### works set 1 
        a1 = -30.0  #-10.0  
        mu1 = 3.0
        sig1 = 0.5
        a2 = -20.0 #-50.0 #-30.0 
        mu2 = 5.0
        sig2 = 0.5

        ### works set 2 
        a1 = -30.0  #-10.0  
        mu1 = 2.0
        sig1 = 0.5
        a2 = -20.0 #-50.0 #-30.0 
        mu2 = 5.0
        sig2 = 0.5
        ###
        
        ### works set 3
        #a1 = -50.0  #-10.0  
        #mu1 = 2.0
        #sig1 = 0.5
        #a2 = -20.0 #-50.0 #-30.0 
        #mu2 = 5.0
        #sig2 = 0.5
        ###
        
        ## works set 5
        a1 = -30.0  #-10.0  
        mu1 = 2.0
        sig1 = 0.5 #0.5 0.4 1.0 work #0.2 narrow, does not work 
        a2 = -20.0 #-50.0 #-30.0 
        mu2 = 6.0
        sig2 = 0.5 #0.5 0.4 1.0 work #0.2 narrow, does not work 
        ##

        rx = r[ind][0][0]  
        x = np.mod(rx, config.box_size[0]) ## correct pbc.. 

        #v = gaussian(a1, x, mu1, sig1) + gaussian(a2, x, mu2, sig2)
        df =  gaussian_f(a1, x, mu1, sig1) + gaussian_f(a2, x, mu2, sig2)
        force[ind, d] += df
        




#def compute_field_energy_q(   
#    phi_q_fourier,
#    elec_energy_field, #for energy calculation
#    field_q_energy,
#    comm=MPI.COMM_WORLD,
#):
#    
#    COULK_GMX = 138.935458 
#
#    def transfer_energy(k,v):  ### potential field is electric field / (-ik)  --> potential field * q --> 
#        return 4.0 * np.pi * COULK_GMX * np.abs(v)**2  / k.normp(p=2,zeromode=1) ## zeromode = 1 needed here?
#
#    phi_q_fourier.apply(transfer_energy,  kind='wavenumber', out=elec_energy_field)    
#    field_q_energy = 0.5 * comm.allreduce(np.sum(elec_energy_field.value))
#
#    return field_q_energy.real


def compute_field_energy_q(   
    config,
    phi_q_fourier,
    elec_energy_field, #for energy calculation
    field_q_energy,
    comm=MPI.COMM_WORLD,
):
    
    COULK_GMX = 138.935458 / config.dielectric_const

    def transfer_energy(k,v):  ### potential field is electric field / (-ik)  --> potential field * q --> 
        return 4.0 * np.pi * COULK_GMX * np.abs(v)**2  / k.normp(p=2,zeromode=1) ## zeromode = 1 needed here?

    phi_q_fourier.apply(transfer_energy,  kind='wavenumber', out=elec_energy_field)    

    V = np.prod(config.box_size)

    field_q_energy = 0.5 * V * comm.allreduce(np.sum(elec_energy_field.value))

    return field_q_energy.real


def update_field_force_q(
    charges,# charge
    phi_q,  # chage density
    phi_q_fourier,   
    elec_field_fourier, #for force calculation 
    elec_field,     
    elec_forces,    
    layout_q, #### general terms  
    pm,
    positions,  
    config,
):
    """
    - added for the simple piosson equation eletrostatics (follow PIC)
    - this funciton get the electrostatic forces 
    - refering to the test-pure-sphere-new.py, this inlcudes: 
        [O] hpf_init_simple_gmx_units(grid_num,box_size,coords,charges,masses)
        ## ^----- define the pm and layout, already out outside this fucntion 
        [Y] gen_qe_hpf_use_self(out_phiq_paraview_file)
        [Y] calc_phiq_fft_use_self_applyH_checkq(grid_num, out_phiq_paraview_file2 ) 
        [Y] poisson_solver(calc_energy, out_elec_field_paraview_file)
        [Y] compute_electric_field_on_particle()
        [Y] compute_electric_force_on_particle()
    """
    ## basic setup 
    V = np.prod(config.box_size)
    n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh_cells
    
    ## paint  ## pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
    ## old protocol in gen_qe_hpf_use_self
    pm.paint(positions, layout=layout_q, mass=charges, out=phi_q) ## 
    ## scale and fft  
    ## old protocol in gen_qe_hpf_use_self
    phi_q /= volume_per_cell 
    phi_q.r2c(out=phi_q_fourier)
    
    def phi_transfer_function(k, v): 
        return v * np.exp(-0.5*config.sigma**2*k.normp(p=2, zeromode=1))
    phi_q_fourier.apply(phi_transfer_function, out=phi_q_fourier) 
    ## ^------ use the same gaussian as the \kai interaciton 
    ## ^------ tbr; phi_transfer_funciton by hamiltonian.H ??
    phi_q_fourier.c2r(out=phi_q) ## this phi_q is after applying the smearing function 
    
    ## electric field via solving poisson equation 
    ## old protol in poisson_solver 
    _SPACE_DIM = 3     
    
    COULK_GMX = 138.935458 / config.dielectric_const
    
    for _d in np.arange(_SPACE_DIM):    
        def poisson_transfer_function(k, v, d=_d):
            return - 1j * k[_d] * 4.0 * np.pi * COULK_GMX * v / k.normp(p=2,zeromode=1)
            ######return - 1j * k[_d] * 4.0 * np.pi * v /k.normp(p=2) #hymd.py:173: RuntimeWarning: invalid value encountered in true_divide
        phi_q_fourier.apply(poisson_transfer_function, out = elec_field_fourier[_d])
        elec_field_fourier[_d].c2r(out=elec_field[_d])

    ## calculate electric forces on particles  
    ## old protocol in compute_electric_force_on_particle_onestep
    for _d in np.arange(_SPACE_DIM):
        elec_forces[:,_d] = charges * (elec_field[_d].readout(positions, layout=layout_q))
        ###^------ here the use the column, as the elec_forces are defined as (N,3) dimension
    



def update_field_force_energy_q(
    charges,# charge
    phi_q,  # chage density
    phi_q_fourier,   
    elec_field_fourier, #for force calculation 
    elec_field,     
    elec_forces,    
    elec_energy_field, # for energy calculation 
    field_q_energy, # electric energy
    layout_q, #### general terms  
    pm,
    positions,  
    config,
    compute_energy=False,
    comm=MPI.COMM_WORLD,
    ):
    """
    - added for the simple piosson equation eletrostatics (follow PIC)
    - this funciton get the electrostatic forces 
    - refering to the test-pure-sphere-new.py, this inlcudes: 
        [O] hpf_init_simple_gmx_units(grid_num,box_size,coords,charges,masses)
        ## ^----- define the pm and layout, already out outside this fucntion 
        [Y] gen_qe_hpf_use_self(out_phiq_paraview_file)
        [Y] calc_phiq_fft_use_self_applyH_checkq(grid_num, out_phiq_paraview_file2 ) 
        [Y] poisson_solver(calc_energy, out_elec_field_paraview_file)
        [Y] compute_electric_field_on_particle()
        [Y] compute_electric_force_on_particle()
    """
    ## basic setup 
    V = np.prod(config.box_size)
    n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh_cells
    
    ## paint  ## pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
    ## old protocol in gen_qe_hpf_use_self
    pm.paint(positions, layout=layout_q, mass=charges, out=phi_q) ## 
    ## scale and fft  
    ## old protocol in gen_qe_hpf_use_self
    phi_q /= volume_per_cell 
    phi_q.r2c(out=phi_q_fourier)
    
    def phi_transfer_function(k, v): 
        return v * np.exp(-0.5*config.sigma**2*k.normp(p=2, zeromode=1))
    phi_q_fourier.apply(phi_transfer_function, out=phi_q_fourier) 
    ## ^------ use the same gaussian as the \kai interaciton 
    ## ^------ tbr; phi_transfer_funciton by hamiltonian.H ??
    phi_q_fourier.c2r(out=phi_q) ## this phi_q is after applying the smearing function 
    
    ## electric field via solving poisson equation 
    ## old protol in poisson_solver 
    _SPACE_DIM = 3     
    
    COULK_GMX = 138.935458 / config.dielectric_const

    for _d in np.arange(_SPACE_DIM):    
        def poisson_transfer_function(k, v, d=_d):
            return - 1j * k[_d] * 4.0 * np.pi * COULK_GMX * v / k.normp(p=2,zeromode=1)
            ######return - 1j * k[_d] * 4.0 * np.pi * v /k.normp(p=2) #hymd.py:173: RuntimeWarning: invalid value encountered in true_divide
        phi_q_fourier.apply(poisson_transfer_function, out = elec_field_fourier[_d])
        elec_field_fourier[_d].c2r(out=elec_field[_d])

    ## calculate electric forces on particles  
    ## old protocol in compute_electric_force_on_particle_onestep
    for _d in np.arange(_SPACE_DIM):
        elec_forces[:,_d] = charges * (elec_field[_d].readout(positions, layout=layout_q))
        ###^------ here the use the column, as the elec_forces are defined as (N,3) dimension
    
    ## calculate electric energy in Fourier space
    ## old protocol in poisson_solver [ if calc_energy: ] block
    if compute_energy:
        def transfer_energy(k,v):  ### potential field is electric field / (-ik)  --> potential field * q --> 
            return 4.0 * np.pi * COULK_GMX * np.abs(v)**2  / k.normp(p=2,zeromode=1) ## zeromode = 1 needed here?
        phi_q_fourier.apply(transfer_energy,  kind='wavenumber', out=elec_energy_field)
        
        field_q_energy = 0.5 * comm.allreduce(np.sum(elec_energy_field.value))

    return field_q_energy.real


def update_field(
    phi,
    layouts,
    force_mesh,
    hamiltonian,
    pm,
    positions,
    #masses, # xinmeng <-------
    types,
    config,
    v_ext,
    phi_fourier,
    v_ext_fourier,
    compute_potential=False,
): 
    ## for simplicity, assume there is mass 
    ## Masses!!!! not used ....
    ## ---- used config.kai_types_id instead of range(config.n_types)
    ## 
    V = np.prod(config.box_size)
    n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh_cells
    for t in config.kai_types_id : #xinmeng !!! 
        #for t in range(config.n_types):  
        
        ###### !!!!! 
        #if t == 11:
        #    #pm.paint(positions[types == t], mass=np.zeros(len(positions[types == t])), layout=layouts[t], out=phi[t])
        #    pm.paint(positions[types == t], mass=np.zeros(len(positions[types == t])), layout=layouts[t], out=phi[t])
        #else:
        #    pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
        
        #pm.paint(positions[types == t], mass=masses[types==t], layout=layouts[t], out=phi[t])
        pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
        
        phi[t] /= volume_per_cell
        phi[t].r2c(out=phi_fourier[t])
        phi_fourier[t].apply(hamiltonian.H, out=Ellipsis)
        phi_fourier[t].c2r(out=phi[t]) 

    # External potential 
    for t in config.kai_types_id: # xinmeng !!! 
        #for t in range(config.n_types): 
        hamiltonian.v_ext[t](phi).r2c(out=v_ext_fourier[0])
        v_ext_fourier[0].apply(hamiltonian.H, out=Ellipsis)
        np.copyto(
            v_ext_fourier[1].value, v_ext_fourier[0].value, casting="no", where=True
        )
        np.copyto(
            v_ext_fourier[2].value, v_ext_fourier[0].value, casting="no", where=True
        )
        if compute_potential:
            np.copyto(
                v_ext_fourier[3].value, v_ext_fourier[0].value, casting="no", where=True
            )

        # Differentiate the external potential in fourier space
        for d in range(3):

            def force_transfer_function(k, v, d=d):
                return -k[d] * 1j * v

            v_ext_fourier[d].apply(force_transfer_function, out=Ellipsis)
            v_ext_fourier[d].c2r(out=force_mesh[t][d])

        if compute_potential:
            v_ext_fourier[3].c2r(out=v_ext[t])

def update_field_with_ghost(
    step,
    phi,
    phi_ghost,
    layouts,
    force_mesh,
    hamiltonian,
    pm,
    positions,
    #masses, # xinmeng <-------
    types,
    config,
    v_ext,
    phi_fourier,
    v_ext_fourier,
    compute_potential=False,
): 
    
    ## 
    V = np.prod(config.box_size)
    n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh_cells
    for t in config.kai_types_id : #xinmeng !!! 
        #for t in range(config.n_types):  
        
        ###### !!!!! 
        #if t == 11:
        #    #pm.paint(positions[types == t], mass=np.zeros(len(positions[types == t])), layout=layouts[t], out=phi[t])
        #    pm.paint(positions[types == t], mass=np.zeros(len(positions[types == t])), layout=layouts[t], out=phi[t])
        #else:
        #    pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
        
        #pm.paint(positions[types == t], mass=masses[types==t], layout=layouts[t], out=phi[t])
        pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
        
        phi[t] /= volume_per_cell

        #print('here', phi[t], np.sum(phi[t]), np.sum(phi[t])*volume_per_cell)
        

        phi[t].r2c(out=phi_fourier[t])
        phi_fourier[t].apply(hamiltonian.H, out=Ellipsis)
        phi_fourier[t].c2r(out=phi[t]) 
        
        #print('there', phi[t], np.sum(phi[t]), np.sum(phi[t])*volume_per_cell)
        #exit()
        
    # External potential 
    for t in config.kai_types_id: # xinmeng !!! 
        #### thanks to manuel 2021-11-30 
        # if w = w_0 + \kai \rho_L \rho_L'
        # first is the v from w_0
        v_full = hamiltonian.v_ext[t](phi)
        #print('t in kai_types_id', t,config.kai_types_id )
        ### comment this extra part, with behave 'normal'
        # second consider the extra part
        if config.meta_ghost_types:            
            if t in config.meta_ghost_types_id:
                #print(np.sum(phi[t])*volume_per_cell)
                
                ### test 
                #phi_ghost += phi[t] #/config.meta_ghost_stat_step
                ###
                #print('xx', t)
                #v_full += config.meta_ghost_kai * phi_ghost
                #print(np.max(v_full))

                ###
                #v_full += config.meta_ghost_weight * phi_ghost/config.rho0
                v_full = config.meta_ghost_weight * phi_ghost/config.rho0
        
        #if np.mod(step, config.meta_ghost_flush) == 0 :## here
        #    #v_ext_fourier[0].c2r(out=v_ext[t])  
        #    np.save(f"ghost_density_{step}.npy", config.meta_ghost_weight * phi_ghost/config.rho0)
        
                
        v_full.r2c(out=v_ext_fourier[0])

        #hamiltonian.v_ext[t](phi).r2c(out=v_ext_fourier[0])
        v_ext_fourier[0].apply(hamiltonian.H, out=Ellipsis)

        np.copyto(
            v_ext_fourier[1].value, v_ext_fourier[0].value, casting="no", where=True
        )
        np.copyto(
            v_ext_fourier[2].value, v_ext_fourier[0].value, casting="no", where=True
        )

        #### v after H 
        if np.mod(step, config.meta_ghost_flush) == 0 :## here
            np.copyto(
                v_ext_fourier[3].value, v_ext_fourier[0].value, casting="no", where=True
            )
            v_ext_fourier[3].c2r(out=v_ext[t])
            np.save(f"ghost_density_{step}.npy", v_ext[t])
        
        # Differentiate the external potential in fourier space
        for d in range(3):

            def force_transfer_function(k, v, d=d):
                return -k[d] * 1j * v

            v_ext_fourier[d].apply(force_transfer_function, out=Ellipsis)
            v_ext_fourier[d].c2r(out=force_mesh[t][d])

        
        
def update_field_ghost_old(
    step,
    phi,
    phi_ghost,
    v_ghost,
    layouts,
    force_mesh,
    hamiltonian,
    pm,
    positions,
    #masses, # xinmeng <-------
    types,
    config,
    v_ext,
    phi_fourier,
    v_ext_fourier,
    compute_potential=False,
): 
      
    ### 
    #V = np.prod(config.box_size)
    #n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    #volume_per_cell = V / n_mesh_cells

    #v_ghost *= 0.0  
    #if np.mod(step, config.meta_ghost_window ) == 0 :
    #    ## apply the ghost interaction
    #
    

    v_ghost *= 0.0  

    for t in config.meta_ghost_types_id:  
        v_full = config.meta_ghost_weight * phi_ghost/config.rho0
        v_full.r2c(out=v_ext_fourier[0])
        v_ext_fourier[0].apply(hamiltonian.H, out=Ellipsis)
        np.copyto(
            v_ext_fourier[1].value, v_ext_fourier[0].value, casting="no", where=True
        )
        np.copyto(
            v_ext_fourier[2].value, v_ext_fourier[0].value, casting="no", where=True
        )
        ## for external bias potential 
        np.copyto(
            v_ext_fourier[3].value, v_ext_fourier[0].value, casting="no", where=True
        )
        v_ext_fourier[3].c2r(out=v_ext[t])
        #v_ghost += v_ext[t] ## this will contain several types of beads 
        v_ghost = v_ext[t] ## this will contain several types of beads 
        
        # Differentiate the external potential in fourier space
        for d in range(3):
            def force_transfer_function(k, v, d=d):
                return -k[d] * 1j * v
            v_ext_fourier[d].apply(force_transfer_function, out=Ellipsis)
            v_ext_fourier[d].c2r(out=force_mesh[t][d])        
    
    for t in config.meta_ghost_types_id: 
        #pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
        #phi[t] /= volume_per_cell
        ##print('here', phi[t], np.sum(phi[t]), np.sum(phi[t])*volume_per_cell)
        #phi[t].r2c(out=phi_fourier[t])
        #phi_fourier[t].apply(hamiltonian.H, out=Ellipsis)
        #phi_fourier[t].c2r(out=phi[t]) 
        
        phi_ghost += phi[t] # just add the raw density; 

    if np.mod(step, config.meta_ghost_flush) == 0 :## here
        np.save(f"ghost_potential_{step}.npy", v_ghost)
        np.save(f"real_density_{step}.npy", sum(phi))
        
        
def update_field_ghost_nowelltempered(
    step,
    phi,
    phi_ghost,
    v_ghost,
    layouts,
    force_mesh,
    hamiltonian,
    pm,
    positions,
    #masses, # xinmeng <-------
    types,
    config,
    v_ext,
    phi_fourier,
    v_ext_fourier,
    compute_potential=False,
): 
    ###### omited before  ########### 
    V = np.prod(config.box_size)
    n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh_cells
    for t in config.kai_types_id : #xinmeng !!! 
        #for t in range(config.n_types):  
        
        ###### !!!!! 
        #if t == 11:
        #    #pm.paint(positions[types == t], mass=np.zeros(len(positions[types == t])), layout=layouts[t], out=phi[t])
        #    pm.paint(positions[types == t], mass=np.zeros(len(positions[types == t])), layout=layouts[t], out=phi[t])
        #else:
        #    pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
        
        #pm.paint(positions[types == t], mass=masses[types==t], layout=layouts[t], out=phi[t])
        pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
        
        phi[t] /= volume_per_cell

        #print('here', phi[t], np.sum(phi[t]), np.sum(phi[t])*volume_per_cell)
        

        phi[t].r2c(out=phi_fourier[t])
        phi_fourier[t].apply(hamiltonian.H, out=Ellipsis)
        phi_fourier[t].c2r(out=phi[t]) 
        
        #print('there', phi[t], np.sum(phi[t]), np.sum(phi[t])*volume_per_cell)
        #exit()
    ###################################
    
    _v_ghost = v_ghost*0.0  
    
    for t in config.meta_ghost_types_id:  
        
        #phi_ghost 
        #print()

        v_full = config.meta_ghost_weight * phi_ghost/config.rho0


        v_full.r2c(out=v_ext_fourier[0])
        v_ext_fourier[0].apply(hamiltonian.H, out=Ellipsis)
        np.copyto(
            v_ext_fourier[1].value, v_ext_fourier[0].value, casting="no", where=True
        )
        np.copyto(
            v_ext_fourier[2].value, v_ext_fourier[0].value, casting="no", where=True
        )
        ## for external bias potential 
        np.copyto(
            v_ext_fourier[3].value, v_ext_fourier[0].value, casting="no", where=True
        )
        v_ext_fourier[3].c2r(out=v_ext[t])
        _v_ghost += v_ext[t] ## this is targeting for several types of beads ; not tested 
        #v_ghost = v_ext[t] ## this will contain one 
        
        # Differentiate the external potential in fourier space
        for d in range(3):
            def force_transfer_function(k, v, d=d):
                return -k[d] * 1j * v
            v_ext_fourier[d].apply(force_transfer_function, out=Ellipsis)
            v_ext_fourier[d].c2r(out=force_mesh[t][d])     

    v_ghost = _v_ghost
    
    for t in config.meta_ghost_types_id: 
        #pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
        #phi[t] /= volume_per_cell
        ##print('here', phi[t], np.sum(phi[t]), np.sum(phi[t])*volume_per_cell)
        #phi[t].r2c(out=phi_fourier[t])
        #phi_fourier[t].apply(hamiltonian.H, out=Ellipsis)
        #phi_fourier[t].c2r(out=phi[t]) 
        
        phi_ghost += phi[t] # just add the raw density; 

    if np.mod(step, config.meta_ghost_flush) == 0 :## here
        np.save(f"ghost_potential_{step}.npy", v_ghost)
        np.save(f"real_density_{step}.npy", sum(phi))
    
    ### sees return or not does not matter 
    return phi_ghost

def update_field_ghost(
    step,
    phi,
    phi_ghost,
    v_ghost,
    layouts,
    force_mesh,
    hamiltonian,
    pm,
    positions,
    #masses, # xinmeng <-------
    types,
    config,
    v_ext,
    phi_fourier,
    v_ext_fourier,
    compute_potential=False,
):  
    ### with well-tempered by default... 2021-12-06
    
    ### 
    #V = np.prod(config.box_size)
    #n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    #volume_per_cell = V / n_mesh_cells

    #v_ghost *= 0.0  
    #if np.mod(step, config.meta_ghost_window ) == 0 :
    #    ## apply the ghost interaction
    #
    #if np.mod(step, config.meta_ghost_flush) == 0 :
    #    print( v_ghost)

    
    _v_ghost = v_ghost.copy()*0.0  
    
    for t in config.meta_ghost_types_id:  
        
        ##print(config.meta_ghost_weight)
        #_tempered_meta_ghost_weight = config.meta_ghost_weight*np.exp(-(phi_ghost / config.meta_bias_temp))
        _tempered_meta_ghost_weight = config.meta_ghost_weight*np.exp(-(v_ghost / config.meta_bias_temp/ config.R))
        #
        #if np.mod(step, config.meta_ghost_flush) == 0 :
        #    print(_tempered_meta_ghost_weight)
        
        
        #v_full = config.meta_ghost_weight * phi_ghost/config.rho0
        #v_full = _tempered_meta_ghost_weight * phi_ghost/config.rho0
        
        #v_full = config.meta_ghost_weight * phi[t] /config.rho0  + phi_ghost
        
        #v_full = _tempered_meta_ghost_weight * phi[t] /config.rho0  + phi_ghost
        v_full =  phi_ghost
        
        if np.mod(step, config.meta_ghost_flush) == 0 :## here
            print(step, _tempered_meta_ghost_weight, np.sum(v_full), np.sum(phi_ghost), np.sum(v_ghost))
            # exit()
        
        v_full.r2c(out=v_ext_fourier[0])

        v_ext_fourier[0].apply(hamiltonian.H, out=Ellipsis)
        np.copyto(
            v_ext_fourier[1].value, v_ext_fourier[0].value, casting="no", where=True
        )
        np.copyto(
            v_ext_fourier[2].value, v_ext_fourier[0].value, casting="no", where=True
        )
        ## for external bias potential 
        np.copyto(
            v_ext_fourier[3].value, v_ext_fourier[0].value, casting="no", where=True
        )
        v_ext_fourier[3].c2r(out=v_ext[t])
        
        _v_ghost += v_ext[t] ## this is targeting for several types of beads ; not tested 
        #v_ghost = v_ext[t] ## this will contain one 
        
        # Differentiate the external potential in fourier space
        for d in range(3):
            def force_transfer_function(k, v, d=d):
                return -k[d] * 1j * v
            v_ext_fourier[d].apply(force_transfer_function, out=Ellipsis)
            v_ext_fourier[d].c2r(out=force_mesh[t][d])     

    v_ghost = _v_ghost
    
    
    for t in config.meta_ghost_types_id: 
        #pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
        #phi[t] /= volume_per_cell
        ##print('here', phi[t], np.sum(phi[t]), np.sum(phi[t])*volume_per_cell)
        #phi[t].r2c(out=phi_fourier[t])
        #phi_fourier[t].apply(hamiltonian.H, out=Ellipsis)
        #phi_fourier[t].c2r(out=phi[t]) 
        
        #phi_ghost += phi[t] # just add the raw density; 
        
        ##phi_ghost += config.meta_ghost_weight * phi[t] /config.rho0
        phi_ghost += _tempered_meta_ghost_weight * phi[t] /config.rho0
        
    
    if np.mod(step, config.meta_ghost_flush) == 0 :## here
        #np.save(f"ghost_potential_{step}.npy", v_ghost)
        np.save(f"estimated_free_energy_{step}.npy", v_ghost*( -(config.target_temperature+ config.meta_bias_temp)/config.meta_bias_temp ))
        #np.save(f"estimated_free_energy_{step}.npy", -v_ghost)
        np.save(f"real_density_{step}.npy", sum(phi))

    ### this phi_ghost is the trace of history 
    ### not the density but v hiostry config.meta_ghost_weight /config.rho0
    return v_ghost, phi_ghost    
    

def compute_field_and_kinetic_energy(
    phi,
    velocity,
    hamiltonian,
    positions,
    types,
    v_ext,
    config,
    layouts,
    comm=MPI.COMM_WORLD,
):
    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells

    w = hamiltonian.w(phi) * volume_per_cell
    field_energy = w.csum()
    kinetic_energy = comm.allreduce(0.5 * config.mass * np.sum(velocity ** 2))
    return field_energy, kinetic_energy


def domain_decomposition(
    positions,
    pm,
    *args,
    molecules=None,
    bonds=None,
    verbose=0,
    comm=MPI.COMM_WORLD
):
    if molecules is not None:
        assert bonds is not None, "bonds must be provided when molecules are present"
        unique_molecules = np.sort(np.unique(molecules))
        molecules_com = np.empty_like(positions)
        for m in unique_molecules:
            ind = molecules == m
            r = positions[ind, :][0, :]
            molecules_com[ind, :] = r
        layout = pm.decompose(molecules_com, smoothing=0)
        args = (*args, bonds, molecules)
    else:
        layout = pm.decompose(positions, smoothing=0)
    if verbose > 1:
        Logger.rank0.log(
            logging.INFO,
            "DOMAIN_DECOMP: Total number of particles to be exchanged = %d",
            np.sum(layout.get_exchange_cost()),
        )
    return layout.exchange(positions, *args)
    