import logging
import numpy as np
from mpi4py import MPI
from logger import Logger


def compute_field_force(layouts, r, force_mesh, force, types, n_types):
    for t in range(n_types):
        ind = types == t
        for d in range(3):
            force[ind, d] = force_mesh[t][d].readout(r[ind], layout=layouts[t])


def compute_field_energy_q(   
    phi_q_fourier,
    elec_energy_field, #for energy calculation
    field_q_energy,
    comm=MPI.COMM_WORLD,
):
    
    COULK_GMX = 138.935458 

    def transfer_energy(k,v):  ### potential field is electric field / (-ik)  --> potential field * q --> 
        return 4.0 * np.pi * COULK_GMX * np.abs(v)**2  / k.normp(p=2,zeromode=1) ## zeromode = 1 needed here?

    phi_q_fourier.apply(transfer_energy,  kind='wavenumber', out=elec_energy_field)    
    field_q_energy = 0.5 * comm.allreduce(np.sum(elec_energy_field.value))

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
    COULK_GMX = 138.935458 
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
    COULK_GMX = 138.935458 
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
    types,
    config,
    v_ext,
    phi_fourier,
    v_ext_fourier,
    compute_potential=False,
):
    V = np.prod(config.box_size)
    n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh_cells
    for t in range(config.n_types):
        pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
        phi[t] /= volume_per_cell
        phi[t].r2c(out=phi_fourier[t])
        phi_fourier[t].apply(hamiltonian.H, out=Ellipsis)
        phi_fourier[t].c2r(out=phi[t]) 

    # External potential
    for t in range(config.n_types):
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
