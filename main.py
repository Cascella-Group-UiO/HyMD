from mpi4py import MPI
#from mpi4py.MPI import COMM_WORLD
import cProfile, pstats
import numpy as np
import h5py
import sys
import pmesh.pm as pmesh # Particle mesh Routine
import time

pr = cProfile.Profile()
pr.enable()

# INITIALIZE MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set simulation input data
CONF = {}
exec(open(sys.argv[1]).read(), CONF)

# Set seed for all simulations
np.random.seed(0)

# Initialization of simulation variables
kb=2.479/298
if 'T0' in CONF:
    CONF['kbT0']=kb*CONF['T0']
    E_kin0 = kb*3*CONF['Np']*CONF['T0']/2.


Ncells = (CONF['Nv']**3)
CONF['L']      = np.array(CONF['L'])
CONF['V']      = CONF['L'][0]*CONF['L'][1]*CONF['L'][2]
CONF['dV']     = CONF['V']/(CONF['Nv']**3)
if CONF['nprint']==0:
    CONF['n_frames'] = 1
else:
    CONF['n_frames'] = CONF['NSTEPS']//CONF['nprint'] + 1

if 'phi0' not in CONF:
    CONF['phi0']=CONF['Np']/CONF['V']


# Create index ranges for MPI
np_per_MPI=CONF['Np']//size
if rank==size-1:
    p_mpi_range = list(range(rank*np_per_MPI,CONF['Np']))
else:
    p_mpi_range = list(range(rank*np_per_MPI,(rank+1)*np_per_MPI))

# Read input to each mpi-task
f_input = h5py.File(sys.argv[2], 'r',driver='mpio', comm=comm)
r=f_input['coordinates'][-1,p_mpi_range,:]
vel=f_input['velocities'][-1,p_mpi_range,:]
f=np.zeros((len(r),3))
f_old=np.copy(f)
types=f_input['types'][p_mpi_range]
indices=f_input['indicies'][p_mpi_range]
names = f_input['names'][p_mpi_range]
f_input.close()


#Particle-Mesh initialization
pm   = pmesh.ParticleMesh((CONF['Nv'],CONF['Nv'],CONF['Nv']),BoxSize=CONF['L'], dtype='f8',comm=comm)


# INTILIZE PMESH ARRAYS

phi_t=[]
force_ds=[]
v_pot=[]

for t in range(CONF['ntypes']):
    phi_t.append(pm.create('real'))
    v_pot.append(pm.create('real'))
    force_ds.append([pm.create('real') for d in range(3)])

# Output files
f_hd5 = h5py.File('sim.hdf5', 'w',driver='mpio', comm=comm)
dset_pos  = f_hd5.create_dataset("coordinates", (CONF['n_frames'],CONF['Np'],3), dtype="Float32")
dset_vel  = f_hd5.create_dataset("velocities", (CONF['n_frames'],CONF['Np'],3), dtype="Float32")

dset_pos.attrs['units']="nanometers"
dset_time = f_hd5.create_dataset("time", shape=(CONF['n_frames'],), dtype="Float32")

dset_time.attrs['units']="picoseconds"
dset_names=f_hd5.create_dataset("names",  (CONF['Np'],), dtype="S10")
dset_indices=f_hd5.create_dataset("indices",  (CONF['Np'],), dtype='i')
dset_types=f_hd5.create_dataset("types",  (CONF['Np'],), dtype='i')
dset_lengths=f_hd5.create_dataset("cell_lengths",  (1,3,3), dtype='Float32')
dset_tot_energy=f_hd5.create_dataset("tot_energy",  (CONF['n_frames'],), dtype='Float32')
dset_pot_energy=f_hd5.create_dataset("pot_energy",  (CONF['n_frames'],), dtype='Float32')
dset_kin_energy=f_hd5.create_dataset("kin_energy",  (CONF['n_frames'],), dtype='Float32')

dset_names[p_mpi_range]=names
dset_types[p_mpi_range]=types
dset_indices[p_mpi_range]=indices
dset_lengths[0,0,0]=CONF["L"][0]
dset_lengths[0,1,1]=CONF["L"][1]
dset_lengths[0,2,2]=CONF["L"][2]

if rank==0:
    fp_E   = open('E.dat','w')

#FUNCTION DEFINITIONS



def INTEGERATE_POS(x, vel, a):
# Velocity Verlet integration I
    return x + vel*CONF['dt'] + 0.5*a*CONF['dt']**2

def INTEGRATE_VEL(vel, a, a_old):
# Velocity Verlet integration II
    return vel + 0.5*(a+a_old)*CONF['dt']

def VEL_RESCALE(vel, tau):
    """ Velocity rescale thermostat, see 
        https://doi.org/10.1063/1.2408420
            Parameters
            ----------
            vel : [N_mpi,3] float array beloning to MPI-task
            tau : float, relaxation time of thermostat.
            Returns:
            ----------
            out : vel
            Thermostatted velocity array.  
    """    

    # INITIAL KINETIC ENERGY
    E_kin  = comm.allreduce(0.5*CONF['mass']*np.sum(vel**2))


    #BERENDSEN LIKE TERM
    d1 = (E_kin0-E_kin)*CONF['dt']/tau

    # WIENER NOISE
    dW = np.sqrt(CONF['dt'])*np.random.normal()

    # STOCHASTIC TERM
    d2 = 2*np.sqrt(E_kin*E_kin0/(3*CONF['Np']))*dW/np.sqrt(tau)

    # TARGET KINETIC ENERGY
    E_kin_target = E_kin + d1 + d2

    #VELOCITY SCALING
    alpha = np.sqrt(E_kin_target/E_kin)

    vel=vel*alpha

    return vel

def STORE_DATA(step,frame):
    """ prints positions, velocities and energies sim.hdf5
            Parameters
            ----------
            step : int
                current MD step 
            frame : int
                current data frame used store data.
    """
    ind_sort=np.argsort(indices)
    dset_pos[frame,indices[ind_sort]] = r[ind_sort]
    dset_vel[frame,indices[ind_sort]] = vel[ind_sort]


    if rank==0:
        dset_tot_energy[frame]=W+E_kin
        dset_pot_energy[frame]=W
        dset_kin_energy[frame]=E_kin
        dset_time[frame]=CONF['dt']*step

        fp_E.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(step*CONF['dt'],W+E_kin,W,E_kin,T,mom[0],mom[1],mom[2]))
        fp_E.flush()



def COMP_FORCE(layouts, f, r, force_ds):
    for t in range(CONF['ntypes']):
        for d in range(3):
            f[types==t, d] = force_ds[t][d].readout(r[types==t], layout=layouts[t])

def COMP_PRESSURE():
    # COMPUTES HPF PRESSURE FOR EACH MPI TASK
    P=[]
    p_val=[]
    for d in range(3):

        p = CONF['w'](phi_t)
        for t in range(CONF['ntypes']):
            p += -v_pot[t]*phi_t[t] + (v_pot[t].r2c(out=Ellipsis).apply(CONF['kdHdk'])[d]).c2r(out=Ellipsis)


        P.append(p*CONF['dV']/CONF['V'])
        p_val.append(p.csum())
    return np.array(p_val)

def UPDATE_FIELD(layouts,comp_v_pot=False):

    # Filtered density
    for t in range(CONF['ntypes']):
        phi_t[t] = (pm.paint(r[types==t], layout=layouts[t])/CONF['dV']).r2c(out=Ellipsis).apply(CONF['H'], out=Ellipsis).c2r(out=Ellipsis)
        
    # External potential
    for t in range(CONF['ntypes']):
        v_p_fft=CONF['V_EXT'][t](phi_t).r2c(out=Ellipsis).apply(CONF['H'], out=Ellipsis)

        # Derivative of external potential
        for d in range(3):
            def force_transfer_function(k, v, d=d):
                return -k[d] * 1j * v
            force_ds[t][d]=(v_p_fft.copy().apply(force_transfer_function).c2r(out=Ellipsis))

        if(comp_v_pot):
            v_pot[t]=v_p_fft.c2r(out=Ellipsis)



def COMPUTE_ENERGY():
    E_hpf = 0
    E_kin = pm.comm.allreduce(0.5*CONF['mass']*np.sum(vel**2))
    W = CONF['w'](phi_t)*CONF['dV']

    W = W.csum()

    return E_hpf,E_kin,W



# pr = cProfile.Profile()
# pr.enable()

if rank==0:
    start_t = time.time()


def DOMAIN_DECOMP(r, vel, f, indices, f_old, types):
    # E_kin = pm.comm.allreduce(0.5*CONF['mass']*np.sum(vel**2))
    # if pm.comm.rank == 0:
    #     print('Kin energy before domain', E_kin)
    #Particles are exchanged between mpi-processors
    layout=pm.decompose(r,smoothing=0)
    r=layout.exchange(r)
    vel=layout.exchange(vel)
    f=layout.exchange(f)
    indices=layout.exchange(indices)
    f_old=layout.exchange(f_old)
    types=layout.exchange(types)

    #E_kin = pm.comm.allreduce(0.5*CONF['mass']*np.sum(vel**2))
    # if pm.comm.rank == 0:
    #     print('Kin energy after domain', E_kin)
    return r, vel, f, indices, f_old, types

# First step
if "domain_decomp" in CONF:
    r, vel, f, indices, f_old, types = DOMAIN_DECOMP(r, vel, f, indices, f_old, types)

layouts  = [pm.decompose(r[types==t]) for t in range(CONF['ntypes'])]
UPDATE_FIELD(layouts,True)
COMP_FORCE(layouts, f, r, force_ds)

for step in range(CONF['NSTEPS']):

    if CONF['nprint']>0:
        frame=step//CONF['nprint']

        if(np.mod(step,CONF['nprint'])==0):
            E_hpf, E_kin,W = COMPUTE_ENERGY()
            T     =   2*E_kin/(kb*3*CONF['Np'])
            mom=pm.comm.allreduce(np.sum(vel,axis=0))

    f_old = np.copy(f)

    #Integrate positions
    r     = INTEGERATE_POS(r, vel, f/CONF['mass'])

    #PERIODIC BC
    r     = np.mod(r, CONF['L'][None,:])

    if "domain_decomp" in CONF:
        r, vel, f, indices, f_old, types = DOMAIN_DECOMP(r, vel, f, indices, f_old, types)

    # Particles are kept on mpi-task
    layouts  = [pm.decompose(r[types==t]) for t in range(CONF['ntypes'])]

    if(np.mod(step+1,CONF['quasi'])==0):
        UPDATE_FIELD(layouts, np.mod(step+1,CONF['quasi'])==0)


    COMP_FORCE(layouts,f,r,force_ds)


    # Integrate velocity
    vel = INTEGRATE_VEL(vel, f/CONF['mass'], f_old/CONF['mass'])

    # Thermostat
    if('T0' in CONF):
        vel = VEL_RESCALE(vel,tau)

    # Print trajectory
    if CONF['nprint']>0:
        if(np.mod(step,CONF['nprint'])==0):
            STORE_DATA(step, frame)


# End simulation
if rank==0:
    #print('Simulation time elapsed:', time.time()-start_t, "for",size, "cpus")
    print(size, time.time()-start_t)



if CONF['nprint']>0:
    UPDATE_FIELD(layouts,True)
    frame=(step+1)//CONF['nprint']
    E_hpf, E_kin,W = COMPUTE_ENERGY()
    T     =   2*E_kin/(kb*3*CONF['Np'])
    STORE_DATA(step,frame)

f_hd5.close()
pr.disable()

# Dump results:
# - for binary dump
pr.dump_stats('cpu_%d.prof' %comm.rank)

with open( 'cpu_%d.txt' %comm.rank, 'w') as output_file:
    sys.stdout = output_file
    pr.print_stats( sort='time' )
    sys.stdout = sys.__stdout__

