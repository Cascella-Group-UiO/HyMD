from argparse import ArgumentParser
import atexit
from mpi4py import MPI
#from mpi4py.MPI import COMM_WORLD
import cProfile, pstats
import numpy as np
import h5py
import os
import pmesh.pm as pmesh # Particle mesh Routine
import time
import logging

def clog(level, msg, *args, **kwargs):
    comm = kwargs.pop('comm', MPI.COMM_WORLD)
    if comm.rank == 0:
        logging.log(level, msg, *args, **kwargs)

def CONFIGURE_RUNTIME(comm):
    ap = ArgumentParser()
    ap.add_argument("--verbose", default=False, action='store_true')
    ap.add_argument("--profile", default=False, action='store_true')
    ap.add_argument("--destdir", default=".", help="Write to destdir")
    ap.add_argument("confscript", help="CONF.py")
    ap.add_argument("input", help="input.hdf5")
    args = ap.parse_args()

    if comm.rank == 0:
        os.makedirs(args.destdir, exist_ok=True)
    comm.barrier()

    if args.verbose:
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        )

    if args.profile:
        output_file = open(os.path.join(args.destdir,
            'cpu.txt-%06d-of-%06d' % (comm.rank, comm.size))
        , 'w')
        pr = cProfile.Profile()
        def profile_atexit():
            pr.disable()
            # Dump results:
            # - for binary dump
            pr.dump_stats(os.path.join(args.destdir,
                'cpu.prof-%06d-of-%06d' % (comm.rank, comm.size))
            )
            stats = pstats.Stats(pr, stream=output_file)
            stats.sort_stats('time').print_stats()
            output_file.close()

        # TODO: if we have a main function then we can properly do set up and teardown
        # without using atexit.
        atexit.register(profile_atexit)

        pr.enable()

    CONF = {}
    exec(open(args.confscript).read(), CONF)

    for key, value in sorted(CONF.items()):
        if key.startswith("_"):
            continue
        clog(logging.INFO, "%s = %s", key, value)

    return args, CONF

# INITIALIZE MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

args, CONF = CONFIGURE_RUNTIME(comm)

# Set simulation input data

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


np_per_MPI=CONF['Np']//size
if rank==size-1:
    np_cum_mpi=[rank*np_per_MPI,CONF['Np']]
else:
    np_cum_mpi=[rank*np_per_MPI,(rank+1)*np_per_MPI]

grab_extra = 30
_np_cum_mpi = np.empty(shape=(2,), dtype=int)
_np_cum_mpi[0] = max(0, np_cum_mpi[0] - grab_extra)
_np_cum_mpi[1] = min(CONF['Np'], np_cum_mpi[1] + grab_extra)
_p_mpi_range = list(range(_np_cum_mpi[0], _np_cum_mpi[1]))

# Read input
f_input = h5py.File(args.input, 'r')
molecules_flag = False
if 'molecules' in f_input:
    molecules_flag = True
    molecules = f_input['molecules'][_p_mpi_range]
    indices = f_input['indices'][_p_mpi_range]
    bonds = f_input['bonds'][p_mpi_range]

    if rank == 0:
        mpi_range_start = 0
    else:
        mpi_range_start = grab_extra
        while molecules[mpi_range_start - 1] == molecules[mpi_range_start]:
            mpi_range_start += 1
    mpi_range_start = indices[mpi_range_start]

    if rank == size - 1:
        mpi_range_end = CONF['Np']
    else:
        mpi_range_end = grab_extra + np_per_MPI if rank != 0 else np_per_MPI
        while molecules[mpi_range_end - 1] == molecules[mpi_range_end]:
            mpi_range_end += 1
        mpi_range_end = indices[mpi_range_end]
    p_mpi_range = list(range(mpi_range_start, mpi_range_end))
else:
    p_mpi_range = list(range(np_cum_mpi[0], np_cum_mpi[1]))

if molecules_flag:
    molecules = f_input['molecules'][p_mpi_range]

indices = f_input['indices'][p_mpi_range]
r=f_input['coordinates'][-1,p_mpi_range,:]
vel=f_input['velocities'][-1,p_mpi_range,:]
f=np.zeros((len(r),3))
f_bonds=np.zeros((len(r),3))
f_old=np.copy(f)
types=f_input['types'][p_mpi_range]
names = f_input['names'][p_mpi_range]
f_input.close()


bond_energy = 0.0
# print(f'{rank}: {Counter(names)}\n{molecules}\n{indices}\n{bonds}\n')


#Particle-Mesh initialization
pm   = pmesh.ParticleMesh((CONF['Nv'],CONF['Nv'],CONF['Nv']),BoxSize=CONF['L'], dtype='f8',comm=comm)

clog(logging.INFO, "ProcMesh = %s", str(pm.np))

# INTILIZE PMESH ARRAYS

phi_t=[]
force_ds=[]
v_pot=[]

for t in range(CONF['ntypes']):
    phi_t.append(pm.create('real'))
    v_pot.append(pm.create('real'))
    force_ds.append([pm.create('real') for d in range(3)])

# Output files
f_hd5 = h5py.File(os.path.join(args.destdir, 'sim.hdf5-%06d-of%06d' % (comm.rank, comm.size)), 'w')
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
dset_bond_energy=f_hd5.create_dataset("bond_energy",  (CONF['n_frames'],), dtype='Float32')
dset_kin_energy=f_hd5.create_dataset("kin_energy",  (CONF['n_frames'],), dtype='Float32')

dset_names[p_mpi_range]=names
dset_types[p_mpi_range]=types
dset_indices[p_mpi_range]=indices
dset_lengths[0,0,0]=CONF["L"][0]
dset_lengths[0,1,1]=CONF["L"][1]
dset_lengths[0,2,2]=CONF["L"][2]

if rank==0:
    fp_E   = open(os.path.join(args.destdir, 'E.dat'),'w')

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
        if molecules_flag:
            dset_bond_energy[frame]=bond_energy
        dset_time[frame]=CONF['dt']*step

        if molecules_flag:
            fp_E.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(step*CONF['dt'],W+E_kin,W,E_kin,bond_energy,T,mom[0],mom[1],mom[2]))
        else:
            fp_E.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(step*CONF['dt'],W+E_kin,W,E_kin,T,mom[0],mom[1],mom[2]))
        fp_E.flush()



def COMP_FORCE(layouts, r, force_ds, out=None):
    if out is None:
        out = np.zeros((len(r),3))
    else:
        assert out.shape == r.shape
    for t in range(CONF['ntypes']):
        for d in range(3):
            out[types==t, d] = force_ds[t][d].readout(r[types==t], layout=layouts[t])
    return out

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


def COMP_BONDS(f_bonds, r):
    k = CONF['bond_strenght']
    r0 = CONF['bond_length']

    f_bonds.fill(0.0)
    energy = 0.0

    for i, atom_bonds in enumerate(bonds):
        global_index_i = indices[i]
        local_index_i = i
        for bond in atom_bonds:
            if bond != -1:
                global_index_j = bond
                local_index_j = np.squeeze(np.where(indices == global_index_j))

                if global_index_i < global_index_j:
                    continue

                ri = r[local_index_i, :]
                rj = r[local_index_j, :]
                rij = rj - ri
                rij = np.squeeze(rij)

                # Apply periodic boundary conditions to the distance rij
                for dim in range(len(rij)):
                    rij[dim] -= CONF['L'][dim] * np.around(rij[dim] / CONF['L'][dim])
                dr = np.linalg.norm(rij)
                df = - k * (dr - r0)
                f_bond_vector = df * rij / dr
                f_bonds[local_index_i, :] -= f_bond_vector
                f_bonds[local_index_j, :] += f_bond_vector

                energy += 0.5 * k * (dr - r0)**2
    global bond_energy
    bond_energy = comm.allreduce(energy, MPI.SUM)


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
    clog(logging.INFO, "DOMAIN_DECOMP: Total number of particles to be exchanged = %d",
        np.sum(layout.get_exchange_cost()))

    return layout.exchange(r, vel, f, indices, f_old, types)

# First step
if "domain_decomp" in CONF:
    r, vel, f, indices, f_old, types = DOMAIN_DECOMP(r, vel, f, indices, f_old, types)

layouts  = [pm.decompose(r[types==t]) for t in range(CONF['ntypes'])]
UPDATE_FIELD(layouts,True)
f_field = COMP_FORCE(layouts, r, force_ds)
if molecules_flag:
    COMP_BONDS(f_bonds, r)
    f = f_field + f_bonds
else:
    f = f_field

for step in range(CONF['NSTEPS']):
    clog(logging.INFO, "=> step = %d", step)

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
    for t in range(CONF['ntypes']):
        clog(logging.INFO, "GHOSTS: Total number of particles (%d) to be exchanged = %d",
             t, np.sum(layouts[t].get_exchange_cost()))

    if(np.mod(step+1,CONF['quasi'])==0):
        UPDATE_FIELD(layouts, np.mod(step+1,CONF['quasi'])==0)

    f_field = COMP_FORCE(layouts, r, force_ds)
    if molecules_flag:
        COMP_BONDS(f_bonds,r)
        f = f_field + f_bonds
    else:
        f = f_field

    # Integrate velocity
    vel = INTEGRATE_VEL(vel, f/CONF['mass'], f_old/CONF['mass'])

    # Thermostat
    if('T0' in CONF):
        vel = VEL_RESCALE(vel,CONF['tau'])

    # Print trajectory
    if CONF['nprint']>0:
        if(np.mod(step,CONF['nprint'])==0):
            STORE_DATA(step, frame)
            if rank == 0:
                print(f"Step: {step:5} | {100.0 * step / float(CONF['NSTEPS']):5.2f} % | t={step*CONF['dt']:5.4f}  Max(avg) f_bonds: {np.max(f_bonds):10.5f} ({np.mean(f_bonds):5.3f})\tMax(avg) f_field: {np.max(f_field):10.5f} ({np.mean(f_field):5.3f})\tE: {W+E_kin+bond_energy:10.5f}\tEk {E_kin:10.5f}\tW: {W:10.5f}\tEb: {bond_energy:10.5f}")

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

