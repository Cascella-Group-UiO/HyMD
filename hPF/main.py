from argparse import ArgumentParser
import atexit
from mpi4py import MPI
import cProfile
import pstats
import numpy as np
import h5py
import os
import pmesh.pm as pmesh
import time
import logging
from types import ModuleType as moduleobj

from distribute_input import distribute_input
from force import prepare_bonds, compute_bond_forces, compute_angle_forces
from integrator import integrate_velocity, integrate_position
from input_parser import read_config_toml, parse_config_toml
from logger import Logger


def CONFIGURE_RUNTIME(comm):
    ap = ArgumentParser()
    ap.add_argument("--verbose", default=False, action='store_true',
                    help="Increase logging verbosity")
    ap.add_argument("--profile", default=False, action='store_true',
                    help="Profile program execution with cProfile")
    ap.add_argument("--disable-mpio", default=False, action='store_true',
                    help=("Avoid using h5py-mpi, potentially decreasing IO "
                          "performance"))
    ap.add_argument("--destdir", default=".",
                    help="Write output to specified directory")
    ap.add_argument("--seed", default=None,
                    help="Set the numpy random generator seed for every rank")
    ap.add_argument("--logfile", default=None,
                    help="Redirect event logging to specified file")
    ap.add_argument("config",
                    help="Config .py or .toml input configuration script")
    ap.add_argument("input", help="input.hdf5")
    args = ap.parse_args()

    if comm.rank == 0:
        os.makedirs(args.destdir, exist_ok=True)
    comm.barrier()

    if args.seed is not None:
        np.random.seed(args.seed)
    else:
        np.random.seed()

    # Setup logger
    Logger.setup(default_level=logging.INFO,
                 log_file=args.logfile,
                 log_to_stdout=args.verbose)

    if args.profile:
        prof_file_name = 'cpu.txt-%05d-of-%05d' % (comm.rank, comm.size)
        output_file = open(os.path.join(args.destdir, prof_file_name), 'w')
        pr = cProfile.Profile()

        def profile_atexit():
            pr.disable()
            # Dump results:
            # - for binary dump
            prof_file_bin = 'cpu.prof-%05d-of-%05d' % (comm.rank, comm.size)
            pr.dump_stats(os.path.join(args.destdir, prof_file_bin))
            stats = pstats.Stats(pr, stream=output_file)
            stats.sort_stats('time').print_stats()
            output_file.close()

        # TODO: if we have a main function then we can properly do set up and
        # teardown without using atexit.
        atexit.register(profile_atexit)

        pr.enable()

    try:
        Logger.rank0.log(
            logging.INFO,
            f'Attempting to parse config file {args.config} as .toml'
        )
        toml_config = read_config_toml(args.config)
        config = parse_config_toml(toml_config, file_path=args.config)
        Logger.rank0.log(
            logging.INFO,
            f'Successfully parsed {args.config} as .toml file'
        )
        Logger.rank0.log(logging.INFO, str(config))
    except ValueError as ve:
        try:
            Logger.rank0.log(
                logging.INFO,
                (f'Attempt to parse {args.config} as .toml failed, trying '
                 'to parse as python file')
            )
            CONF = {}
            exec(open(args.config).read(), CONF)
            CONF = {
                k: v for k, v in CONF.items() if (not k.startswith('_') and
                                                  not isinstance(v, moduleobj))
            }

            Logger.rank0.log(
                logging.INFO,
                f'Successfully parsed {args.config} as .py file'
            )
            for key, value in sorted(CONF.items()):
                Logger.rank0.log(logging.INFO, f"{key} = {value}")
        except NameError as ne:
            Logger.rank0.log(
                logging.ERROR,
                (f"Attempt to parse {args.config} as .py failed"
                 f", ")
            )
            raise ValueError(
                f"Unable to parse configuration file {args.config}" +
                "\n\ntoml parse traceback:" +
                repr(ve) +
                "\n\npython parse traceback:" +
                repr(ne)
            )
    return args, CONF


# INITIALIZE MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

args, CONF = CONFIGURE_RUNTIME(comm)

# Set simulation input data


# Initialization of simulation variables
kb=2.479/298
if 'T0' in CONF:
    CONF['kbT0']=kb*CONF['T0']
    E_kin0 = kb*3*CONF['Np']*CONF['T0']/2.


if 'RESPA' in CONF or 'respa' in CONF:
    respa_inner = CONF['RESPA'] if 'RESPA' in CONF else CONF['respa']
else:
    respa_inner = 1
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


driver = 'mpio' if not args.disable_mpio else None
with h5py.File(args.input, 'r', driver=driver, comm=MPI.COMM_WORLD) as in_file:
    rank_range, molecules_flag = distribute_input(
        in_file, rank, size, CONF['Np']
    )
    indices = in_file['indices'][rank_range]
    r = in_file['coordinates'][-1, rank_range, :]
    vel = in_file['velocities'][-1, rank_range, :]
    types = in_file['types'][rank_range]
    names = in_file['names'][rank_range]
    if molecules_flag:
        molecules = in_file['molecules'][rank_range]
        bonds = in_file['bonds'][rank_range]

f = np.zeros((len(r), 3))
f_bonds = np.zeros((len(r), 3))
f_angles = np.zeros((len(r), 3))
f_old = np.copy(f)

bond_energy = 0.0
angle_energy = 0.0
# print(f'{rank}: {Counter(names)}\n{molecules}\n{indices}\n{bonds}\n')


#Particle-Mesh initialization
pm   = pmesh.ParticleMesh((CONF['Nv'],CONF['Nv'],CONF['Nv']),BoxSize=CONF['L'], dtype='f8',comm=comm)

Logger.rank0.log(logging.INFO, "ProcMesh = %s", str(pm.np))

# INTILIZE PMESH ARRAYS

phi_t=[]
force_ds=[]
v_pot=[]

for t in range(CONF['ntypes']):
    phi_t.append(pm.create('real'))
    v_pot.append(pm.create('real'))
    force_ds.append([pm.create('real') for d in range(3)])

# Output files
if args.disable_mpio:
    # FIXME(rainwoodman): This does not crash, but
    # we shall not create one file per rank with most ranges in the file untouched.
    # A Proper fix is the ranks shall open sim.hdf5 in round-robin; however
    # the dset_xxx aliases are getting in the way.
    f_hd5 = h5py.File(os.path.join(args.destdir, 'sim.hdf5-%06d-of-%06d' % (comm.rank, comm.size)), 'w')
else:
    f_hd5 = h5py.File(os.path.join(args.destdir, 'sim.hdf5'), 'w', driver='mpio', comm=comm)

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

# FIXME: this can be inefficient if p_mpi_range is discontiguous (depends on hdf-mpi impl detail)
dset_names[rank_range]=names
dset_types[rank_range]=types
dset_indices[rank_range]=indices
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
    Logger.rank0.log(
        logging.INFO,
        "DOMAIN_DECOMP: Total number of particles to be exchanged = %d",
        np.sum(layout.get_exchange_cost())
    )

    return layout.exchange(r, vel, f, indices, f_old, types)

# First step
if "domain_decomp" in CONF:
    r, vel, f, indices, f_old, types = DOMAIN_DECOMP(r, vel, f, indices, f_old, types)

layouts  = [pm.decompose(r[types==t]) for t in range(CONF['ntypes'])]

UPDATE_FIELD(layouts,True)
f_field = COMP_FORCE(layouts, r, force_ds)
if molecules_flag:
    bonds_2, bonds_3 = prepare_bonds(molecules, names, bonds, indices, CONF)
    eb = compute_bond_forces(f_bonds, r, bonds_2, CONF['L'], MPI.COMM_WORLD)
    ea = compute_angle_forces(f_angles, r, bonds_3, CONF['L'], MPI.COMM_WORLD)
    bond_energy = MPI.COMM_WORLD.allreduce(eb, MPI.SUM)
    angle_energy = MPI.COMM_WORLD.allreduce(ea, MPI.SUM)

    f = f_field + f_bonds + f_angles
else:
    f = f_field

for step in range(CONF['NSTEPS']):
    Logger.rank0.log(logging.INFO, "=> step = %d", step)

    if CONF['nprint']>0:
        frame=step//CONF['nprint']

        if(np.mod(step,CONF['nprint'])==0):
            E_hpf, E_kin,W = COMPUTE_ENERGY()
            T     =   2*E_kin/(kb*3*CONF['Np'])
            mom=pm.comm.allreduce(np.sum(vel,axis=0))

    f_old = np.copy(f)

    # Initial rRESPA velocity step
    vel = integrate_velocity(vel, f_field / CONF['mass'], CONF['dt'])

    # Inner rRESPA steps
    for inner in range(respa_inner):
        vel = integrate_velocity(vel, (f_bonds + f_angles) / CONF['mass'],
                                 CONF['dt'] / respa_inner)
        r = integrate_position(r, vel, CONF['dt'] / respa_inner)
        r = np.mod(r, CONF['L'][None, :])

        # Update fast forces
        if molecules_flag:
            eb = compute_bond_forces(f_bonds, r, bonds_2, CONF['L'],
                                     MPI.COMM_WORLD)
            ea = compute_angle_forces(f_angles, r, bonds_3, CONF['L'],
                                      MPI.COMM_WORLD)
        vel = integrate_velocity(vel, (f_bonds + f_angles) / CONF['mass'],
                                 CONF['dt'] / respa_inner)

    # Update slow forces
    f_field = COMP_FORCE(layouts, r, force_ds)

    # Second rRESPA velocity step
    vel = integrate_velocity(vel, f_field / CONF['mass'], CONF['dt'])

    # Only compute and keep the molecular bond energy from the last rRESPA
    # inner step
    if molecules_flag:
        bond_energy = MPI.COMM_WORLD.allreduce(eb, MPI.SUM)
        angle_energy = MPI.COMM_WORLD.allreduce(ea, MPI.SUM)

    if "domain_decomp" in CONF:
        r, vel, f, indices, f_old, types = DOMAIN_DECOMP(r, vel, f, indices, f_old, types)
        if molecules_flag:
            bonds_2, bonds_3 = prepare_bonds(molecules, names, bonds, indices,
                                             CONF)

    # Particles are kept on mpi-task
    layouts  = [pm.decompose(r[types==t]) for t in range(CONF['ntypes'])]
    for t in range(CONF['ntypes']):
        Logger.rank0.log(
            logging.INFO,
            "GHOSTS: Total number of particles (%d) to be exchanged = %d",
            t,
            np.sum(layouts[t].get_exchange_cost())
        )

    if(np.mod(step+1,CONF['quasi'])==0):
        UPDATE_FIELD(layouts, np.mod(step+1,CONF['quasi'])==0)

    # Thermostat
    if('T0' in CONF):
        vel = VEL_RESCALE(vel,CONF['tau'])

    # Print trajectory
    if CONF['nprint']>0:
        if(np.mod(step,CONF['nprint'])==0):
            STORE_DATA(step, frame)
            if rank == 0:
                print(f"Step: {step:5} | {100.0 * step / float(CONF['NSTEPS']):5.2f} % |", end="")
                print(f"t={step*CONF['dt']:5.4f} f_bonds:{np.max(f_bonds):10.5f} ", end="")
                print(f"f_angle:{np.max(f_angles):10.5f} ", end="")
                print(f"f_field:{np.max(f_field):10.5f} ", end="")
                print(f" E: {W+E_kin+bond_energy:10.5f} Ek:", end="")
                print(f"{E_kin:10.5f} W: {W:10.5f} Eb: {bond_energy:10.5f}", end="")
                print(f" Ea: {angle_energy:10.5f} mom: {mom[0]:10.5f} {mom[1]:10.5f} {mom[2]:10.5f}")

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
