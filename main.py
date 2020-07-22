import numpy as np
from mpi4py import MPI
from scipy.spatial.transform import Rotation as R
import sys
import pmesh.pm as pmesh # Particle mesh Routine
import operator
import functools
import time

# INITIALIZE MPIW=
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
NAME=['A','B']
def scatter_from_root(a):
    if rank==0:
        a=np.split(a,size)
    else:
        a=None

    a = comm.scatter(a, root=0)
    return a

def gather_to_root(a):
    a=comm.gather(a, root=0)
    if not(rank==0):
        a=None

    return a

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

if 'T_start' in CONF:
    CONF['kbT_start']=kb*CONF['T_start']

Ncells = (CONF['Nv']**3)
CONF['L']      = np.array(CONF['L'])
CONF['V']      = CONF['L'][0]*CONF['L'][1]*CONF['L'][2]
CONF['dV']     = CONF['V']/(CONF['Nv']**3)

if 'phi0' not in CONF:
    CONF['phi0']=CONF['Np']/CONF['V']
    

#Initialize particles
if rank==0:
    r=np.zeros((CONF['Np'],3))
    vel=np.zeros((CONF['Np'],3))
    f=np.zeros((CONF['Np'],3))
    f_old=np.copy(f)

else:
    r=None
    vel=None
    f=None
    f_old=None


 
    
# SAVE FIRST STEP
if rank==0:
    np.savetxt('start.dat',np.hstack((r,vel)))

#Particle-Mesh initialization
pm   = pmesh.ParticleMesh((CONF['Nv'],CONF['Nv'],CONF['Nv']),BoxSize=CONF['L'], dtype='f8',comm=comm)

indicies=[]

if 'chi' in CONF and 'NB' in CONF:
    types = 2
else:
    types = 1

if rank==0:     
    if 'chi' in CONF and 'NB' in CONF:
        indicies.append(np.arange(0,CONF['Np']-CONF['NB']))#A
        indicies.append(np.arange(CONF['Np']-CONF['NB'],CONF['Np']))#B

        indicies2=np.zeros(CONF['Np'])
        indicies2[CONF['Np']-CONF['NB']:]=1
        

        
#        layout  = [pm.decompose(r[indicies[0]]),pm.decompose(r[indicies[1]])]  
        names=['A']*(CONF['Np']-CONF['NB'])+['B']*CONF['NB']
    
    else:
        names=['A']*CONF['Np']
        indicies.append(np.arange(CONF['Np']))
        indicies2=np.zeros(CONF['Np'],dtype=int)
   
        names=['A']*(CONF['Np'])
 
      # layout  = [pm.decompose(r[indicies[0]])]
else:
    indicies2=None
    names=None

# INTILIZE PMESH ARRAYS
phi=[]
phi_t=[]
force_ds=[]
v_pot=[]

for t in range(types):
    phi.append(pm.create('real'))
    phi_t.append(pm.create('real'))
    v_pot.append(pm.create('real'))
    force_ds.append([pm.create('real') for d in range(3)])

# Output files
if rank==0:
    fp_trj = open('trj.gro','w')
    fp_E   = open('E.dat','w')

# #FUNCTION DEFINITIONS

def GEN_START_VEL(vel):
    #NORMAL DISTRIBUTED PARTICLES FIRST FRAME
    std  = np.sqrt(CONF['kbT_start']/CONF['mass'])
    vel = np.random.normal(loc=0, scale=std,size=(CONF['Np'],3))
    
    vel = vel-np.mean(vel, axis=0)
    fac= np.sqrt((3*CONF['Np']*CONF['kbT_start']/2.)/(0.5*CONF['mass']*np.sum(vel**2)))
    
    return fac*vel

 
def GEN_START_UNIFORM(r):
     
    n=int(np.ceil(CONF['Np']**(1./3.)))
    l=CONF['L'][0]/n
    x=np.linspace(0.5*l,(n-0.5)*l,n)
    
    j=0
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                if(j<CONF['Np']):
                    r[j,0]=x[ix]
                    r[j,1]=x[iy]
                    r[j,2]=x[iz]
                
                j+=1
    if(j<CONF['Np']):
        r[j:]=CONF['L']*np.random.random((CONF['Np']-j,3))
    return r


def INTEGERATE_POS(x, vel, a):
# Velocity Verlet integration I
    return x + vel*CONF['dt'] + 0.5*a*CONF['dt']**2

def INTEGRATE_VEL(vel, a, a_old):
# Velocity Verlet integration II
    return vel + 0.5*(a+a_old)*CONF['dt']

def VEL_RESCALE(vel, tau):
    #https://doi.org/10.1063/1.2408420

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


def sphere(r):
    radius=np.linalg.norm(r-CONF['L'][None,:]*0.5,axis=1)
    ind=np.argsort(radius)[::-1]
    r=r.copy()[ind,:]
    
    return r

def cube(r):
    radius=np.max(np.abs(r-CONF['L'][None,:]*0.5),axis=1)
    ind=np.argsort(radius)[::-1]
    r=r.copy()[ind,:]
    
    return r


def WRITE_TRJ_GRO(fp, x, vel,names,t):    
    fp.write('MD of %d mols, t=%.3f\n'%(CONF['Np'],t))
    fp.write('%-10d\n'%(CONF['Np']))
    for i in range(len(x)):
        fp.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n"%(i//10+1,names[i],names[i],i+1,x[i,0],x[i,1],x[i,2],vel[i,0],vel[i,1],vel[i,2]))
    fp.write("%-5.5f\t%5.5f\t%5.5f\n"%(CONF['L'][0],CONF['L'][1],CONF['L'][2]))
    fp.flush()
    return fp


def WRITE_TRJ(fp, x, vel,f=None):    
    fp.write('%d\n'%(len(x)))
    fp.write('\n')
    
    if f==None:
        for i in range(len(x)):
            fp.write("%s %f %f %f %f %f %f\n"%(names[i],x[i,0],x[i,1],x[i,2],vel[i,0],vel[i,1],vel[i,2]))
    else:
        for i in range(len(x)):
            fp.write("%s %f %f %f %f %f %f\n"%(names[i],x[i,0],x[i,1],x[i,2],vel[i,0],vel[i,1],vel[i,2],f[i,0],f[i,1],f[i,2]))
    return fp

def COMP_FORCE(f, r, force_ds):
    for t in range(types):
        for d in range(3):
            f[indicies2==t, d] = force_ds[t][d].readout(r[indicies2==t], layout=layout[t])

def COMP_PRESSURE():
    # COMPUTES HPF PRESSURE FOR EACH MPI TASK
    P=[]
    p_val=[]
    for d in range(3):
        
        p = CONF['w'](phi_t)
        for t in range(CONF['types']):
            p += -v_pot[t]*phi_t[t] + (v_pot[t].r2c(out=Ellipsis).apply(CONF['kdHdk'])[d]).c2r(out=Ellipsis)


        P.append(p*CONF['dV']/CONF['V'])
        p_val.append(p.csum())
    return np.array(p_val)
      
def UPDATE_FIELD(layout,comp_v_pot=False):

    # Filtered density
    for t in range(types):
        p = pm.paint(r[indicies2==t], layout=layout[t])
        p = p/CONF['dV']
        phi_t[t] = p.r2c(out=Ellipsis).apply(CONF['H'], out=Ellipsis).c2r(out=Ellipsis)

    # External potential
    for t in range(types):
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
    #print(W)
    W = W.csum()
    #print(W) 
    return E_hpf,E_kin,W
    

def PRESET_VEL(vel,T):
    std  = np.sqrt(CONF['kbT0']/CONF['mass'])
    vel[:,0]=np.random.normal(loc=0, scale=std, size=CONF['Np'])
    vel[change,1]=np.random.normal(loc=0, scale=std, size=CONF['Np'])
    vel[change,2]=np.random.normal(loc=0, scale=std, size=CONF['Np'])
    return vel

def REMOVE_CM_VEL(vel,T=None):
    if T==None:
        E_KIN_1 = 0.5*CONF['mass']*np.sum(vel**2)
    else:
        E_KIN_1 = kb*3*CONF['Np']*T/2.
 
    vcm     = -np.mean(vel,axis=0)
    E_KIN_2 = 0.5*CONF['mass']*np.sum((vel-vcm)**2)

    return (vel-vcm)*np.sqrt(E_KIN_1/E_KIN_2)

 

if rank==0:
    #Degrees of freedom
    if('read_start' in CONF):
        dat = np.loadtxt(read_start)
        r   = dat[:,:3]
        vel = dat[:,3:6]

    else:
        if 'uniform_start' in CONF:
            if uniform_start==True:
                r = GEN_START_UNIFORM(r)
            else:
                r   = np.random.random((CONF['Np'],3))*CONF['L']
        else:
            r   = np.random.random((CONF['Np'],3))*CONF['L']
        

    if 'T_start' in CONF:
        vel = np.zeros((CONF['Np'],3))
        vel = GEN_START_VEL(vel) 

    if 'gen_sphere' in CONF:
        if CONF['gen_sphere']:
            r=sphere(r)

    if 'gen_cube' in CONF:
        if CONF[gen_cube]:
            r=cube(r)
    
    if 'rm_cm_vel' in CONF:
        if CONF['rm_cm_vel']:
            vel=REMOVE_CM_VEL(vel)    

# Split data and communicate from root to the other mpi-tasks
r     = scatter_from_root(r)
vel   = scatter_from_root(vel)
f     = scatter_from_root(f)
f_old = scatter_from_root(f_old)
indicies2=scatter_from_root(indicies2)
names=[NAME[i] for i in indicies2]




if rank==0:
    start_t = time.time()


# First step
layout  = [pm.decompose(r[indicies2==t]) for t in range(types)]
UPDATE_FIELD(layout,True)
COMP_FORCE(f,r,force_ds)

        
for step in range(CONF['NSTEPS']):

    if(np.mod(step,CONF['nprint'])==0):      
        E_hpf, E_kin,W = COMPUTE_ENERGY()        
        T     =   2*E_kin/(kb*3*CONF['Np'])
        mom=pm.comm.allreduce(np.sum(vel,axis=0))

    f_old = np.copy(f)

    #Integrate positions
    r     = INTEGERATE_POS(r, vel, f/CONF['mass'])

    #PERIODIC BC
    r     = np.mod(r, CONF['L'][None,:])
    
    layout  = [pm.decompose(r[indicies2==t]) for t in range(types)]

    if(np.mod(step+1,CONF['quasi'])==0):
        UPDATE_FIELD(layout, np.mod(step+1,CONF['quasi'])==0)
         

    COMP_FORCE(f,r,force_ds)
    
    # Integrate velocity
    vel = INTEGRATE_VEL(vel, f/CONF['mass'], f_old/CONF['mass'])

    # Thermostat
    if('T0' in CONF):
        vel = VEL_RESCALE(vel,tau)

#   Print trajectory
    if(np.mod(step,CONF['nprint'])==0):
        r_out= comm.gather(r, root=0)
        vel_out=comm.gather(vel, root=0)
        names_out=comm.gather(names, root=0)
        
        if(rank==0):
            names_out=functools.reduce(operator.add,names_out)
            fp_trj=WRITE_TRJ_GRO(fp_trj, np.concatenate(r_out,axis=0), np.concatenate(vel_out,axis=0),names_out,CONF['dt']*step)
            fp_E.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(step*CONF['dt'],W+E_kin,W,E_kin,T,mom[0],mom[1],mom[2]))
            fp_E.flush()

# End simulation
if rank==0:
    print('Simulation time elapsed:', time.time()-start_t)

        
UPDATE_FIELD(layout,True)
                               
E_hpf, E_kin, W = COMPUTE_ENERGY()
T     =  2*E_kin/(kb*3*CONF['Np'])
mom=pm.comm.allreduce(np.sum(vel,axis=0))
if rank==0:
    fp_E.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(CONF['NSTEPS']*CONF['dt'],W+E_kin,W,E_kin,T,mom[0],mom[1],mom[2]))
 
    # Write last frame
r_out= comm.gather(r, root=0)
vel_out=comm.gather(vel, root=0)
names_out=comm.gather(names, root=0)
if rank==0:       
    names_out=functools.reduce(operator.add,names_out)
    fp_trj=WRITE_TRJ_GRO(fp_trj, np.concatenate(r_out,axis=0), np.concatenate(vel_out,axis=0),names_out,CONF['dt']**CONF['NSTEPS'])
    np.savetxt('final.dat',np.hstack((r,vel,f_old)))
 
