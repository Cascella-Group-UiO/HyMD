import numpy as np
from mpi4py import MPI
from scipy.spatial.transform import Rotation as R
import sys
import pmesh.pm as pmesh # Particle mesh Routine

CONF = {}
exec(open(sys.argv[1]).read(), CONF)

# Same seed for all simulations
np.random.seed(0)

#Initialization of simulation variables
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
r=np.zeros((CONF['Np'],3))
vel=np.zeros((CONF['Np'],3))

#Intialize MPI
comm = MPI.COMM_WORLD
 
    
# SAVE FIRST STEP
np.savetxt('start.dat',np.hstack((r,vel)))


# Particle force array
f=np.zeros((CONF['Np'],3))
f_old=np.copy(f)
if "test_quasi" in CONF: 
    f_test=np.copy(f)
    f_zero=np.copy(f)
    r_zero=np.copy(r)

#Particle-Mesh initialization
pm   = pmesh.ParticleMesh((CONF['Nv'],CONF['Nv'],CONF['Nv']),BoxSize=CONF['L'], dtype='f8',comm=comm)
density = pm.create(mode='real')
indicies=[]
     
if 'chi' in CONF and 'NB' in CONF:
    indicies.append(np.arange(0,CONF['Np']-CONF['NB']))#A
    indicies.append(np.arange(CONF['Np']-CONF['NB'],CONF['Np']))#B

    types = 2
    layout  = [pm.decompose(r[indicies[0]]),pm.decompose(r[indicies[1]])]  
    names=['A']*(CONF['Np']-CONF['NB'])+['B']*CONF['NB']
    
else:
    names=['A']*CONF['Np']
    indicies.append(np.arange(CONF['Np']))
    types=1
    layout  = [pm.decompose(r[indicies[0]])]


# INTILIZE PMESH ARRAYS
phi=[]
phi_fft=[]
phi_fft_tt=[]
phi_t=[]
force_ds=[]
v_pot=[]
v_pot_fft=[]
if 'test_quasi' in CONF:
    force_test=[]
for t in range(types):
    phi.append(pm.create('real'))
    phi_t.append(pm.create('real'))
    phi_fft.append(pm.create('complex'))
    phi_fft_tt.append(pm.create('complex'))
    v_pot.append(pm.create('real'))
    v_pot_fft.append(pm.create('complex'))

    force_ds.append([pm.create('real') for d in range(3)])
    if 'test_quasi' in CONF:
        force_test.append([pm.create('real') for d in range(3)])

# Output files
fp_trj = open('trj.gro','w')
fp_E   = open('E.dat','w')

#FUNCTION DEFINITIONS

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

def potential_transfer_function(k, v):
    return (v * np.exp(-0.5*CONF['sigma']**2*k.normp(p=2))**2)
def phi_transfer_function(k, v):
    return v * np.exp(-0.5*CONF['sigma']**2*k.normp(p=2))

def INTEGERATE_POS(x, vel, a):
# Velocity Verlet integration I
    return x + vel*CONF['dt'] + 0.5*a*CONF['dt']**2

def INTEGRATE_VEL(vel, a, a_old):
# Velocity Verlet integration II
    return vel + 0.5*(a+a_old)*CONF['dt']

def VEL_RESCALE(vel, tau):
    #https://doi.org/10.1063/1.2408420

    # INITIAL KINETIC ENERGY
    E_kin  = 0.5*CONF['mass']*np.sum(vel**2)

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


def WRITE_TRJ_GRO(fp, x, vel,t):    
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
            f[indicies[t], d] = force_ds[t][d].readout(r[indicies[t]], layout=layout[t])

def UPDATE_FIELD(comp_v_pot=False):

    for k in range(types):      
        # Distribute paticles
        phi[k].value[:]=0.0
        phi[k]=pm.paint(r[indicies[k]], layout=layout[k])
        phi[k] = phi[k]/CONF['dV']

        phi_fft[k] = phi[k].r2c(out=Ellipsis)
        phi_fft_tt[k]=phi_fft[k].copy().apply(potential_transfer_function, out=Ellipsis)

        
    if types==2:
        v_pot_fft[0]=((1/(CONF['kappa']*CONF['phi0'])*(phi_fft_tt[0]+phi_fft_tt[1]) + CONF['chi']/CONF['phi0']*phi_fft_tt[1]))
        v_pot_fft[1]=((1/(CONF['kappa']*CONF['phi0'])*(phi_fft_tt[0]+phi_fft_tt[1]) + CONF['chi']/CONF['phi0']*phi_fft_tt[0]))
    else:
        v_pot_fft[0]=(1/(CONF['kappa']*CONF['phi0'])*phi_fft_tt[0])
        
    for t in range(types):
        force_d=[]
        for d in range(3):    
            def force_transfer_function(k, v, d=d):
                return -k[d] * 1j * v 
            force_ds[t][d]=(v_pot_fft[t].apply(force_transfer_function).c2r(out=Ellipsis))
        if(comp_v_pot):
            v_pot[t]=v_pot_fft[t].c2r(out=Ellipsis)-1./CONF['kappa']
            phi_t[t]=phi_fft[t].c2r(out=Ellipsis)

def COMPUTE_ENERGY():
    E_hpf=0
    for i in range(types):
        E_hpf = pm.comm.allreduce(0.5*np.sum(v_pot[0].readout(r[indicies[i]],layout=layout[i]))) 
    E_kin = pm.comm.allreduce(0.5*CONF['mass']*np.sum(vel**2))
 
    
    
    for k in range(types):
        phi_t[k]=phi[k].r2c(out=Ellipsis).apply(phi_transfer_function, out=Ellipsis).c2r(out=Ellipsis)
        
    if types==2:
        W=CONF['dV']/CONF['phi0']*np.sum(0.5/CONF['kappa']*(phi_t[0] + phi_t[1]-CONF['phi0'])**2 + CONF['chi']*phi_t[0]*phi_t[1])
    else:
        W = 0.5*CONF['dV']/CONF['phi0']*np.sum(1./CONF['kappa']*(phi_t[0]-CONF['phi0'])**2)
    return E_hpf,E_kin,W
    
def ANDERSEN(vel):
    
    std  = np.sqrt(CONF['kbT0']/CONF['mass']) 
    change = np.random.random((len(vel))) < taut*CONF['dt']

    if(sum(change)>0):
        vel[change,0]=np.random.normal(loc=0, scale=std, size=sum(change))
        vel[change,1]=np.random.normal(loc=0, scale=std, size=sum(change))
        vel[change,2]=np.random.normal(loc=0, scale=std, size=sum(change))
    
    return vel

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


def TEST_QUASI(step,v_test,force_test):
    E_hpf_zero=0.
    E_hpf_test=0.0
    for t in range(types):
        E_hpf_test += pm.comm.allreduce(0.5*np.sum(v_test[t].readout(r[indicies[t]],layout=layout[t])))
        E_hpf_zero += pm.comm.allreduce(0.5*np.sum(v_test[t].readout(r_0[indicies[t]],layout=layout[t])))
        for d in range(3):
            f_test[indicies[t],d] = force_test[t][d].readout(r[indicies[t]], layout=layout[t])
            f_zero[indicies[t],d] = force_test[t][d].readout(r_0[indicies[t]], layout=layout[t])

    f_mag=np.sqrt(np.mean(np.sum((f)**2,axis=1)))
    f_test_std=np.sqrt(np.mean(np.sum((f-f_test)**2,axis=1)))
    f_zero_std=np.sqrt(np.mean(np.sum((f-f_zero)**2,axis=1)))
                  
    fp_test.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(step*CONF['dt'],E_hpf,E_hpf_test,E_hpf_zero,f_mag, f_test_std,f_zero_std))            

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
# For first step
UPDATE_FIELD(True)
COMP_FORCE(f,r,force_ds)

        
if "test_quasi" in CONF:    
    force_tests=force_ds.copy()
    v_test=v_pot.copy()
    r_0=r.copy()

for step in range(CONF['NSTEPS']):

     
    if(np.mod(step,CONF['nprint'])==0):      
        E_hpf, E_kin,W = COMPUTE_ENERGY()        
        T     =   2*E_kin/(kb*3*CONF['Np'])
        mom=np.sum(vel,axis=0)
        if "test_quasi" in CONF:        
            TEST_QUASI(step,v_test,force_tests)

    f_old = np.copy(f)

    #Integrate positions
    r     = INTEGERATE_POS(r, vel, f/CONF['mass'])

    #PERIODIC BC
    r     = np.mod(r, CONF['L'][None,:])

    if(np.mod(step+1,CONF['quasi'])==0):
        UPDATE_FIELD(np.mod(step+1,CONF['quasi'])==0)
         
    if "test_quasi" in CONF:
        if(np.mod(step+1,test_quasi)==0):
            force_tests=force_ds.copy()
            v_test=v_pot.copy()
            r_0=r.copy()

    COMP_FORCE(f,r,force_ds)
    
    # Integrate velocity
    vel = INTEGRATE_VEL(vel, f/CONF['mass'], f_old/CONF['mass'])

    # Thermostat
    if('T0' in CONF):
        #vel = ANDERSEN(vel)
        vel = VEL_RESCALE(vel,tau)
    # Print trajectory
    if(np.mod(step,CONF['nprint'])==0):
        
        fp_trj=WRITE_TRJ_GRO(fp_trj, r, vel,CONF['dt']*step)
        fp_E.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(step*CONF['dt'],W+E_kin,W,E_kin,T,mom[0],mom[1],mom[2]))

        fp_E.flush()


UPDATE_FIELD(True)
                              
E_hpf, E_kin, W = COMPUTE_ENERGY()
T     =  2*E_kin/(kb*3*CONF['Np'])
mom=np.sum(vel,axis=0)

fp_E.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(CONF['NSTEPS']*CONF['dt'],W+E_kin,W,E_kin,T,mom[0],mom[1],mom[2]))
 
# Write last frame
fp_trj=WRITE_TRJ_GRO(fp_trj, r, vel,CONF['dt']*CONF['NSTEPS'])

np.savetxt('final.dat',np.hstack((r,vel,f_old)))
