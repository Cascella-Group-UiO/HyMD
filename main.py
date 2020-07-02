import numpy as np
from mpi4py import MPI
from scipy.spatial.transform import Rotation as R
from CONF import *     # "myapp" case
import CONF as sim 
import pmesh.pm as pm # Particle mesh Routine



#Initialization of simulation variables
kb=2.479/298
if('T0' in globals()):
    kbT0=kb*T0
    E_kin0 = kb*3*Np*T0/2.

if 'T_start' in globals():
    kbT_start=kb*T_start

Ncells = (Nv**3)
L      = np.array(L)
V      = L[0]*L[1]*L[2]
dV     = V/(Nv**3)

if 'phi0' not in globals():
    phi0=Np/V
    

#Initialize particles
r=np.zeros((Np,3))
vel=np.zeros((Np,3))

#Intialize MPI
comm = MPI.COMM_WORLD
 
    
# SAVE FIRST STEP
np.savetxt('start.dat',np.hstack((r,vel)))


# Particle force array
f=np.zeros((Np,3))
f_old=np.copy(f)
if "test_quasi" in globals(): 
    f_test=np.copy(f)
    f_zero=np.copy(f)
    r_zero=np.copy(r)

#Particle-Mesh initialization
pmesh   = pm.ParticleMesh((Nv,Nv,Nv),BoxSize=L, dtype='f8',comm=comm)
density = pmesh.create(mode='real')
indicies=[]
     
if 'chi' in globals() and 'NB' in globals():
    indicies.append(np.arange(0,Np-NB))#A
    indicies.append(np.arange(Np-NB,Np))#B

    types = 2
    layout  = [pmesh.decompose(r[indicies[0]]),pmesh.decompose(r[indicies[1]])]  
    names=['A']*(Np-NB)+['B']*NB
    
else:
    names=['A']*Np
    indicies.append(np.arange(Np))
    types=1
    layout  = [pmesh.decompose(r[indicies[0]])]


# INTILIZE PMESH ARRAYS
phi=[]
phi_fft=[]
phi_fft_tt=[]
phi_t=[]
force_ds=[]
v_pot=[]
v_pot_fft=[]
if 'test_quasi' in globals():
    force_test=[]
for t in range(types):
    phi.append(pmesh.create('real'))
    phi_t.append(pmesh.create('real'))
    phi_fft.append(pmesh.create('complex'))
    phi_fft_tt.append(pmesh.create('complex'))
    v_pot.append(pmesh.create('real'))
    v_pot_fft.append(pmesh.create('complex'))

    force_ds.append([pmesh.create('real') for d in range(3)])
    if 'test_quasi' in globals():
        force_test.append([pmesh.create('real') for d in range(3)])

# Output files
fp_trj = open('trj.gro','w')
fp_E   = open('E.dat','w')

#FUNCTION DEFINITIONS

def GEN_START_VEL(vel):
    #NORMAL DISTRIBUTED PARTICLES FIRST FRAME
    std  = np.sqrt(kbT_start/mass)
    vel = np.random.normal(loc=0, scale=std,size=(Np,3))
    
    vel = vel-np.mean(vel, axis=0)
    fac= np.sqrt((3*Np*kbT_start/2.)/(0.5*mass*np.sum(vel**2)))
    
    return fac*vel

 
def GEN_START_UNIFORM(r):
     
    n=int(np.ceil(Np**(1./3.)))
    l=L[0]/n
    x=np.linspace(0.5*l,(n-0.5)*l,n)
    
    j=0
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                if(j<Np):
                    r[j,0]=x[ix]
                    r[j,1]=x[iy]
                    r[j,2]=x[iz]
                
                j+=1
    if(j<Np):
        r[j:]=L*np.random.random((Np-j,3))
    return r

def potential_transfer_function(k, v):
    return (v * np.exp(-0.5*sigma**2*k.normp(p=2))**2)
def phi_transfer_function(k, v):
    return v * np.exp(-0.5*sigma**2*k.normp(p=2))

def INTEGERATE_POS(x, vel, a):
# Velocity Verlet integration I
    return x + vel*dt + 0.5*a*dt**2

def INTEGRATE_VEL(vel, a, a_old):
# Velocity Verlet integration II
    return vel + 0.5*(a+a_old)*dt

def VEL_RESCALE(vel, tau):
    #https://doi.org/10.1063/1.2408420

    # INITIAL KINETIC ENERGY
    E_kin  = 0.5*mass*np.sum(vel**2)

    #BERENDSEN LIKE TERM
    d1 = (E_kin0-E_kin)*dt/tau

    # WIENER NOISE
    dW = np.sqrt(dt)*np.random.normal()

    # STOCHASTIC TERM
    d2 = 2*np.sqrt(E_kin*E_kin0/(3*Np))*dW/np.sqrt(tau)

    # TARGET KINETIC ENERGY
    E_kin_target = E_kin + d1 + d2

    #VELOCITY SCALING
    alpha = np.sqrt(E_kin_target/E_kin)

    vel=vel*alpha
    
    return vel

def sphere(r):
    radius=np.linalg.norm(r-L[None,:]*0.5,axis=1)
    ind=np.argsort(radius)[::-1]
    r=r.copy()[ind,:]
    
    return r

def cube(r):
    radius=np.max(np.abs(r-L[None,:]*0.5),axis=1)
    ind=np.argsort(radius)[::-1]
    r=r.copy()[ind,:]
    
    return r


def WRITE_TRJ_GRO(fp, x, vel,t):    
    fp.write('MD of %d mols, t=%.3f\n'%(Np,t))
    fp.write('%-10d\n'%(Np))
    for i in range(len(x)):
        fp.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n"%(i//10+1,names[i],names[i],i+1,x[i,0],x[i,1],x[i,2],vel[i,0],vel[i,1],vel[i,2]))
    fp.write("%-5.5f\t%5.5f\t%5.5f\n"%(L[0],L[1],L[2]))
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
        phi[k]=pmesh.paint(r[indicies[k]], layout=layout[k])
        phi[k] = phi[k]/dV

        phi_fft[k] = phi[k].r2c(out=Ellipsis)
        phi_fft_tt[k]=phi_fft[k].copy().apply(potential_transfer_function, out=Ellipsis)

        
    if types==2:
        v_pot_fft[0]=((1/(kappa*phi0)*(phi_fft_tt[0]+phi_fft_tt[1]) + chi/phi0*phi_fft_tt[1]))
        v_pot_fft[1]=((1/(kappa*phi0)*(phi_fft_tt[0]+phi_fft_tt[1]) + chi/phi0*phi_fft_tt[0]))
    else:
        v_pot_fft[0]=(1/(kappa*phi0)*phi_fft_tt[0])
        
    for t in range(types):
        force_d=[]
        for d in range(3):    
            def force_transfer_function(k, v, d=d):
                return -k[d] * 1j * v 
            force_ds[t][d]=(v_pot_fft[t].apply(force_transfer_function).c2r(out=Ellipsis))
        if(comp_v_pot):
            v_pot[t]=v_pot_fft[t].c2r(out=Ellipsis)-1./kappa
            phi_t[t]=phi_fft[t].c2r(out=Ellipsis)

def COMPUTE_ENERGY():
    E_hpf=0
    for i in range(types):
        E_hpf = pmesh.comm.allreduce(0.5*np.sum(v_pot[0].readout(r[indicies[i]],layout=layout[i]))) 
    E_kin = pmesh.comm.allreduce(0.5*mass*np.sum(vel**2))
 
    
    
    for k in range(types):
        phi_t[k]=phi[k].r2c(out=Ellipsis).apply(phi_transfer_function, out=Ellipsis).c2r(out=Ellipsis)
        
    if types==2:
        W=dV/phi0*np.sum(0.5/kappa*(phi_t[0] + phi_t[1]-phi0)**2 + chi*phi_t[0]*phi_t[1])
    else:
        W = 0.5*dV/phi0*np.sum(1./kappa*(phi_t[0]-phi0)**2)
    return E_hpf,E_kin,W
    
def ANDERSEN(vel):
    
    std  = np.sqrt(kbT0/mass) 
    change = np.random.random((len(vel))) < tau*dt

    if(sum(change)>0):
        vel[change,0]=np.random.normal(loc=0, scale=std, size=sum(change))
        vel[change,1]=np.random.normal(loc=0, scale=std, size=sum(change))
        vel[change,2]=np.random.normal(loc=0, scale=std, size=sum(change))
    
    return vel

def PRESET_VEL(vel,T):
    std  = np.sqrt(kbT0/mass)
    vel[:,0]=np.random.normal(loc=0, scale=std, size=Np)
    vel[change,1]=np.random.normal(loc=0, scale=std, size=Np)
    vel[change,2]=np.random.normal(loc=0, scale=std, size=Np)
    return vel

def REMOVE_CM_VEL(vel,T=None):
    if T==None:
        E_KIN_1 = 0.5*mass*np.sum(vel**2)
    else:
        E_KIN_1 = kb*3*Np*T/2.
 
    vcm     = -np.mean(vel,axis=0)
    E_KIN_2 = 0.5*mass*np.sum((vel-vcm)**2)

    return (vel-vcm)*np.sqrt(E_KIN_1/E_KIN_2)


def TEST_QUASI(step,v_test,force_test):
    E_hpf_zero=0.
    E_hpf_test=0.0
    for t in range(types):
        E_hpf_test += pmesh.comm.allreduce(0.5*np.sum(v_test[t].readout(r[indicies[t]],layout=layout[t])))
        E_hpf_zero += pmesh.comm.allreduce(0.5*np.sum(v_test[t].readout(r_0[indicies[t]],layout=layout[t])))
        for d in range(3):
            f_test[indicies[t],d] = force_test[t][d].readout(r[indicies[t]], layout=layout[t])
            f_zero[indicies[t],d] = force_test[t][d].readout(r_0[indicies[t]], layout=layout[t])

    f_mag=np.sqrt(np.mean(np.sum((f)**2,axis=1)))
    f_test_std=np.sqrt(np.mean(np.sum((f-f_test)**2,axis=1)))
    f_zero_std=np.sqrt(np.mean(np.sum((f-f_zero)**2,axis=1)))
                  
    fp_test.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(step*dt,E_hpf,E_hpf_test,E_hpf_zero,f_mag, f_test_std,f_zero_std))            

#Degrees of freedom
if('read_start' in globals()):
    dat = np.loadtxt(read_start)
    r   = dat[:,:3]
    vel = dat[:,3:6]

else:
    if 'uniform_start' in globals():
        if uniform_start==True:
            r = GEN_START_UNIFORM(r)
        else:
            r   = np.random.random((Np,3))*L
    else:
        r   = np.random.random((Np,3))*L
        

if 'T_start' in globals():
    vel = np.zeros((Np,3))
    vel = GEN_START_VEL(vel) 

if 'gen_sphere' in globals():
    if gen_sphere:
        r=sphere(r)

if 'gen_cube' in globals():
    if gen_cube:
        r=cube(r)
    
if 'rm_cm_vel' in globals():
    if rm_cm_vel:
        vel=REMOVE_CM_VEL(vel)    
# For first step
UPDATE_FIELD(True)
COMP_FORCE(f,r,force_ds)

        
if "test_quasi" in globals():    
    force_tests=force_ds.copy()
    v_test=v_pot.copy()
    r_0=r.copy()

for step in range(NSTEPS):

     
    if(np.mod(step,nprint)==0):      
        E_hpf, E_kin,W = COMPUTE_ENERGY()        
        T     =   2*E_kin/(kb*3*Np)
        mom=np.sum(vel,axis=0)
        if "test_quasi" in globals():        
            TEST_QUASI(step,v_test,force_tests)

    f_old = np.copy(f)

    #Integrate positions
    r     = INTEGERATE_POS(r, vel, f/mass)

    #PERIODIC BC
    r     = np.mod(r, L[None,:])

    if(np.mod(step+1,quasi)==0):
        UPDATE_FIELD(np.mod(step+1,quasi)==0)
         
    if "test_quasi" in globals():
        if(np.mod(step+1,test_quasi)==0):
            force_tests=force_ds.copy()
            v_test=v_pot.copy()
            r_0=r.copy()

    COMP_FORCE(f,r,force_ds)
    
    # Integrate velocity
    vel = INTEGRATE_VEL(vel, f/mass, f_old/mass)

    # Thermostat
    if('T0' in globals()):
        #vel = ANDERSEN(vel)
        vel = VEL_RESCALE(vel,tau)
    # Print trajectory
    if(np.mod(step,nprint)==0):
        
        fp_trj=WRITE_TRJ_GRO(fp_trj, r, vel,dt*step)
        fp_E.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(step*dt,W+E_kin,W,E_kin,T,mom[0],mom[1],mom[2]))

        fp_E.flush()


UPDATE_FIELD(True)
                              
E_hpf, E_kin, W = COMPUTE_ENERGY()
T     =  2*E_kin/(kb*3*Np)
mom=np.sum(vel,axis=0)

fp_E.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(NSTEPS*dt,W+E_kin,W,E_kin,T,mom[0],mom[1],mom[2]))
 
# Write last frame
fp_trj=WRITE_TRJ_GRO(fp_trj, r, vel,dt*NSTEPS)

np.savetxt('final.dat',np.hstack((r,vel,f_old)))
