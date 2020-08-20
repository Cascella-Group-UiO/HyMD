import h5py
import sys
import numpy as np
import json

CONF = {}
exec(open(sys.argv[1]).read(), CONF)
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
#CONF['n_frames'] = CONF['NSTEPS']//CONF['nprint']

if 'phi0' not in CONF:
    CONF['phi0']=CONF['Np']/CONF['V']

f_hd5      = h5py.File('input.hdf5', 'w')

dset_pos      = f_hd5.create_dataset("coordinates", (1,CONF['Np'],3), dtype="Float32")
dset_vel      = f_hd5.create_dataset("velocities",  (1,CONF['Np'],3), dtype="Float32")
dset_types    = f_hd5.create_dataset("types",       (CONF['Np'],), dtype="i")
dset_indices = f_hd5.create_dataset("indices",    (CONF['Np'],), dtype="i")

dt =  h5py.string_dtype(encoding='ascii')
dset_names = f_hd5.create_dataset("names",  (CONF['Np'],), dtype="S10")
dset_mass = f_hd5.create_dataset("mass",  (CONF['Np'],), dtype="Float32")

dset_pos.attrs['units']="nanometers"
dset_pos.attrs['units']="nanometers/picoseconds"

indices=[]
parts=1000
n=int(CONF['Np']/parts)
#print(n,CONF['Np'])
for i in range(parts-1):
    indices.append(list(range(i*n,(i+1)*n)))

if not(n*i==CONF['Np']):
    indices.append(list(range((i+1)*n,CONF['Np'])))


def cube(r):
    radius=np.max(np.abs(r-CONF['L'][None,:]*0.5),axis=1)
    ind=np.argsort(radius)[::-1]
    r=r.copy()[ind,:]

    return r


def GEN_START_UNIFORM():

    n=int(np.ceil(CONF['Np']**(1./3.)))
    print(n)
    l=CONF['L'][0]/n
    x=np.linspace(0.5*l,(n-0.5)*l,n)

    j=0
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                if(j<CONF['Np']):
                    print(x[ix])
                    dset_pos[0,j,0]=x[ix]
                    dset_pos[0,j,1]=x[iy]
                    dset_pos[0,j,2]=x[iz]
    if(j<CONF['Np']):
        dset_pos[0,j:,:]=CONF['L']*np.random.random((CONF['Np']-j,3))
    return r

def GEN_RANDOM(dset):

    # nrange=[0]
    # for i in range(CONF['Np']//1000-1):
    #     nrange.append(i*1000)
    # nrange.append(-nrange[-1]+CONF['Np'])
    # for i in range(len(nrange)-1):
    #     if(print(i,-nrange[i]+nrange[i+1])
    #     dset[0,i=CONF['L']*np.random.random((nrange[i+1]-nrange[i],3))

#    for i in range(CONF['Np']):
    for ind in indices:
        dset[0,ind,:]=CONF['L']*np.random.random((len(ind),3))

def GEN_START_VEL(dset):
    #NORMAL DISTRIBUTED PARTICLES FIRST FRAME
    std  = np.sqrt(CONF['kbT_start']/CONF['mass'])
    dset[0,:,:] = np.random.normal(loc=0, scale=std,size=(CONF['Np'],3))

    dset[0,:,:] = dset[0,:,:]-np.mean(dset[0,:,:], axis=0)
    fac= np.sqrt((3*CONF['Np']*CONF['kbT_start']/2.)/(0.5*CONF['mass']*np.sum(dset[0,:,:]**2)))

    dset[0,:,:]=dset[0,:,:]*fac




if 'uniform_start' in CONF:
    if CONF['uniform_start']==True:
        GEN_START_UNIFORM()
else:
    GEN_RANDOM(dset_pos)

if 'T_start' in CONF:
    GEN_START_VEL(dset_vel)

#for i in range(CONF['Np']):


for ind in indices:
    dset_types[ind]=0
    dset_names[ind]=np.string_("A")
    dset_mass[ind]=CONF['mass']
    dset_indices[ind]=np.array(ind)


if 'chi' in CONF and 'NB' in CONF:
   #" for i in range(CONF['Np']-CONF['NB'],CONF['Np']):
    #indices2=np.array(range(CONF['Np']-CONF['NB']))
    for ind in np.split(indices,100):
        ind2=ind[ind>CONF['Np']-CONF['NB']]
        if(len(ind2)>0):
            dset_types[ind2]=1
            dset_names[ind2]=np.string_("B")

f_hd5.close()


# f_hd5      = h5py.File('input.hdf5', 'r')
# print(f_hd5.get('coordinates')[:])
# print(f_hd5.get('velocities')[:])
# print(f_hd5.get('types')[:])
# print(f_hd5.get('names')[:])
# print(f_hd5.get('mass')[:])
# print(f_hd5.get('indices')[:])
