"""
This routine 
- read the hdf5 file e.g. dppc-test.h5
- generate/load the charge information 
- add the charge information and write out a new h5 file 
""" 

import h5py
import numpy as np
 

print('--- import success ---')

## input hdf5 file 
in_h5 = 'dppc-with-charge.h5' 
#################### cp ori_dppc.h5 dppc.h5
## check data via:   h5dump -n dppc.h5   
## HDF5 "dppc.h5" {
## FILE_CONTENTS {
##  group      /
##  dataset    /bonds
##  dataset    /coordinates
##  dataset    /indices
##  dataset    /molecules
##  dataset    /names
##  dataset    /typesd
##  dataset    /velocities
##  }
## }
### h5dump -d "/charge" dppc-with-charge.h5 

hf = h5py.File(in_h5, 'a')

####################### check the some read items 
##print(hf.keys())
coord_array = np.array(hf.get('coordinates'))[0]
##print( coord_array.shape )
name_array = np.array(hf.get('names'))
##print( name_array)
##print( name_array.shape )
names = hf.get('names')[:]
print(names)
##print(type(names))
##[b'N' b'P' b'G' ... b'W' b'W' b'W']
print(set(names))
## {b'C', b'W', b'N', b'P', b'G'}
######### define the charge according to the _particle name_
######### https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
particle_type_charge_dict = {
    b'C':   0.01,
    b'W':  -0.01, 
    b'N':   0.01,
    b'P':   0.01,
    b'G':   0.01 
} ## Here the 

####################### generate/load charge information 
####### as a first try: generate a uniform charge array 
##charges = np.ones(
##        shape=len(coord_array)  
##) 
#print(np.array(names))
charges = np.vectorize(particle_type_charge_dict.get)(np.array(names)) ## option1 
#charges = np.arange(len(names))
##charges = [particle_type_charge_dict.get(i) for i in names] ## option2 
print(charges)

####################### add the charge array 
if not hf.get('charge'):
    hf.create_dataset('charge', data=charges, dtype='float32')
    print(hf.keys()) 
hf.close()

hf = h5py.File(in_h5, 'r')
charges_read = hf.get('charge')[:]
print(charges_read)
print( np.around(charges_read - charges, 3) ) 
print(hf["charge"])






