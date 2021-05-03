### reference
###    - https://medium.com/@jerilkuriakose/using-hdf5-with-python-6c5242d08773
###    - https://stackoverflow.com/questions/28770189/get-list-of-hdf5-contents-pandas-hdfstore
### - https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html
### - https://docs.h5py.org/en/stable/quick.html

import h5py
import numpy as np
import pandas as pd
from pandas import HDFStore 
from pandas import (
    DataFrame, HDFStore
)

print('--- import success ---')

## input hdf5 file 
in_h5 = 'dppc-test.h5' 
#################### cp dppc.h5 dppc-test.h5
## check data via:   h5dump -n dppc-test.h5   
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
## h5dump -d "/charges" dppc-test.h5 

hf = h5py.File(in_h5, 'a')

## test 
##print(hf.keys())
coord_array = np.array(hf.get('coordinates'))[0]
print( coord_array.shape )

name_array = np.array(hf.get('names'))
print( name_array)
print( name_array.shape )

names = hf.get('names')[:]
print(names)
print(type(names))
charges = np.ones(
        shape=len(coord_array)  
) 

if not hf.get('charge'):
    hf.create_dataset('charge', data=charges, dtype='float32')
    print(hf.keys()) 
hf.close()

hf = h5py.File(in_h5, 'r')
charges = hf.get('charge')[:]
print(charges)
print(type(charges))

### https://nongnu.org/h5md/proposals/0102_particles_charge.html





## 
# Create a storage file where data is to be stored
##df = HDFStore(in_h5)
##hdf.put('d2', DataFrame(np.random.randn(7,4)))
##print(hdf.__dict__)
#print(store.groups()) ##print(store.keys())
# reading a HDF5 file
# this method is not recommended
# the HDF5 file can store only a single file
#df = pd.read_hdf(in_h5)
#print(df)
#hdf.close()
