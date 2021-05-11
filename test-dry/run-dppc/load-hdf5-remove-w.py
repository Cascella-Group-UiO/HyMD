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
## $ cp dppc.h5 dppc-dry.h5   
in_h5 = 'dppc-dry.h5' 

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
### h5dump -n dppc-dry.h5 

## load the h5 file
hf = h5py.File(in_h5, 'r+')

## find the index of first name = W  or type == 4  
target = b'W'
check = np.array(hf.get('names'))
##print(check)
##print(len(check))
cutoff_idx = check.tolist().index(target)

check = np.array(hf.get('types'))[:cutoff_idx]
print(check)
print(len(check))




#### GEM https://stackoverflow.com/questions/22922584/how-to-overwrite-array-inside-h5-file-using-h5py
#### !!!! 
##### askewchan's answer describes the way to do it (you cannot create a dataset under a name that already exists, 
##### but you can of course modify the dataset's data). Note, however, that the dataset must have the same shape as
##### the data (X1) you are writing to it. If you want to replace the dataset with some other dataset of different shape, 
##### you first have to delete it:
##### del f1['meas/frame1/data']
##### dset = f1.create_dataset('meas/frame1/data', data=X1)


for  dataset_name in ['bonds', 'coordinates','indices','molecules','names','types','velocities']:
    print(hf[dataset_name])
    new_data = hf[dataset_name][:cutoff_idx]
    del hf[dataset_name]
    hf.create_dataset(dataset_name, data=new_data)
    
####################### check the some read items 
##print(hf.keys())
#coord_array = np.array(hf.get('coordinates'))
#print(coord_array)











hf.close()


"""
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
    b'C':   0.1,
    b'W':  -0.1, 
    b'N':   0.1,
    b'P':   0.1,
    b'G':   0.1 
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
"""





