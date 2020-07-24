import h5py
import sys



f_hd5      = h5py.File(sys.argv[1], 'r')
fp = open('%s.gro'%(sys.argv[1].split('.')[0]),'w')

shape=f_hd5['coordinates'].shape[0]
names=f_hd5['names'][:]
Np=f_hd5['coordinates'].shape[1]
for f in range(shape):
    
    fp.write('MD of %d mols, t=%.3f\n'%(Np,f_hd5['time'][f]))
    fp.write('%-10d\n'%(Np))
    for i in range(Np):
        fp.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n"%(i//10+1,names[i],names[i],i+1,f_hd5['coordinates'][f,i,0],f_hd5['coordinates'][f,i,1],f_hd5['coordinates'][f,i,2],f_hd5['velocities'][f,i,0],f_hd5['velocities'][f,i,1],f_hd5['velocities'][f,i,2]))
    fp.write("%-5.5f\t%5.5f\t%5.5f\n"%(f_hd5['cell_lengths'][0,0,0],f_hd5['cell_lengths'][0,1,1],f_hd5['cell_lengths'][0,2,2]))
    fp.flush()

# dset_pos =




