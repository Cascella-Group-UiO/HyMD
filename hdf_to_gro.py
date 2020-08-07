import h5py
import argparse

description = 'Convert .hdf5 data files to .gro format'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('hdf5_file', type=str, help='input .hdf5 file name')
parser.add_argument('--out', type=str, help='output file name', default=None)
parser.add_argument('--molecule_file', type=str,
                    help='optional .hdf5 file containing molecule details')
args = parser.parse_args()

f_hd5 = h5py.File(args.hdf5_file, 'r')
if args.out is None:
    fp = open('%s.gro' % (args.hdf5_file), 'w')
else:
    fp = open(args.out, 'w')


shape = f_hd5['coordinates'].shape[0]
names = f_hd5['names'][:]
Np = f_hd5['coordinates'].shape[1]

for f in range(shape):
    fp.write('MD of %d mols, t=%.3f\n' % (Np, f_hd5['time'][f]))
    fp.write('%-10d\n' % (Np))
    for i in range(Np):
        name = names[i].decode('UTF-8').strip()
        dump_str = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n" % (
            i//10+1,
            name,
            name,
            i+1,
            f_hd5['coordinates'][f, i, 0],
            f_hd5['coordinates'][f, i, 1],
            f_hd5['coordinates'][f, i, 2],
            f_hd5['velocities'][f, i, 0],
            f_hd5['velocities'][f, i, 1],
            f_hd5['velocities'][f, i, 2]
        )
        fp.write(dump_str)
    fp.write("%-5.5f\t%5.5f\t%5.5f\n" % (
        f_hd5['cell_lengths'][0, 0, 0],
        f_hd5['cell_lengths'][0, 1, 1],
        f_hd5['cell_lengths'][0, 2, 2])
    )
    fp.flush()
