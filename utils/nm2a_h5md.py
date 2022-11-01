import os
import argparse
import h5py
import numpy as np

def nm2a_h5md(h5md_file, overwrite=False, out_path=None):
    if out_path is None:
        out_path = os.path.join(os.path.abspath(os.path.dirname(h5md_file)),
                                os.path.splitext(os.path.split(h5md_file)[-1])[0]+'_ang'
                               +os.path.splitext(os.path.split(h5md_file)[-1])[1])
    if os.path.exists(out_path) and not overwrite:
        error_str = (f'The specified output file {out_path} already exists. '
                     f'use overwrite=True ("-f" flag) to overwrite.')
        raise FileExistsError(error_str)

    f_in = h5py.File(h5md_file, 'r')
    f_out = h5py.File(out_path, 'w')

    for k in f_in.keys():
        f_in.copy(k, f_out)

    box_size = f_in['particles/all/box/edges'][:] * 10.
    f_out['particles/all/box/edges'][:] = box_size

    tpos = f_in['particles/all/position/value'][:] * 10.
    f_out['particles/all/position/value'][:] = np.mod(tpos, box_size)

    f_in.close()
    f_out.close()

if __name__ == '__main__':
    description = 'Convert .H5 file coordinates from nm to A'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('h5md_file', type=str, help='input .H5MD file name')
    parser.add_argument('--out', type=str, default=None, dest='out_path',
                        metavar='file name', help='output hymd HDF5 file name')
    parser.add_argument('-f', action='store_true', default=False, dest='force',
                        help='overwrite existing output file')
    args = parser.parse_args()

    nm2a_h5md(args.h5md_file, overwrite=args.force, out_path=args.out_path)
