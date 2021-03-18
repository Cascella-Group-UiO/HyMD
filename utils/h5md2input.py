import os
import argparse
import h5py


def h5md_to_input(h5md_file, old_input, out_path=None, overwrite=False):
    if out_path is None:
        out_path = os.path.join(os.path.abspath(os.path.dirname(old_input)),
                                os.path.split(old_input)[-1] + '_new')
    if os.path.exists(out_path) and not overwrite:
        error_str = (f'The specified output file {out_path} already exists. '
                     f'use overwrite=True ("-f" flag) to overwrite.')
        raise FileExistsError(error_str)

    f_in = h5py.File(old_input, 'r')
    f_out = h5py.File(out_path, 'w')
    new_values = h5py.File(h5md_file, 'r')

    for k in f_in.keys():
        f_in.copy(k, f_out)

    new_positions = new_values['particles/all/position/value'][-1, :, :]
    new_velocities = new_values['particles/all/velocity/value'][-1, :, :]

    f_out['coordinates'][:, :] = new_positions
    f_out['velocities'][:, :] = new_velocities

    f_in.close()
    f_out.close()
    new_values.close()


if __name__ == '__main__':
    description = 'Convert .h5md files to the hymd input format'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('h5md_file', type=str, help='input .H5MD file name')
    parser.add_argument('old_input_file', type=str,
                        help='previous .H5 file name')
    parser.add_argument('--out', type=str, default=None, dest='out_path',
                        metavar='file name', help='output hymd HDF5 file name')
    parser.add_argument('-f', action='store_true', default=False, dest='force',
                        help='overwrite existing output file')
    args = parser.parse_args()

    h5md_to_input(args.h5md_file, args.old_input_file, overwrite=args.force,
                  out_path=args.out_path)
