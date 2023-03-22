import os
import argparse
import h5py
import numpy as np

def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    From https://stackoverflow.com/a/11541495
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def add_charges(hdf5_file, charges, out_path=None, overwrite=False):
    if out_path is None:
        out_path = os.path.join(os.path.abspath(os.path.dirname(hdf5_file)),
                                os.path.splitext(hdf5_file)[0]+"_new.hdf5")
    if os.path.exists(out_path) and not overwrite:
        error_str = (f'The specified output file {out_path} already exists. '
                     f'use overwrite=True ("-f" flag) to overwrite.')
        raise FileExistsError(error_str)

    # read charges
    charge_dict = {}
    with open(charges, "r") as f:
        for line in f:
            beadtype, beadchrg = line.split()
            charge_dict[beadtype] = float(beadchrg)

    # create output file with the same values
    f_in = h5py.File(hdf5_file, 'r')
    f_out = h5py.File(out_path, 'w')

    for k in f_in.keys():
        f_in.copy(k, f_out)

    # create charges array
    charges_dataset = f_out.create_dataset(
    "charge",
    f_out["names"][:].shape,
    dtype="float32",
    )
    charges_list = []
    for bead in f_out["names"][:].tolist():
        charges_list.append(charge_dict[bead.decode()])

    charges_dataset[:] = np.array(charges_list)

    f_in.close()
    f_out.close()


if __name__ == '__main__':
    description = 'Given a .hdf5 input and a charges.txt file assigning bead type to charges, create new .hdf5.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('hdf5_file', type=str, help='input .hdf5 file name')
    parser.add_argument('charges', type=extant_file, default=None,
                        help='file containing the charge for each atom type')
    parser.add_argument('--out', type=str, default=None, dest='out_path',
                        metavar='file name', help='output hymd HDF5 file name')
    parser.add_argument('-f', action='store_true', default=False, dest='force',
                        help='overwrite existing output file')
    args = parser.parse_args()

    add_charges(args.hdf5_file, args.charges,
                overwrite=args.force, out_path=args.out_path)
