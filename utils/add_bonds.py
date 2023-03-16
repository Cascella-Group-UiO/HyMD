import os
import argparse
import h5py
import numpy as np
import tomli

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


def read_config_toml(file_path):
    with open(file_path, "rb") as in_file:
        toml_config = tomli.load(in_file)
    return toml_config


def prepare_index_based_bonds(molecules, topol):
    bonds_2 = []

    different_molecules = np.unique(molecules)
    for mol in different_molecules:
        resid = mol + 1
        top_summary = topol["system"]["molecules"]
        resname = None
        test_mol_number = 0
        for molname in top_summary:
            test_mol_number += molname[1]
            if resid <= test_mol_number:
                resname = molname[0]
                break

        if resname is None:
            break

        if "bonds" in topol[resname]:
            first_id = np.where(molecules == mol)[0][0]
            for bond in topol[resname]["bonds"]:
                index_i = bond[0] - 1 + first_id
                index_j = bond[1] - 1 + first_id
                equilibrium = bond[3]
                strength = bond[4]
                bonds_2.append([index_i, index_j, equilibrium, strength])

    return bonds_2


def add_bonds(hdf5_file, topofile, out_path=None, overwrite=False):
    if out_path is None:
        out_path = os.path.join(os.path.abspath(os.path.dirname(hdf5_file)),
                                os.path.splitext(hdf5_file)[0]+"_new.hdf5")
    if os.path.exists(out_path) and not overwrite:
        error_str = (f'The specified output file {out_path} already exists. '
                     f'use overwrite=True ("-f" flag) to overwrite.')
        raise FileExistsError(error_str)

    # read topology
    topol = read_config_toml(topofile)
    if "include" in topol["system"]:
        for file in topol["system"]["include"]:
            itps = read_config_toml(file)
            for mol, itp in itps.items():
                topol[mol] = itp

    # create output file with the same values
    f_in = h5py.File(hdf5_file, 'r')
    f_out = h5py.File(out_path, 'w')

    for k in f_in.keys():
        f_in.copy(k, f_out)

    n_particles = len(f_in["names"][:].tolist())

    # create bonds dataset and fill it
    bonds_dataset = f_out.create_dataset(
        "bonds",
        (n_particles, 4,),
        dtype="i",
    )
    bonds_dataset[...] = -1
    bonds = prepare_index_based_bonds(f_out["molecules"], topol)
    for bond in bonds:
        idx = 0
        while bonds_dataset[bond[0], idx] != -1:
            idx += 1
        bonds_dataset[bond[0], idx] = bond[1]

        idx = 0
        while bonds_dataset[bond[1], idx] != -1:
            idx += 1
        bonds_dataset[bond[1], idx] = bond[0]

    f_in.close()
    f_out.close()


if __name__ == '__main__':
    description = 'Given a .hdf5 input and a .toml with bonds create new bonded .hdf5.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('hdf5_file', type=str, help='input .hdf5 file name')
    parser.add_argument('topology_file', type=extant_file, default=None,
                        help='toml file with topology')
    parser.add_argument('--out', type=str, default=None, dest='out_path',
                        metavar='file name', help='output hymd HDF5 file name')
    parser.add_argument('-f', action='store_true', default=False, dest='force',
                        help='overwrite existing output file')
    args = parser.parse_args()

    add_bonds(args.hdf5_file, args.topology_file,
                overwrite=args.force, out_path=args.out_path)
