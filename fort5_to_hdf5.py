import numpy as np
import os
import argparse
import h5py


def is_int(possible_int):
    try:
        _ = int(possible_int)
        return True
    except ValueError:
        return False


def is_number(possible_number):
    try:
        _ = float(possible_number)
        return True
    except ValueError:
        return False


def check_molecule_type(lines):
    atoms = ['' for _ in range(len(lines))]
    for i, line in enumerate(lines):
        split_line = line.split()
        atoms[i] = split_line[1]

    if len(lines) == 10:
        expected_atoms_end = ['N', 'P', 'G', 'G',
                              'C', 'C', 'CE',
                              'C', 'C', 'CE']
        expected_atoms = ['N', 'P', 'G', 'G',
                          'C', 'C', 'C',
                          'C', 'C', 'C']
        if atoms == expected_atoms or atoms == expected_atoms_end:
            return 'DMPC'
    if len(lines) == 12:
        expected_atoms_end = ['N', 'P', 'G', 'G',
                              'C', 'C', 'C', 'CE',
                              'C', 'C', 'C', 'CE']
        expected_atoms = ['N', 'P', 'G', 'G',
                          'C', 'C', 'C', 'C',
                          'C', 'C', 'C', 'C']
        if atoms == expected_atoms or atoms == expected_atoms_end:
            return 'DPPC'
    elif len(lines) == 14:
        expected_atoms_end = ['N', 'P', 'G', 'G',
                              'C', 'C', 'C', 'C', 'CE',
                              'C', 'C', 'C', 'C', 'CE']
        expected_atoms = ['N', 'P', 'G', 'G',
                          'C', 'C', 'C', 'C', 'C',
                          'C', 'C', 'C', 'C', 'C']
        if atoms == expected_atoms or atoms == expected_atoms_end:
            return 'DSPC'
        expected_atoms_end = ['N', 'P', 'G', 'G',
                              'C', 'C', 'D', 'C', 'CE',
                              'C', 'C', 'D', 'C', 'CE']
        expected_atoms = ['N', 'P', 'G', 'G',
                          'C', 'C', 'D', 'C', 'C',
                          'C', 'C', 'D', 'C', 'C']
        if atoms == expected_atoms or atoms == expected_atoms_end:
            return 'DOPC'

    elif len(lines) == 5:
        expected_atoms = ['B' for _ in range(5)]
        if atoms == expected_atoms:
            return 'B5'

    elif len(lines) == 1:
        if atoms[0] == 'B':
            return 'B'
        elif atoms[0] == 'W':
            return 'W'
        elif atoms[0] == 'WF':
            return 'WF'

    raise NotImplementedError('Only water, DPPC, DMPC, DOPC, DSPC supported')


def fort5_to_hdf5(path, out_path=None, force=False):
    if out_path is None:
        out_path = os.path.join(os.path.dirname(path), 'converted.gro')

    if os.path.exists(out_path) and not force:
        raise FileExistsError(f'The target .h5 file, {out_path} already '
                              f'exists, set force=True (-f flag) to overwrite')

    with open(path, 'r') as in_file:
        lines = in_file.readlines()
    box = np.array([float(lines[1].split()[i]) for i in range(3)])
    n_atoms = int(lines[-1].split()[0])
    if is_number(lines[2].split()[0]):
        n_molecules = int(lines[4].split()[0])
    else:
        n_molecules = int(lines[3].split()[0])

    f_hd5 = h5py.File('input.hdf5', 'w')

    dset_pos = f_hd5.create_dataset("coordinates", (1, n_atoms, 3),
                                    dtype="Float32")
    dset_vel = f_hd5.create_dataset("velocities", (1, n_atoms, 3),
                                    dtype="Float32")
    dset_types = f_hd5.create_dataset("types", (n_atoms,), dtype="i")
    dset_molecule_type = f_hd5.create_dataset("molecule_type", (n_atoms,),
                                              dtype="i")
    dset_molecule_index = f_hd5.create_dataset("molecules", (n_atoms,),
                                               dtype="i")
    dset_indicies = f_hd5.create_dataset("indicies", (n_atoms,), dtype="i")
    dset_names = f_hd5.create_dataset("names", (n_atoms,), dtype="S5")
    dset_bonds = f_hd5.create_dataset("bonds", (n_atoms, 2), dtype="i")
    f_hd5.attrs['box'] = box
    f_hd5.attrs['n_molecules'] = n_molecules

    molecule_index = 0

    def write_to_dataset(molecule_lines, molecule_index):
        atom_indices = [int(s.split()[0]) for s in molecule_lines]
        type_indices = [int(s.split()[2]) for s in molecule_lines]
        names = [s.split()[1] for s in molecule_lines]
        if len(names) == 1 and names[0] == 'A':
            molecule_type = 0
        elif len(names) == 5 and names == ['B' for _ in range(5)]:
            molecule_type = 1

        for i, line in enumerate(molecule_lines):
            atom_index = atom_indices[i] - 1
            type_index = type_indices[i] - 1
            pos_vel = [float(s.replace('D', 'E')) for s in line.split()[4:10]]
            dset_pos[0, atom_index, :] = np.array(pos_vel[:3])
            dset_vel[0, atom_index, :] = np.array(pos_vel[3:])
            dset_types[atom_index] = type_index
            dset_molecule_type[atom_index] = molecule_type
            dset_molecule_index[atom_index] = molecule_index
            dset_names[atom_index] = np.string_(names[i])
            dset_indicies[atom_index] = atom_index

            bonds = [int(s) for s in line.split()[10:]]
            dset_bonds[atom_index] = bonds[:2]
        molecule_index += 1

    for i, line in enumerate(lines):
        split_line = line.split()
        if i > 4 and len(split_line) == 1 and is_int(split_line[0]):
            molecule_start_line = i
            atoms_in_molecule = int(split_line[0])

            molecules_lines = lines[
                molecule_start_line + 1:
                molecule_start_line + atoms_in_molecule + 1
            ]
            write_to_dataset(molecules_lines, molecule_index)
            continue
            """
            molecule_type = check_molecule_type(molecules_lines)
            if molecule_type == 'DPPC':
                NotImplementedError('Fix this')
                # write_DPPC(out_file, molecules_lines, molecule_index)
                # molecule_index += 1
            elif molecule_type == 'DSPC':
                NotImplementedError('Fix this')
                # write_DSPC(out_file, molecules_lines, molecule_index)
                # molecule_index += 1
            elif molecule_type == 'DOPC':
                NotImplementedError('Fix this')
                # write_DOPC(out_file, molecules_lines, molecule_index)
                # molecule_index += 1
            elif molecule_type == 'DMPC':
                NotImplementedError('Fix this')
                # write_DMPC(out_file, molecules_lines, molecule_index)
                # molecule_index += 1
            elif molecule_type == 'W':
                NotImplementedError('Fix this')
                # write_water(out_file, molecules_lines, water_index)
                # water_index += 1
            else:
                raise NotImplementedError(
                    "unrecognized mol type {molecule_type}"
                )
            """


if __name__ == '__main__':
    description = 'Convert fort.5 OCCAM format to HDF5 format files'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('file_name', type=str, help='input fort.5 file name')
    parser.add_argument('--out', type=str, default=None, dest='out_path',
                        metavar='file name', help='output .h5 file name')
    parser.add_argument('-f', action='store_true', default=False, dest='force',
                        help='overwrite existing output file')
    args = parser.parse_args()

    fort5_to_hdf5(args.file_name, out_path=args.out_path, force=args.force)
