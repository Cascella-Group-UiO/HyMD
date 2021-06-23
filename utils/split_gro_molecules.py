import os
import subprocess
import argparse


def split_gro_molecules(path, out_path, overwrite=False):
    path = os.path.abspath(path)
    if out_path is None:
        xyz_file_name = os.path.basename(path)
        if '.gro' in xyz_file_name:
            gro_file_name = xyz_file_name.split('.')[0] + '.5'
        else:
            gro_file_name = xyz_file_name + '.5'
        out_path = os.path.join(os.path.dirname(path), gro_file_name)

    out_path = os.path.abspath(out_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Provided file {path} could not be found.')
    if os.path.exists(out_path) and not overwrite:
        error_str = (f'The specified output file {out_path} already exists. '
                     f'use overwrite=True ("-f" flag) to overwrite.')
        raise FileExistsError(error_str)

    wc = subprocess.run(['wc', '-l', f'{path}'], capture_output=True)
    total_lines = int(wc.stdout.split()[0])

    with open(path, 'r') as in_file, open(out_path, 'w') as out_file:
        doc_line = in_file.readline()
        line = in_file.readline()
        n_atoms = int(line)

        out_file.write(f'{doc_line.strip()}\n')
        out_file.write(f'\t{n_atoms}\n')

        mol_index = 0
        for i in range(total_lines - 3):
            line = in_file.readline()

            if i == 0:
                sline = line.split()
                # no_velocity_gro = None
                if len(sline) == 6:
                    # no_velocity_gro = True
                    raise NotImplementedError(
                        '.gro files not containing velocities not supported'
                    )
                elif len(sline) == 9:
                    # no_velocity_gro = False
                    ...
                else:
                    raise ValueError(
                        f'Unrecognized line in .gro, {line}\n'
                        f'Expected 6 or 9 strings, got {len(line)}'
                    )
            _ = int(line[:5])
            molecule_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            atom_index = int(line[15:20])
            pos_x = float(line[20:28])
            pos_y = float(line[28:36])
            pos_z = float(line[36:44])
            vel_x = float(line[44:52])
            vel_y = float(line[52:60])
            vel_z = float(line[60:68])

            if molecule_name == 'DPPC' and atom_name == 'NC3':
                mol_index += 1
            if molecule_name == 'W' and atom_name == 'W':
                mol_index += 1
            if mol_index == 100000:
                mol_index = 1
            out_string = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n" % (
                mol_index, molecule_name, atom_name, atom_index,
                pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
            )
            out_file.write(out_string)

        box_line = in_file.readline()
        out_file.write(f'{box_line.strip()}\n')


if __name__ == '__main__':
    description = 'Split incorrectly combined .gro file molecules'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('file_name', type=str, help='input .gro file name')
    parser.add_argument('--out', type=str, default=None, dest='out_path',
                        metavar='file name', help='output fort.5 file name')
    parser.add_argument('-f', action='store_true', default=False, dest='force',
                        help='overwrite existing output file')
    args = parser.parse_args()

    split_gro_molecules(args.file_name, args.out_path, overwrite=args.force)
