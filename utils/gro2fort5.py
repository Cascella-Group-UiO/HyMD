import os
import subprocess
import argparse
import numpy as np


def _format_fort5_line(index, atom_type, atom_index, total_bonds,
                       x, y, z, vx, vy, vz, bonds, velocity=True,
                       zero_velocity=False):
    return_str = f'{index:>5d} {atom_type:<3s} {atom_index} {total_bonds}\t'
    return_str += ''.join(f'{p:<20.15f}' for p in (x, y, z))
    if velocity:
        if zero_velocity:
            return_str += ''.join(f'{v:<20.15f}' for v in (0.0, 0.0, 0.0))
        else:
            return_str += ''.join(f'{v:<20.15f}' for v in (vx, vy, vz))
    return_str += '\t' + ' '.join(f'{b}' for b in bonds)
    return return_str, [vx, vy, vz]


def _convert_DPPC(molecule_index, name, index,
                  x, y, z, vx, vy, vz, split_carbons=False, velocity=True,
                  zero_velocity=False, water_index=5):
    index, x, y, z, vx, vy, vz = (int(index), float(x), float(y), float(z),
                                  float(vx), float(vy), float(vz))
    bonds = [0, 0, 0, 0, 0, 0]
    if name == 'NC3':  # index 1, bonds 2
        atom_type = 'N'
        atom_index = 1
        bonds[0] = index + 1
        total_bonds = 1
    elif name == 'PO4':  # index 2, bonds 1 3
        atom_type = 'P'
        atom_index = 2
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'GL1':  # index 3, bonds 2 4 5
        atom_type = 'G'
        atom_index = 3
        bonds[:3] = index - 1, index + 1, index + 2
        total_bonds = 3
    elif name == 'GL2':  # index 4, bonds 3 9
        atom_type = 'G'
        atom_index = 3
        bonds[:2] = index - 1, index + 5
        total_bonds = 2
    elif name == 'C1A':  # index 5, bonds 3 6
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 2, index + 1
        total_bonds = 2
    elif name == 'C2A':  # index 6, 5 7
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C3A':  # index 7, 6 8
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C4A':  # index 8, 7
        atom_type = 'CE' if split_carbons else 'C'
        atom_index = 6 if split_carbons else 4
        bonds[0] = index - 1
        total_bonds = 1
    elif name == 'C1B':  # index 9, 4 10
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 5, index + 1
        total_bonds = 2
    elif name == 'C2B':  # index 10, 9 11
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C3B':  # index 11, 10 12
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C4B':  # index 12, 11
        atom_type = 'CE' if split_carbons else 'C'
        atom_index = 6 if split_carbons else 4
        bonds[0] = index - 1
        total_bonds = 1
    else:
        raise ValueError(f'Unrecognized name in DPPC: {name}')
    return _format_fort5_line(index, atom_type, atom_index, total_bonds,
                              x, y, z, vx, vy, vz, bonds, velocity=velocity,
                              zero_velocity=zero_velocity)


"""
FORT.5
-------------------------------------------------------------------------------
1   N      1  1      x y z   vx vy vz    2
2   P      2  2      x y z   vx vy vz    1        3
3   G      3  3      x y z   vx vy vz    2        4         5
4   G      3  2      x y z   vx vy vz    3        10
5   C      4  2      x y z   vx vy vz    3        6
6   C      4  2      x y z   vx vy vz    5        7
7   D      5  2      x y z   vx vy vz    6        8
8   C      4  2      x y z   vx vy vz    7        9
9   C      4  1      x y z   vx vy vz    8
10   C      4  2     x y z   vx vy vz    4        11
11   C      4  2     x y z   vx vy vz    10       12
12   D      5  2     x y z   vx vy vz    11       13
13   C      4  2     x y z   vx vy vz    12       14
14   C      4  1     x y z   vx vy vz    13

GRO
-------------------------------------------------------------------------------
    1DOPC   NC3    1   0.933   0.415   4.450
    1DOPC   PO4    2   0.878   0.370   4.150
    1DOPC   GL1    3   0.928   0.463   3.850
    1DOPC   GL2    4   1.056   0.406   3.850
    1DOPC   C1A    5   0.864   0.402   3.550
    1DOPC   D2A    6   0.895   0.429   3.250
    1DOPC   C3A    7   0.871   0.445   2.950
    1DOPC   C4A    8   0.861   0.369   2.650
    1DOPC   C1B    9   1.251   0.317   3.550
    1DOPC   D2B   10   1.244   0.321   3.250
    1DOPC   C3B   11   1.203   0.320   2.950
    1DOPC   C4B   12   1.213   0.309   2.650

    1DOPC   NC3    1   0.328   4.563   9.750
    1DOPC   PO4    2   0.331   4.516   9.450
    1DOPC   GL1    3   0.304   4.547   9.150
    1DOPC   GL2    4   0.494   4.673   9.150
    1DOPC   C1A    5   0.304   4.588   8.850
    1DOPC   C2A    6   0.244   4.570   8.550
    1DOPC   D3A    7   0.245   4.578   8.250
    1DOPC   C4A    8   0.294   4.501   7.950
    1DOPC   C5A    9   0.259   4.594   7.650
    1DOPC   C1B   10   0.637   4.748   8.850
    1DOPC   C2B   11   0.614   4.711   8.550
    1DOPC   D3B   12   0.619   4.746   8.250
    1DOPC   C4B   13   0.664   4.743   7.950
    1DOPC   C5B   14   0.601   4.714   7.650
"""


def _convert_DOPC(molecule_index, name, index,
                  x, y, z, vx, vy, vz, split_carbons=False, velocity=True,
                  zero_velocity=False, water_index=5):
    index, x, y, z, vx, vy, vz = (int(index), float(x), float(y), float(z),
                                  float(vx), float(vy), float(vz))
    bonds = [0, 0, 0, 0, 0, 0]
    if name == 'NC3':  # index 1, bonds 2
        atom_type = 'N'
        atom_index = 1
        bonds[0] = index + 1
        total_bonds = 1
    elif name == 'PO4':  # index 2, bonds 1 3
        atom_type = 'P'
        atom_index = 2
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'GL1':  # index 3, bonds 2 4 5
        atom_type = 'G'
        atom_index = 3
        bonds[:3] = index - 1, index + 1, index + 2
        total_bonds = 3
    elif name == 'GL2':  # index 4, bonds 3 10
        atom_type = 'G'
        atom_index = 3
        bonds[:2] = index - 1, index + 6
        total_bonds = 2
    elif name == 'C1A':  # index 5, bonds 3 6
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 2, index + 1
        total_bonds = 2
    elif name == 'C2A':  # index 6, 5 7
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'D3A':  # index 7, 6 8
        atom_type = 'D'
        atom_index = 5
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C4A':  # index 8, 7 9
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C5A':  # index 9, 8
        atom_type = 'C'
        atom_index = 4
        bonds[0] = index - 1
        total_bonds = 1

    elif name == 'C1B':  # index 10, 4 11
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 6, index + 1
        total_bonds = 2
    elif name == 'C2B':  # index 11, 10 12
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'D3B':  # index 12, 11 13
        atom_type = 'D'
        atom_index = 5
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C4B':  # index 13, 12 14
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C5B':  # index 14, 13
        atom_type = 'C'
        atom_index = 4
        bonds[0] = index - 1
        total_bonds = 1
    else:
        raise ValueError(f'Unrecognized name in DOPC: {name}')
    return _format_fort5_line(index, atom_type, atom_index, total_bonds,
                              x, y, z, vx, vy, vz, bonds, velocity=velocity,
                              zero_velocity=zero_velocity)


def _convert_DSPC(molecule_index, name, index,
                  x, y, z, vx, vy, vz, split_carbons=False, velocity=True,
                  zero_velocity=False, water_index=5):
    index, x, y, z, vx, vy, vz = (int(index), float(x), float(y), float(z),
                                  float(vx), float(vy), float(vz))
    bonds = [0, 0, 0, 0, 0, 0]
    if name == 'NC3':  # index 1, bonds 2
        atom_type = 'N'
        atom_index = 1
        bonds[0] = index + 1
        total_bonds = 1
    elif name == 'PO4':  # index 2, bonds 1 3
        atom_type = 'P'
        atom_index = 2
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'GL1':  # index 3, bonds 2 4 5
        atom_type = 'G'
        atom_index = 3
        bonds[:3] = index - 1, index + 1, index + 2
        total_bonds = 3
    elif name == 'GL2':  # index 4, bonds 3 10
        atom_type = 'G'
        atom_index = 3
        bonds[:2] = index - 1, index + 6
        total_bonds = 2
    elif name == 'C1A':  # index 5, bonds 3 6
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 2, index + 1
        total_bonds = 2
    elif name == 'C2A':  # index 6, 5 7
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C3A':  # index 7, 6 8
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C4A':  # index 8, 7 9
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C5A':  # index 9, 8
        atom_type = 'C'
        atom_index = 4
        bonds[0] = index - 1
        total_bonds = 1

    elif name == 'C1B':  # index 10, 4 11
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 6, index + 1
        total_bonds = 2
    elif name == 'C2B':  # index 11, 10 12
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C3B':  # index 12, 11 13
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C4B':  # index 13, 12 14
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C5B':  # index 14, 13
        atom_type = 'C'
        atom_index = 4
        bonds[0] = index - 1
        total_bonds = 1
    else:
        raise ValueError(f'Unrecognized name in DSPC: {name}')
    return _format_fort5_line(index, atom_type, atom_index, total_bonds,
                              x, y, z, vx, vy, vz, bonds, velocity=velocity,
                              zero_velocity=zero_velocity)


def _convert_DMPC(molecule_index, name, index,
                  x, y, z, vx, vy, vz, split_carbons=False, velocity=True,
                  zero_velocity=False, water_index=5):
    index, x, y, z, vx, vy, vz = (int(index), float(x), float(y), float(z),
                                  float(vx), float(vy), float(vz))
    bonds = [0, 0, 0, 0, 0, 0]
    if name == 'NC3':  # index 1, bonds 2
        atom_type = 'N'
        atom_index = 1
        bonds[0] = index + 1
        total_bonds = 1
    elif name == 'PO4':  # index 2, bonds 1 3
        atom_type = 'P'
        atom_index = 2
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'GL1':  # index 3, bonds 2 4 5
        atom_type = 'G'
        atom_index = 3
        bonds[:3] = index - 1, index + 1, index + 2
        total_bonds = 3
    elif name == 'GL2':  # index 4, bonds 3 8
        atom_type = 'G'
        atom_index = 3
        bonds[:2] = index - 1, index + 4
        total_bonds = 2
    elif name == 'C1A':  # index 5, bonds 3 6
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 2, index + 1
        total_bonds = 2
    elif name == 'C2A':  # index 6, bonds 5 7
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C3A':  # index 7, bonds 6
        atom_type = 'C'
        atom_index = 4
        bonds[0] = index - 1
        total_bonds = 1

    elif name == 'C1B':  # index 8, bonds 4 9
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 4, index + 1
        total_bonds = 2
    elif name == 'C2B':  # index 9, bonds 8 10
        atom_type = 'C'
        atom_index = 4
        bonds[:2] = index - 1, index + 1
        total_bonds = 2
    elif name == 'C3B':  # index 10, bonds 9
        atom_type = 'C'
        atom_index = 4
        bonds[0] = index - 1
        total_bonds = 1
    else:
        raise ValueError(f'Unrecognized name in DMPC: {name}')
    return _format_fort5_line(index, atom_type, atom_index, total_bonds,
                              x, y, z, vx, vy, vz, bonds, velocity=velocity,
                              zero_velocity=zero_velocity)


def _convert_W(molecule_index, name, index,
               x, y, z, vx, vy, vz, velocity=True, zero_velocity=False,
               water_index=5, split_carbons=False):
    index, x, y, z, vx, vy, vz = (int(index), float(x), float(y), float(z),
                                  float(vx), float(vy), float(vz))
    total_bonds = 0
    bonds = [0, 0, 0, 0, 0, 0]
    atom_type = 'W'
    atom_index = water_index
    return _format_fort5_line(index, atom_type, atom_index, total_bonds,
                              x, y, z, vx, vy, vz, bonds, velocity=velocity,
                              zero_velocity=zero_velocity)


def gro2fort5(path, out_path, overwrite=False, split_carbons=False,
              velocity=True, zero_velocity=False,
              cancel_momentum_separately=False, water_index=5):
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

    with open(path, 'r') as in_file:
        _ = in_file.readline()
        line = in_file.readline()
        n_atoms = int(line)
        line = in_file.readline().split()
        if len(line) == 6:
            no_velocity_gro = True
        elif len(line) == 9:
            no_velocity_gro = False
        else:
            raise ValueError(f'Unrecognized line in .gro, {line}\n'
                             f'Expected 6 or 9 strings, got {len(line)}')

    wc = subprocess.run(['wc', '-l', f'{path}'], capture_output=True)
    tail = subprocess.run(['tail', '-n', '2', f'{path}'], capture_output=True)
    total_lines = int(wc.stdout.split()[0])
    last_two_lines = tail.stdout
    last_two_lines = last_two_lines.decode().split('\n')
    if len(last_two_lines) == 3 and last_two_lines[-1] == '':
        last_two_lines = [l for l in last_two_lines[:2]]

    box = [float(s) for s in last_two_lines[-1].split()]
    box.append(0.0)

    n_molecules = 0

    n_lipid_atoms = 0
    n_solvent_atoms = 0
    n_lipid_molecules = 0
    n_solvent_molecules = 0

    with open(path, 'r') as in_file, open(out_path, 'w') as out_file:
        out_file.write('box:\n\t')
        out_file.write('\t'.join(f'{b:20.15f}' for b in box[:3]) + '   0.0\n')
        out_file.write('total number of molecules:\n')

        # Spaces intentionally inserted so we can open the file in 'r+' mode
        # later and edit this line in place without having to rewrite the
        # whole file. The {n_molecules} is just a placeholder at this point.
        out_file.write(f'{n_molecules}                                     \n')

        v_total = np.zeros(3)
        v_lipid = np.zeros(3)
        v_solvent = np.zeros(3)

        molecule_add = 0
        atom_add = 0
        last_molecule_index = -1
        last_atom_index = -1

        for i, line in enumerate(in_file):
            if i < 2:
                continue
            elif i >= n_atoms+2:
                break

            if i % (total_lines//min(10000, total_lines)) == 0:
                mol_print = last_molecule_index + molecule_add * (99999+1)
                atom_print = last_atom_index + atom_add * (99999+1)
                print(f'Progress: {i/total_lines*100.0:5.3f} %', end='')
                print(f'{" "*10}molecules: {mol_print:10d}', end='')
                print(f'{" "*10}atoms: {atom_print:10d}', end='\r')

            molecule_index = int(line[:5])
            molecule_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            atom_index = int(line[15:20])
            pos_x = float(line[20:28])
            pos_y = float(line[28:36])
            pos_z = float(line[36:44])

            if i > 39608 and i < 39670:
                print(f"{i+1:7} RAW:    ", line, end="")
                print(f"{i+1:7} PARSED: ", molecule_index, molecule_name, atom_name, atom_index, pos_x, pos_y, pos_z)  # noqa: E501

            if no_velocity_gro:
                vel_x = 0.0
                vel_y = 0.0
                vel_z = 0.0
            else:
                vel_x = float(line[44:52])
                vel_y = float(line[52:60])
                vel_z = float(line[60:68])

            if molecule_index != last_molecule_index:
                n_molecules += 1
                if last_molecule_index > 0 and molecule_index == 0:
                    molecule_add += 1

                molecule_write = molecule_index + molecule_add * (99999+1)
                out_file.write(f'molecule #{molecule_write}\n')

                if molecule_name == 'DPPC':
                    out_file.write('\t12\n')
                    n_lipid_molecules += 1
                elif molecule_name == 'DOPC' or molecule_name == 'DGPC':
                    if water_index == 5:
                        raise ValueError(
                            'Water index in the chi matrix overlaps with the '
                            'particles in the DOPC membrane.\n'
                            'Use --water-index 6 when DOPC lipids are present'
                        )
                    out_file.write('\t14\n')
                    n_lipid_molecules += 1
                elif molecule_name == 'DMPC' or molecule_name == 'DLPC':
                    out_file.write('\t10\n')
                    n_lipid_molecules += 1
                elif molecule_name == 'DSPC' or molecule_name == 'DBPC':
                    out_file.write('\t14\n')
                    n_lipid_molecules += 1
                elif molecule_name == 'W':
                    out_file.write('\t1\n')
                    n_solvent_molecules += 1
                elif molecule_name == 'WF':
                    out_file.write('\t1\n')
                    n_solvent_molecules += 1
                else:
                    raise NotImplementedError(
                        f'Only DMPC, DSPC, DOPC, DPPC, W, and WF supported, '
                        f'not {molecule_name}'
                    )
            last_molecule_index = molecule_index

            atom_write = atom_index + atom_add * (99999+1)
            if atom_index == 99999:
                atom_add += 1
            last_atom_index = atom_index

            if molecule_name == 'DPPC':
                convert_function = _convert_DPPC
                n_lipid_atoms += 1

            elif molecule_name == 'DOPC' or molecule_name == 'DGPC':
                n_lipid_atoms += 1
                convert_function = _convert_DOPC

            elif molecule_name == 'DSPC' or molecule_name == 'DBPC':
                n_lipid_atoms += 1
                convert_function = _convert_DSPC

            elif molecule_name == 'DMPC' or molecule_name == 'DLPC':
                n_lipid_atoms += 1
                convert_function = _convert_DMPC

            elif molecule_name == 'W' or molecule_name == 'WF':
                n_solvent_atoms += 1
                convert_function = _convert_W
            else:
                raise NotImplementedError(
                    f'Only DMPC, DPPC, DOPC, DSPC, W, and WF supported, not '
                    f'{molecule_name}'
                )

            converted_line, v = convert_function(
                molecule_index,
                atom_name,
                atom_write,
                pos_x, pos_y, pos_z,
                vel_x, vel_y, vel_z,
                velocity=velocity,
                split_carbons=split_carbons,
                zero_velocity=zero_velocity,
                water_index=water_index
            )

            out_file.write(converted_line + '\n')

    with open(out_path, 'r+') as out_file:
        _ = out_file.readline()
        _ = out_file.readline()
        _ = out_file.readline()
        position_in_file = out_file.tell()
        out_file.seek(position_in_file)
        out_file.write(f'{n_molecules}')

    print(f'Progress: 100.0 % {" "*100}')
    print('\nAtom count:')
    print('---------------')
    print(f'solvent atoms:     {n_solvent_atoms}')
    print(f'lipid atoms:       {n_lipid_atoms}')
    print(f'total atoms:       {n_solvent_atoms+n_lipid_atoms}')
    print('\nMolecule count:')
    print('---------------')
    print(f'solvent molecules: {n_solvent_molecules}')
    print(f'lipid molecules:   {n_lipid_molecules}')
    print(f'total molecules:   {n_solvent_molecules+n_lipid_molecules}')
    print('\nFinal momentum:')
    print('---------------')
    with np.printoptions(suppress=True, precision=15):
        print(f'total momentum:   {72*v_total / n_atoms} amu nm/ps')
        print(f'solvent momentum: {72*v_solvent / n_solvent_atoms} amu nm/ps')
        print(f'lipid momentum:   {72*v_lipid / n_lipid_atoms} amu nm/ps\n')
    return out_path


if __name__ == '__main__':
    description = 'Convert .gro files to the OCCAM fort.5 format'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('file_name', type=str, help='input .gro file name')
    parser.add_argument('--out', type=str, default=None, dest='out_path',
                        metavar='file name', help='output fort.5 file name')
    parser.add_argument('-f', action='store_true', default=False, dest='force',
                        help='overwrite existing output file')
    parser.add_argument('--split-carbons', action='store_true', default=False,
                        dest='split_carbons',
                        help='split carbons into tail / head')
    parser.add_argument('--no-velocity', action='store_false', default=True,
                        dest='velocity',
                        help='dont include velocity in fort.5 file')
    parser.add_argument('--zero-velocity', action='store_true', default=False,
                        dest='zero_velocity',
                        help='write all velocities equal zero')
    parser.add_argument('--cancel-momentum-separately', action='store_true',
                        default=False, dest='cancel_momentum_separately',
                        help='cancel momentum of lipids/solvents separately')
    parser.add_argument('--water-index', type=int, default=5,
                        dest='water_index',
                        help='occam index to use for the solvent')

    args = parser.parse_args()

    if args.zero_velocity and not args.velocity:
        raise ValueError('The --no-velocity and --zero-velocity flags cannot '
                         'both be set.')
    if args.cancel_momentum_separately and not args.velocity:
        raise ValueError('The --cancel-momentum-separately and '
                         '--no-velocity flags cannot both be set.')
    if args.cancel_momentum_separately and args.zero_velocity:
        raise ValueError('The --cancel-momentum-separately and '
                         '--zero-velocity flags cannot both be set.')

    gro2fort5(args.file_name, args.out_path, overwrite=args.force,
              split_carbons=args.split_carbons, velocity=args.velocity,
              zero_velocity=args.zero_velocity,
              cancel_momentum_separately=args.cancel_momentum_separately,
              water_index=args.water_index)
