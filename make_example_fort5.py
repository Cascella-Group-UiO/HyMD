import numpy as np


box_size = (10, 10, 10)
n_polymers = 5
n_atoms_per_polymer = 5
n_solvent = 500
n_molecules = n_polymers + n_solvent
n_atoms = n_polymers * n_atoms_per_polymer + n_solvent


with open('fort.5', 'w') as out_file:
    out_file.write('box:\n\t')
    for b in box_size:
        out_file.write(f'{b:.10f}\t')
    out_file.write('\n\t')
    out_file.write(f'{0.0:.10f}\n')
    out_file.write(f'n_molecules:\n\t{n_molecules}\n')

    for p in range(n_polymers):
        out_file.write(f'molecule_index:\t{p + 1}\n\t\t{n_atoms_per_polymer}\n')
        position_xy = 2.0 + p

        z_start = np.random.uniform(low=0.0, high=7.0, size=1)[0]
        for a in range(n_atoms_per_polymer):
            x, y = position_xy, position_xy
            z = z_start + 0.5 * a
            atom_type_name = 'B'
            atom_type = 2
            atom_index = p * n_atoms_per_polymer + a + 1
            if (a == 0) or (a == n_atoms_per_polymer - 1):
                bonds = 1
                bonded_index = atom_index + 1 if a == 0 else atom_index - 1
                bonds_indices = (bonded_index, 0, 0, 0, 0, 0)
            else:
                bonds = 2
                bonds_indices = (atom_index - 1, atom_index + 1, 0, 0, 0, 0)

            out_file.write(f'{atom_index:4}\t{atom_type_name:4}\t{atom_index:4}\t{bonds:4}')  # noqa: E501
            out_file.write(f'\t\t{x:.3f}\t{y:.3f}\t{z:.3f}')
            out_file.write(f'\t{0.0:.3f}\t{0.0:.3f}\t{0.0:.3f}\t')
            for b in bonds_indices:
                out_file.write(f'\t{b:4}')
            out_file.write('\n')

    for s in range(n_solvent):
        out_file.write(f'molecule_index:\t{n_polymers + s + 1}\n\t{1}\n')
        x, y, z = np.random.uniform(low=0.0, high=10.0, size=3)
        atom_type_name = 'A'
        atom_type = 1
        atom_index = n_polymers * n_atoms_per_polymer + s + 1
        bonds = 0
        bonds_indices = (0, 0, 0, 0, 0, 0)

        out_file.write(f'{atom_index:4}\t{atom_type_name:4}\t{atom_index:4}\t{bonds:4}')  # noqa: E501
        out_file.write(f'\t\t{x:.3f}\t{y:.3f}\t{z:.3f}')
        out_file.write(f'\t{0.0:.3f}\t{0.0:.3f}\t{0.0:.3f}\t')
        for b in bonds_indices:
            out_file.write(f'\t{b:4}')
        out_file.write('\n')

with open('fort.5', 'r') as in_file:
    for i, line in enumerate(in_file):
        print(line.strip())
