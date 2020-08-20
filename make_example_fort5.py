import numpy as np
import argparse
import os
from scipy.spatial.transform import Rotation


def write_fort5(box_size,
                n_polymers,
                n_solvent,
                polymer_topology=None,
                bonds=None,
                out_path=None,
                force=False):
    if n_polymers > 0:
        if polymer_topology is None and bonds is None:
            raise ValueError('polymer_topology and bonds must be specified '
                             'with n_polymers > 0')
        elif polymer_topology is None:
            raise ValueError(
                'polymer_topology must be specified with n_polymers > 0'
            )
        elif bonds is None:
            raise ValueError('bonds must be specified with n_polymers > 0')

    n_atoms_per_polymer = len(polymer_topology)
    n_molecules = n_polymers + n_solvent
    n_atoms = n_molecules * n_atoms_per_polymer + n_solvent

    if out_path is None:
        out_path = 'fort.5'
    if os.path.exists(out_path):
        if not force:
            raise FileExistsError(f'The file {out_path} already exists. Use '
                                  f'the -f flag to force overwrite.')

    rng = np.random.default_rng()

    atom_types = {}
    ind = 1
    for a in polymer_topology:
        if a not in atom_types:
            atom_types[a] = ind
            ind += 1
    atom_types['W'] = ind

    with open(out_path, 'w') as out_file:
        out_file.write('box:\n\t')
        for b in box_size:
            out_file.write(f'{b:.10f}\t')
        out_file.write('\n\t')
        out_file.write(f'{0.0:.10f}\n')
        out_file.write(f'n_molecules:\n\t{n_molecules}\n')

        for p in range(n_polymers):
            out_file.write(
                f'molecule_index:\t{p + 1}\n\t\t{n_atoms_per_polymer}\n'
            )
            position = np.array(
                [rng.uniform(high=box_size[i], size=1) for i in range(3)]
            ).T

            for a in range(n_atoms_per_polymer):
                R = Rotation.random()
                displacement = np.array([0.25, 0, 0])
                displacement = R.apply(displacement)
                position = position + displacement
                position = np.squeeze(position)

                for dim in range(len(position)):
                    while position[dim] > box_size[dim]:
                        position[dim] -= box_size[dim]
                    while position[dim] < 0.0:
                        position[dim] += box_size[dim]

                atom_type_name = polymer_topology[a]
                atom_type = atom_types[atom_type_name]
                atom_index = p * n_atoms_per_polymer + a + 1

                bonds_adjusted = [0 for _ in range(6)]
                num_bonds = len([1 for i in bonds[a] if i != -1])
                for i, b in enumerate(bonds[a]):
                    if b != -1:
                        bonds_adjusted[i] = b + p * n_atoms_per_polymer

                out_file.write(f'{atom_index:4}\t{atom_type_name:4}\t{atom_type:4}\t{num_bonds:4}')  # noqa: E501
                out_file.write(f'\t\t{position[0]:.16f}\t{position[1]:.16f}\t{position[2]:.16f}')  # noqa: E501
                out_file.write(f'\t{0.0:.3f}\t{0.0:.3f}\t{0.0:.3f}\t')
                for b in bonds_adjusted:
                    out_file.write(f'\t{b:4}')
                out_file.write('\n')

        for s in range(n_solvent):
            out_file.write(f'molecule_index:\t{n_polymers + s + 1}\n\t{1}\n')
            x, y, z = np.random.uniform(low=0.0, high=10.0, size=3)
            atom_type_name = 'W'
            atom_type = atom_types[atom_type_name]
            atom_index = n_polymers * n_atoms_per_polymer + s + 1
            bonds = 0
            bonds_indices = (0, 0, 0, 0, 0, 0)

            out_file.write(f'{atom_index:4}\t{atom_type_name:4}\t{atom_type:4}\t{bonds:4}')  # noqa: E501
            out_file.write(f'\t\t{x:.3f}\t{y:.3f}\t{z:.3f}')
            out_file.write(f'\t{0.0:.3f}\t{0.0:.3f}\t{0.0:.3f}\t')
            for b in bonds_indices:
                out_file.write(f'\t{b:4}')
            out_file.write('\n')

    if n_atoms < 1000:
        with open(out_path, 'r') as in_file:
            for i, line in enumerate(in_file):
                print(line.strip())


if __name__ == '__main__':
    description = 'Create example fort.5 files'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--box', type=int, nargs=3,
                        help='simulation box length')
    parser.add_argument('--n-molecules', type=int, dest='n_molecules',
                        help='number of molecules', default=0)
    parser.add_argument('--n-solvent', type=int, dest='n_solvent',
                        help='number of solvent atoms', default=0)
    parser.add_argument('--system', type=str,
                        help='what kind of system to generate',
                        choices=['diblock-copolymer'])
    parser.add_argument('--out', type=str, default=None, dest='out_path',
                        metavar='file name', help='output fort.5 file name')
    parser.add_argument('-f', action='store_true', dest='force',
                        help='overwrite existing output file')
    args = parser.parse_args()

    if args.n_molecules > 0 and args.system == 'diblock-copolymer':
        polymer_topology = ['A' for _ in range(10)] + ['B' for _ in range(10)]
        bonds = np.empty(shape=(20, 2), dtype=int)
        bonds[0, :] = np.array([2, -1])
        bonds[-1, :] = np.array([19, -1])
        bonds[1:-1, 0] = np.arange(1, 19, 1, dtype=int)
        bonds[1:-1, 1] = np.arange(3, 21, 1, dtype=int)
    else:
        raise NotImplementedError('only -diblock-copolymer supported')

    write_fort5(args.box, args.n_molecules, args.n_solvent,
                polymer_topology=polymer_topology, bonds=bonds,
                out_path=args.out_path, force=args.force)
