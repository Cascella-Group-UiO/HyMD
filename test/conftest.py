import numpy as np
import h5py
import pytest
import collections


@pytest.fixture
def dppc_single():
    """
    Sets up a single DPPC molecule test system

    Notes
    -----
    Type names (indices) and bonds::

                      G(3) -- C(8) -- C(9) -- C(10) -- C(11)
                      /
    N(0) -- P(1) -- G(2) -- C(4) -- C(5) -- C(6) -- C(7)
    """
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=int)
    molecules = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    r = np.array([[0.244559E+01, 0.909193E+00, 0.560020E+01],
                  [0.206399E+01, 0.745504E+00, 0.577413E+01],
                  [0.172144E+01, 0.618203E+00, 0.601021E+01],
                  [0.164460E+01, 0.705806E+00, 0.629945E+01],
                  [0.151964E+01, 0.940714E+00, 0.646405E+01],
                  [0.154557E+01, 0.108941E+01, 0.685593E+01],
                  [0.161312E+01, 0.103298E+01, 0.730153E+01],
                  [0.205084E+01, 0.118547E+01, 0.763296E+01],
                  [0.214117E+01, 0.868565E+00, 0.650513E+01],
                  [0.229689E+01, 0.108656E+01, 0.691872E+01],
                  [0.258356E+01, 0.119129E+01, 0.731768E+01],
                  [0.287958E+01, 0.148139E+01, 0.755405E+01]],
                 dtype=np.float64)
    bonds = np.array([[1,           -1, -1],   # N(0)
                      [0, 2,        -1],       # P(1)
                      [1, 3,  4],              # G(2) -- C(4)
                      [2, 8,        -1],       # G(3) -- C(8)
                      [2, 5,        -1],       # C(4)
                      [4, 6,        -1],       # C(5)
                      [5, 7,        -1],       # C(6)
                      [6,           -1, -1],   # C(7)
                      [3, 9,        -1],       # C(8)
                      [8, 10,       -1],       # C(9)
                      [9, 11,       -1],       # C(10)
                      [10,          -1, -1]],  # C(11)
                     dtype=int)
    names = np.array([b'N', b'P', b'G', b'G', b'C', b'C', b'C', b'C', b'C',
                      b'C', b'C', b'C'], dtype='S5')
    CONF = {}
    Bond = collections.namedtuple(
        'Bond', ['atom_1', 'atom_2', 'equilibrium', 'strenght']
    )
    Angle = collections.namedtuple(
        'Angle', ['atom_1', 'atom_2', 'atom_3', 'equilibrium', 'strenght']
    )
    CONF['bond_2'] = (Bond('N', 'P', 0.47, 1250.0),
                      Bond('P', 'G', 0.47, 1250.0),
                      Bond('G', 'G', 0.37, 1250.0),
                      Bond('G', 'C', 0.47, 1250.0),
                      Bond('C', 'C', 0.47, 1250.0))

    CONF['bond_3'] = (Angle('P', 'G', 'G', 120.0, 25.0),
                      Angle('P', 'G', 'C', 180.0, 25.0),
                      Angle('G', 'C', 'C', 180.0, 25.0),
                      Angle('C', 'C', 'C', 180.0, 25.0))
    for k, v in {'Np': 12, 'types': 5, 'mass': 72.0,
                 'L': [13.0, 13.0, 14.0]}.items():
        CONF[k] = v
    return indices, bonds, names, molecules, r, CONF


@pytest.fixture(scope='session')
def h5py_molecules_file(tmp_path_factory):
    n_particles = 1000
    out_path = tmp_path_factory.mktemp('test').joinpath('mols.hdf5')
    indices = np.empty(1000, dtype=int)
    molecules = np.empty(1000, dtype=int)

    with h5py.File(out_path, 'w') as out_file:
        mol_len = np.array([21, 34, 18, 23, 19, 11, 18, 24, 13, 19, 27, 11, 31,
                            14, 37, 30, 38, 24, 16,  5, 30, 25, 19,  5, 31, 14,
                            21, 15, 13, 27, 13, 12,  8,  2, 15, 31, 13, 31, 20,
                            11,  7, 22,  3, 31,  4, 24, 30,  4, 36,  5,  1,  1,
                            1,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                            1,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                            1,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                            1,   1,  1,  1], dtype=int)
        indices = np.arange(n_particles)
        current_ind = 0
        for ind, mol in enumerate(mol_len):
            next_ind = current_ind + mol
            molecules[current_ind:next_ind].fill(ind)
            current_ind = next_ind

        dset_indices = out_file.create_dataset('indices', (n_particles,),
                                               dtype='i')
        dset_molecules = out_file.create_dataset('molecules', (n_particles,),
                                                 dtype='i')
        dset_indices[:] = indices[:]
        dset_molecules[:] = molecules[:]
    return out_path, n_particles, indices, molecules
