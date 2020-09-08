import numpy as np
import h5py
import pytest


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
