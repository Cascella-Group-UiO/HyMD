import numpy as np
import h5py
from hymd.file_io import distribute_input


def _distribute_nranks(size, in_path, n_particles):
    distr = [None] * size
    with h5py.File(in_path, 'r', driver=None) as in_file:
        for rank in range(size):
            rank_range, molecules_flag = distribute_input(
                in_file, rank, size, n_particles
            )
            distr[rank] = (rank_range, molecules_flag)
    return distr


def test_file_io_distribute_input_size_1(h5py_molecules_file):
    in_path, n_particles, _, _ = h5py_molecules_file
    size = 1
    distr = _distribute_nranks(size, in_path, n_particles)
    rank_range, molecules_flag = distr[0]
    assert molecules_flag
    assert rank_range[0] == 0
    assert rank_range[-1] == n_particles - 1
    assert len(rank_range) == n_particles


def test_file_io_distribute_input_size_2(h5py_molecules_file):
    in_path, n_particles, indices, molecules = h5py_molecules_file
    size = 2
    distr = _distribute_nranks(size, in_path, n_particles)
    rank_range_0, molecules_flag_0 = distr[0]
    rank_range_1, molecules_flag_1 = distr[1]

    assert molecules_flag_0
    assert molecules_flag_1

    assert rank_range_0[0] == 0
    assert molecules[rank_range_0[-1]] != molecules[rank_range_0[-1] + 1]

    assert molecules[rank_range_1[0] - 1] != molecules[rank_range_1[0]]
    assert rank_range_1[-1] == n_particles - 1

    assert rank_range_1[0] == rank_range_0[-1] + 1
    rank_range_all = np.concatenate((rank_range_0, rank_range_1))
    assert np.equal(rank_range_all, np.arange(n_particles)).all()


def test_file_io_distribute_input_size_3(h5py_molecules_file):
    in_path, n_particles, indices, molecules = h5py_molecules_file
    size = 3
    distr = _distribute_nranks(size, in_path, n_particles)
    rank_range_0, molecules_flag_0 = distr[0]
    rank_range_1, molecules_flag_1 = distr[1]
    rank_range_2, molecules_flag_2 = distr[2]

    assert molecules_flag_0
    assert molecules_flag_1
    assert molecules_flag_2

    assert rank_range_0[0] == 0
    assert molecules[rank_range_0[-1]] != molecules[rank_range_0[-1] + 1]

    assert molecules[rank_range_1[0] - 1] != molecules[rank_range_1[0]]
    assert molecules[rank_range_1[-1]] != molecules[rank_range_1[-1] + 1]

    assert molecules[rank_range_2[0] - 1] != molecules[rank_range_2[0]]
    assert rank_range_2[-1] == n_particles - 1

    rank_range_all = np.concatenate((rank_range_0, rank_range_1, rank_range_2))
    assert np.equal(rank_range_all, np.arange(n_particles)).all()


def test_file_io_distribute_input_various_sizes(h5py_molecules_file):
    in_path, n_particles, indices, molecules = h5py_molecules_file
    for size in (5, 9, 11, 14, 19, 25):
        distr = _distribute_nranks(size, in_path, n_particles)

        assert distr[0][0][0] == 0
        assert distr[-1][0][-1] == n_particles - 1

        for i, d in enumerate(distr):
            molecules_flag = d[1]
            assert molecules_flag

            rank_range = d[0]
            if i == 0:
                assert rank_range[0] == 0
            else:
                assert molecules[rank_range[0] - 1] != molecules[rank_range[0]]

            if i == size - 1:
                assert rank_range[-1] == n_particles - 1
            else:
                assert (molecules[rank_range[-1]] !=
                        molecules[rank_range[-1] + 1])

        rank_range_all = np.concatenate([d[0] for d in distr])
        assert np.equal(rank_range_all, np.arange(n_particles)).all()
