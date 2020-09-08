import h5py
import numpy as np


def distribute_input(input_path, rank, size, n_particles, driver='mpio'):
    np_per_MPI = n_particles // size

    with h5py.File(input_path, 'r', driver=driver) as in_file:
        molecules_flag = False
        if 'molecules' in in_file:
            molecules_flag = True

        if not molecules_flag:
            if rank == size - 1:
                np_cum_mpi = [rank * np_per_MPI, n_particles]
            else:
                np_cum_mpi = [rank * np_per_MPI, (rank + 1) * np_per_MPI]
            p_mpi_range = list(range(np_cum_mpi[0], np_cum_mpi[1]))
            return p_mpi_range, molecules_flag

        # To avoid splitting molecules across multiple different ranks, we need
        # to read in some extra indices before/after the expected break points
        # and iterate until we find a molecule break.
        #
        # Implicitly assuming no molecule is bigger than
        # min(201, n_particles // n_MPI_ranks) atoms.
        grab_extra = 201 if np_per_MPI > 201 else np_per_MPI
        if rank == 0:
            mpi_range_start = 0
            if size == 1:
                mpi_range_end = n_particles
            else:
                mpi_range_end = (rank + 1) * np_per_MPI + grab_extra
        elif rank == size - 1:
            mpi_range_start = rank * np_per_MPI - 1
            mpi_range_end = n_particles
        else:
            mpi_range_start = rank * np_per_MPI - 1
            mpi_range_end = (rank + 1) * np_per_MPI + grab_extra

        molecules = in_file['molecules'][mpi_range_start:mpi_range_end]
        indices = in_file['indices'][mpi_range_start:mpi_range_end]
        molecule_end_indices = np.nonzero(np.diff(molecules))[0]

        p_mpi_range = [None, None]
        if rank == 0:
            p_mpi_range[0] = 0
            if size == 1:
                p_mpi_range[1] = n_particles
            else:
                p_mpi_range[1] = indices[
                    molecule_end_indices[
                        molecule_end_indices > np_per_MPI
                    ][0] + 1
                ]
        elif rank == size - 1:
            p_mpi_range[0] = indices[
                molecule_end_indices[molecule_end_indices > 0][0]
            ] + 1
            p_mpi_range[1] = n_particles
        else:
            p_mpi_range[0] = indices[
                molecule_end_indices[molecule_end_indices > 0][0]
            ] + 1
            p_mpi_range[1] = indices[
                molecule_end_indices[molecule_end_indices > np_per_MPI][0]
            ] + 1
        return list(range(p_mpi_range[0], p_mpi_range[1])), molecules_flag
