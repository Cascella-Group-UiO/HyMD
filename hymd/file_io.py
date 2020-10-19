import numpy as np
import h5py
import os
import logging
from mpi4py import MPI
from logger import Logger


class OutDataset:
    def __init__(self, dest_directory, config, disable_mpio=False,
                 comm=MPI.COMM_WORLD):
        self.disable_mpio = disable_mpio
        self.config = config

        if disable_mpio:
            self.file = h5py.File(
                os.path.join(dest_directory,
                             f'sim.hdf5-{comm.rank:6d}-of-{comm.size:6d}'),
                'w'
            )
        else:
            self.file = h5py.File(
                os.path.join(dest_directory,
                             'sim.hdf5'),
                'w', driver='mpio', comm=comm
            )
        self.define_datasets(config.n_particles,
                             config.n_steps // config.n_print + 1)

    def close_file(self, comm=MPI.COMM_WORLD):
        comm.Barrier()
        self.file.close()

    def define_datasets(self, n_particles, n_frames):
        self.position = self.file.create_dataset(
            'coordinates', (n_frames, n_particles, 3), dtype="Float32")
        self.velocity = self.file.create_dataset(
            'velocities', (n_frames, n_particles, 3), dtype="Float32")
        self.time = self.file.create_dataset(
            'time', shape=(n_frames,), dtype="Float32")
        self.box_size = self.file.create_dataset(
            'cell_lengths', (n_frames, 3, 3), dtype='Float32')
        self.box_angles = self.file.create_dataset(
            'cell_angles', (n_frames, 3, 3), dtype='Float32')
        self.temperature = self.file.create_dataset(
            'temperature', (n_frames,), dtype='Float32')
        self.names = self.file.create_dataset(
            'names', (n_particles,), dtype="S10")
        self.indices = self.file.create_dataset(
            'indices', (n_particles,), dtype='i')
        self.types = self.file.create_dataset(
            'types', (n_particles,), dtype='i')
        self.total_momentum = self.file.create_dataset(
            'total_momentum', (n_frames, 3), dtype="Float32")
        self.total_energy = self.file.create_dataset(
            'totalEnergy', (n_frames,), dtype='Float32')
        self.potential_energy = self.file.create_dataset(
            'potentialEnergy', (n_frames,), dtype='Float32')
        self.kinetic_energy = self.file.create_dataset(
            'kineticEnergy', (n_frames,), dtype='Float32')
        self.bond2_energy = self.file.create_dataset(
            'bond2Energy', (n_frames,), dtype='Float32')
        self.bond3_energy = self.file.create_dataset(
            'bond3Energy', (n_frames,), dtype='Float32')
        self.field_energy = self.file.create_dataset(
            'fieldEnergy', (n_frames,), dtype='Float32')

        self.position.attrs['units'] = 'nanometers'
        self.velocity.attrs['units'] = 'nanometers/picosecond'
        self.time.attrs['units'] = 'picoseconds'
        self.box_size.attrs['units'] = 'nanometers'
        self.box_angles.attrs['units'] = 'degrees'
        self.temperature.attrs['units'] = 'Kelvin'
        self.total_momentum.attrs['units'] = 'nanometers g/picosecond mol'
        for e in (self.total_energy, self.potential_energy, self.bond2_energy,
                  self.bond3_energy, self.field_energy):
            e.attrs['units'] = 'kJ/mol'


def store_static(out_dataset, rank_range, names, types, indices, box_size,
                 comm=MPI.COMM_WORLD):
    # FIXME: this can be inefficient if p_mpi_range is discontiguous (depends
    # on hdf-mpi impl detail)
    out_dataset.names[rank_range] = names
    out_dataset.types[rank_range] = types
    out_dataset.indices[rank_range] = indices
    if comm.rank == 0:
        out_dataset.box_size[0, 0, 0] = box_size[0]
        out_dataset.box_size[0, 1, 1] = box_size[1]
        out_dataset.box_size[0, 2, 2] = box_size[2]
        out_dataset.box_angles[0, :] = np.full((3, 3), 90.0, dtype='Float32')


def store_data(out_dataset, step, frame, indices, positions, velocities,
               box_size, temperature, kinetic_energy, bond2_energy,
               bond3_energy, field_energy, time_step, config,
               comm=MPI.COMM_WORLD):
    ind_sort = np.argsort(indices)
    out_dataset.position[frame, indices[ind_sort]] = positions[ind_sort]
    out_dataset.velocity[frame, indices[ind_sort]] = velocities[ind_sort]

    total_momentum = config.mass * comm.allreduce(np.sum(velocities, axis=0),
                                                  MPI.SUM)
    assert len(total_momentum) == 3                                             ############################## <<<< CHECK-ME
    if comm.rank == 0:
        out_dataset.time[frame] = time_step * step
        out_dataset.temperature[frame] = temperature
        out_dataset.total_momentum[frame, :] = total_momentum

        out_dataset.box_size[0, 0, 0] = box_size[0]
        out_dataset.box_size[0, 1, 1] = box_size[1]
        out_dataset.box_size[0, 2, 2] = box_size[2]

        potential_energy = bond2_energy + bond3_energy + field_energy
        out_dataset.bond2_energy[frame] = bond2_energy
        out_dataset.bond3_energy[frame] = bond3_energy
        out_dataset.field_energy[frame] = field_energy
        out_dataset.kinetic_energy[frame] = kinetic_energy
        out_dataset.potential_energy[frame] = potential_energy
        out_dataset.total_energy[frame] = kinetic_energy + potential_energy

        header = (
            '{:15} {:15} {:15} {:15} {:15} {:15} {:15} {:15} {:15} {:15} {:15} {:15}'  # noqa: E501
            .format("step",
                    "time",
                    "temperature",
                    "total E",
                    "kinetic E",
                    "potential E",
                    "field E",
                    "bond E",
                    "angle E",
                    "total Px",
                    "total Py",
                    "total Pz"))
        data_fmt = f'{"{:15}"} {11 * "{:15.10g}" }'
        data = data_fmt.format(step,
                               time_step * step,
                               temperature,
                               kinetic_energy + potential_energy,
                               kinetic_energy,
                               potential_energy,
                               field_energy,
                               bond2_energy,
                               bond3_energy,
                               total_momentum[0],
                               total_momentum[1],
                               total_momentum[2])
        Logger.rank0.log(
            logging.INFO, ('\n' + header + '\n' + data)
        )


def distribute_input(in_file, rank, size, n_particles, max_molecule_size=201):
    np_per_MPI = n_particles // size

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
    grab_extra = (
        max_molecule_size if np_per_MPI > max_molecule_size else np_per_MPI
    )
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
