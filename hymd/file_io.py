"""Handle file input/output in parllel HDF5 fashion
"""
import numpy as np
import h5py
import os
import logging
import getpass
from mpi4py import MPI
from .logger import Logger, get_version


class OutDataset:
    """HDF5 dataset handler for file output
    """

    def __init__(
        self, dest_directory, config, double_out=False, disable_mpio=False,
        comm=MPI.COMM_WORLD,
    ):
        """Constructor

        Parameters
        ----------
        dest_directory : str
            Ouput directory for saving data.
        config : Config
            Configuration object.
        double_out : bool, optional
            If :code:`True`, the output HDF5 objects are written in eight byte
            floating point representation. Otherwise, four byte single
            precision is used.
        disable_mpio : bool, optional
            If :code:`True`, disables parallel MPI-enabled HDF5 file output.
            This is a compatibility option used if HDF5 is not compiled with
            MPI support. This makes everything much harder by splitting all
            output in one file per MPI rank, and having an MPI-enabled HDF5
            library is **highly recommended**.
        comm : mpi4py.Comm
            MPI communicator to use for rank commuication.
        """
        self.disable_mpio = disable_mpio
        self.config = config
        if double_out:
            self.float_dtype = "float64"
        else:
            self.float_dtype = "float32"

        if disable_mpio:
            self.file = h5py.File(
                os.path.join(
                    dest_directory,
                    f"sim.hdf5-{comm.rank:6d}-of-{comm.size:6d}"
                ),
                "w",
            )
        else:
            self.file = h5py.File(
                os.path.join(dest_directory, "sim.H5"), "w", driver="mpio",
                comm=comm
            )

    def close_file(self, comm=MPI.COMM_WORLD):
        """Closes the HDF5 output file

        Parameters
        ----------
        comm : mpi4py.Comm
            MPI communicator to use for rank commuication.
        """
        comm.Barrier()
        self.file.close()

    def flush(self):
        """Flushes output buffers, forcing file writes
        """
        self.file.flush()


def setup_time_dependent_element(
    name, parent_group, n_frames, shape, dtype, units=None
):
    """Helper function for setting up time-dependent HDF5 group datasets

    All output groups must adhere to the H5MD standard, meaning a structure of


    |    ┗━ **group** particle group (e.g. :code:`all`) or :code:`observables` group
    |        ┗━ **group** time-dependent data
    |            ┣━ **dataset** :code:`step` :code:`shape=(n_frames,)`
    |            ┣━ **dataset** :code:`time` :code:`shape=(n_frames,)`
    |            ┗━ **dataset** :code:`value` :code:`shape=(n_frames, *)`

    is necessary.

    References
    ----------
    H5MD specification :
        https://www.nongnu.org/h5md/h5md.html
    """  # noqa: E501
    group = parent_group.create_group(name)
    step = group.create_dataset("step", (n_frames,), "int32")
    time = group.create_dataset("time", (n_frames,), "float32")
    value = group.create_dataset("value", (n_frames, *shape), dtype)
    if units is not None:
        group.attrs["units"] = units
    return group, step, time, value


def store_static(
    h5md, rank_range, names, types, indices, config, bonds_2_atom1,
    bonds_2_atom2, molecules=None, velocity_out=False, force_out=False,
    charges=False, comm=MPI.COMM_WORLD,
):
    """Outputs all static time-independent quantities to the HDF5 output file

    Parameters
    ----------
    h5md : OutDataset
        HDF5 dataset handler.
    rank_range : list[int]
        Start and end indices for global arrays for each MPI rank.
    names : (N,) numpy.ndarray
        Array of names for :code:`N` particles.
    types : (N,) numpy.ndarray
        Array of type indices for :code:`N` particles.
    indices : (N,) numpy.ndarray
        Array of indices for :code:`N` particles.
    config : Config
        Configuration object.
    bonds_2_atom1 : (B,) numpy.ndarray
        Array of indices of the first particle for :code:`B` total
        two-particle bonds.
    bonds_2_atom2 : (B,) numpy.ndarray
        Array of indices of the second particle for :code:`B` total
        two-particle bonds.
    molecules : (N,) numpy.ndarray, optional
        Array of integer molecule affiliation for each of :code:`N` particles.
        Global (across all MPI ranks) or local (local indices on this MPI rank
        only) may be used, both, without affecting the result.
    velocity_out : bool, optional
        If :code:`True`, velocities are written to output HDF5 file.
    force_out : bool, optional
        If :code:`True`, forces are written to output HDF5 file.
    charges : (N,) numpy.ndarray
        Array of particle charge values for :code:`N` particles.
    comm : mpi4py.Comm
        MPI communicator to use for rank commuication.

    See also
    --------
    prepare_bonds :
        Constructs two-, three-, and four-particle bonds from toplogy input
        file and bond configuration information.
    distribute_input :
        Distributes input arrays onto MPI ranks, attempting load balancing.
    """
    dtype = h5md.float_dtype

    h5md_group = h5md.file.create_group("/h5md")
    h5md.h5md_group = h5md_group
    h5md.observables = h5md.file.create_group("/observables")
    h5md.connectivity = h5md.file.create_group("/connectivity")
    h5md.parameters = h5md.file.create_group("/parameters")

    h5md_group.attrs["version"] = np.array([1, 1], dtype=int)
    author_group = h5md_group.create_group("author")
    author_group.attrs["name"] = np.string_(getpass.getuser())
    creator_group = h5md_group.create_group("creator")
    creator_group.attrs["name"] = np.string_("Hylleraas MD")

    # Get HyMD version. Also grab the user email from git config if we
    # can find it.
    creator_group.attrs["version"] = np.string_(get_version())
    try:
        import git

        try:
            reader = repo.config_reader()
            user_email = reader.get_value("user", "email")
            author_group.attrs["email"] = np.string_(user_email)
        except:
            pass
    except:
        pass

    h5md.particles_group = h5md.file.create_group("/particles")
    h5md.all_particles = h5md.particles_group.create_group("all")
    mass = h5md.all_particles.create_dataset(
        "mass", (config.n_particles,), dtype
    )
    mass[...] = config.mass

    if charges is not False:
        charge = h5md.all_particles.create_dataset(
            "charge", (config.n_particles,), dtype="float32"
        )
        charge[indices] = charges

    box = h5md.all_particles.create_group("box")
    box.attrs["dimension"] = 3
    box.attrs["boundary"] = np.array(
        [np.string_(s) for s in 3 * ["periodic"]], dtype="S8"
    )
    h5md.edges = box.create_dataset("edges", (3,), dtype)
    h5md.edges[:] = np.array(config.box_size)

    n_frames = config.n_steps // config.n_print
    if np.mod(config.n_steps - 1, config.n_print) != 0:
        n_frames += 1
    if np.mod(config.n_steps, config.n_print) == 1:
        n_frames += 1

    species = h5md.all_particles.create_dataset(
        "species", (config.n_particles,), dtype="i",
    )

    (
        _,
        h5md.positions_step,
        h5md.positions_time,
        h5md.positions,
    ) = setup_time_dependent_element(
        "position",
        h5md.all_particles,
        n_frames,
        (config.n_particles, 3),
        dtype,
        units="nm",
    )
    if velocity_out:
        (
            _,
            h5md.velocities_step,
            h5md.velocities_time,
            h5md.velocities,
        ) = setup_time_dependent_element(
            "velocity",
            h5md.all_particles,
            n_frames,
            (config.n_particles, 3),
            dtype,
            units="nm ps-1",
        )
    if force_out:
        (
            _,
            h5md.forces_step,
            h5md.forces_time,
            h5md.forces,
        ) = setup_time_dependent_element(
            "force",
            h5md.all_particles,
            n_frames,
            (config.n_particles, 3),
            dtype,
            units="kJ nm mol-1",
        )
    (
        _,
        h5md.total_energy_step,
        h5md.total_energy_time,
        h5md.total_energy,
    ) = setup_time_dependent_element(
        "total_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"  # noqa: E501
    )
    (
        _,
        h5md.kinetc_energy_step,
        h5md.kinetc_energy_time,
        h5md.kinetc_energy,
    ) = setup_time_dependent_element(
        "kinetic_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"  # noqa: E501
    )
    (
        _,
        h5md.potential_energy_step,
        h5md.potential_energy_time,
        h5md.potential_energy,
    ) = setup_time_dependent_element(
        "potential_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"  # noqa: E501
    )
    (
        _,
        h5md.bond_energy_step,
        h5md.bond_energy_time,
        h5md.bond_energy,
    ) = setup_time_dependent_element(
        "bond_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"  # noqa: E501
    )
    (
        _,
        h5md.angle_energy_step,
        h5md.angle_energy_time,
        h5md.angle_energy,
    ) = setup_time_dependent_element(
        "angle_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"  # noqa: E501
    )
    (
        _,
        h5md.dihedral_energy_step,
        h5md.dihedral_energy_time,
        h5md.dihedral_energy,
    ) = setup_time_dependent_element(
        "dihedral_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"  # noqa: E501
    )
    (
        _,
        h5md.field_energy_step,
        h5md.field_energy_time,
        h5md.field_energy,
    ) = setup_time_dependent_element(
        "field_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"  # noqa: E501
    )
    if charges is not False:
        (
            _,
            h5md.field_q_energy_step,
            h5md.field_q_energy_time,
            h5md.field_q_energy,
        ) = setup_time_dependent_element(
            "field_q_energy", h5md.observables, n_frames, (1,), dtype, units="kJ mol-1"  # noqa: E501
        )  # <-------- xinmeng
    (
        _,
        h5md.total_momentum_step,
        h5md.total_momentum_time,
        h5md.total_momentum,
    ) = setup_time_dependent_element(  # noqa: E501
        "total_momentum",
        h5md.observables,
        n_frames,
        (3,),
        dtype,
        units="nm g ps-1 mol-1",
    )
    (
        _,
        h5md.angular_momentum_step,
        h5md.angular_momentum_time,
        h5md.angular_momentum,
    ) = setup_time_dependent_element(  # noqa: E501
        "angular_momentum",
        h5md.observables,
        n_frames,
        (3,),
        dtype,
        units="nm+2 g ps-1 mol-1",
    )
    (
        _,
        h5md.torque_step,
        h5md.torque_time,
        h5md.torque,
    ) = setup_time_dependent_element(  # noqa: E501
        "torque",
        h5md.observables,
        n_frames,
        (3,),
        dtype,
        units="kJ nm+2 mol-1",
    )
    (
        _,
        h5md.temperature_step,
        h5md.temperature_time,
        h5md.temperature,
    ) = setup_time_dependent_element(
        "temperature", h5md.observables, n_frames, (3,), dtype, units="K"
    )
    (
        _,
        h5md.thermostat_work_step,
        h5md.thermostat_work_time,
        h5md.thermostat_work,
    ) = setup_time_dependent_element(
        "thermostat_work", h5md.observables, n_frames, (1,), "float32", units="kJ mol-1"  # noqa: E501
    )

    ind_sort = np.argsort(indices)
    for i in ind_sort:
        species[indices[i]] = config.name_to_type_map[names[i].decode("utf-8")]

    h5md.parameters.attrs["config.toml"] = np.string_(str(config))
    vmd_group = h5md.parameters.create_group("vmd_structure")
    index_of_species = vmd_group.create_dataset(
        "indexOfSpecies", (config.n_types,), "i"
    )
    index_of_species[:] = np.array(list(range(config.n_types)))

    # VMD-h5mdplugin maximum name/type name length is 16 characters (for
    # whatever reason [VMD internals?]).
    name_dataset = vmd_group.create_dataset("name", (config.n_types,), "S16")
    type_dataset = vmd_group.create_dataset("type", (config.n_types,), "S16")
    if molecules is not None:
        resid_dataset = vmd_group.create_dataset(
            "resid", (config.n_particles,), "i",
        )

    # Change this
    for i, n in config.type_to_name_map.items():
        name_dataset[i] = np.string_(n[:16])
        if n == "W":
            type_dataset[i] = np.string_("solvent")
        else:
            type_dataset[i] = np.string_("membrane")

    total_bonds = comm.allreduce(len(bonds_2_atom1), MPI.SUM)
    n_bonds_local = len(bonds_2_atom1)

    receive_buffer = comm.gather(n_bonds_local, root=0)
    n_bonds_global = None
    if comm.Get_rank() == 0:
        n_bonds_global = receive_buffer
    n_bonds_global = np.array(comm.bcast(n_bonds_global, root=0))
    rank_bond_start = np.sum(n_bonds_global[: comm.Get_rank()])

    bonds_from = vmd_group.create_dataset("bond_from", (total_bonds,), "i")
    bonds_to = vmd_group.create_dataset("bond_to", (total_bonds,), "i")
    for i in range(n_bonds_local):
        a = bonds_2_atom1[i]
        b = bonds_2_atom2[i]
        bonds_from[rank_bond_start + i] = indices[a] + 1
        bonds_to[rank_bond_start + i] = indices[b] + 1

    if molecules is not None:
        resid_dataset[indices[ind_sort]] = molecules


def store_data(
    h5md, step, frame, indices, positions, velocities, forces, box_size,
    temperature, kinetic_energy, bond2_energy, bond3_energy, bond4_energy,
    field_energy, field_q_energy, time_step, config, velocity_out=False,
    force_out=False, charge_out=False, dump_per_particle=False,
    comm=MPI.COMM_WORLD,
):
    """Writes time-step data to HDF5 output file

    Handles all quantities which change during simulation, as opposed to
    static quanitities (see :code:`store_static`).

    Parameters
    ----------
    h5md : OutDataset
        HDF5 dataset handler.
    step : int
        Step number.
    frame : int
        Output frame number (:code:`step // n_print`).
    indices : (N,) numpy.ndarray
        Array of indices for :code:`N` particles.
    positions : (N,) numpy.ndarray
        Array of positions for :code:`N` particles in :code:`D` dimensions.
    velocities : (N,) numpy.ndarray
        Array of velocities for :code:`N` particles in :code:`D` dimensions.
    forces : (N,) numpy.ndarray
        Array of forces for :code:`N` particles in :code:`D` dimensions.
    box_size : (D,) numpy.ndarray
        Array containing the simulation box size.
    temperature : float
        Calculated instantaneous temperature.
    kinetic_energy : float
        Calculated instantaneous kinetic energy.
    bond2_energy : float
        Calculated instantaneous harmonic two-particle bond energy.
    bond3_energy : float
        Calculated instantaneous harmonic angular three-particle bond energy.
    bond4_energy : float
        Calculated instantaneous dihedral four-particle torsion energy.
    field_energy : float
        Calculated instantaneous particle-field energy.
    field_energy_q : float
        Calculated instantaneous electrostatic energy.
    time_step : float
        Value of the time step.
    config : Config
        Configuration object.
    velocity_out : bool, optional
        If :code:`True`, velocities are written to output HDF5 file.
    force_out : bool, optional
        If :code:`True`, forces are written to output HDF5 file.
    charge_out : bool, optional
        If :code:`True`, electrostatic energies are written to the output
        HDF5 file.
    dump_per_particle : bool, optional
        If :code:`True`, all quantities are written **per particle**.
    comm : mpi4py.Comm
        MPI communicator to use for rank commuication.

    See also
    --------
    store_static :
        Outputs all static time-independent quantities to the HDF5 output file
    """
    for dset in (
        h5md.positions_step,
        h5md.total_energy_step,
        h5md.potential_energy_step,
        h5md.kinetc_energy_step,
        h5md.bond_energy_step,
        h5md.angle_energy_step,
        h5md.dihedral_energy_step,
        h5md.field_energy_step,
        h5md.total_momentum_step,
        h5md.angular_momentum_step,
        h5md.torque_step,
        h5md.temperature_step,
        h5md.thermostat_work_step,
    ):
        dset[frame] = step

    for dset in (
        h5md.positions_time,
        h5md.total_energy_time,
        h5md.potential_energy_time,
        h5md.kinetc_energy_time,
        h5md.bond_energy_time,
        h5md.angle_energy_time,
        h5md.dihedral_energy_time,
        h5md.field_energy_time,
        h5md.total_momentum_time,
        h5md.angular_momentum_time,
        h5md.torque_time,
        h5md.temperature_time,
        h5md.thermostat_work_time,
    ):
        dset[frame] = step * time_step

    if velocity_out:
        h5md.velocities_step[frame] = step
        h5md.velocities_time[frame] = step * time_step
    if force_out:
        h5md.forces_step[frame] = step
        h5md.forces_time[frame] = step * time_step
    if charge_out:
        h5md.field_q_energy_step[frame] = step
        h5md.field_q_energy_time[frame] = step * time_step

    ind_sort = np.argsort(indices)
    h5md.positions[frame, indices[ind_sort]] = positions[ind_sort]

    if velocity_out:
        h5md.velocities[frame, indices[ind_sort]] = velocities[ind_sort]
    if force_out:
        h5md.forces[frame, indices[ind_sort]] = forces[ind_sort]
    if charge_out:
        h5md.field_q_energy[frame] = field_q_energy

    potential_energy = (
        bond2_energy + bond3_energy + bond4_energy + field_energy
        + field_q_energy
    )

    total_momentum = config.mass * comm.allreduce(
        np.sum(velocities, axis=0), MPI.SUM
    )
    angular_momentum = config.mass * comm.allreduce(
        np.sum(np.cross(positions, velocities), axis=0), MPI.SUM
    )
    torque = config.mass * comm.allreduce(
        np.sum(np.cross(positions, forces), axis=0), MPI.SUM
    )
    h5md.total_energy[frame] = kinetic_energy + potential_energy
    h5md.potential_energy[frame] = potential_energy
    h5md.kinetc_energy[frame] = kinetic_energy
    h5md.bond_energy[frame] = bond2_energy
    h5md.angle_energy[frame] = bond3_energy
    h5md.dihedral_energy[frame] = bond4_energy
    h5md.field_energy[frame] = field_energy
    h5md.total_momentum[frame, :] = total_momentum
    h5md.angular_momentum[frame, :] = angular_momentum
    h5md.torque[frame, :] = torque
    h5md.temperature[frame] = temperature
    h5md.thermostat_work[frame] = config.thermostat_work

    fmt_ = [
        "step",
        "time",
        "temp",
        "tot E",
        "kin E",
        "pot E",
        "field E",
        "elec E",
        "bond E",
        "ang E",
        "dih E",
        "Px",
        "Py",
        "Pz",
        "ΔH" if config.target_temperature else "ΔE",
    ]
    fmt_ = np.array(fmt_)
    
    # create mask to show only energies > 0
    en_array = np.array([
        field_energy,
        field_q_energy,
        bond2_energy,
        bond3_energy,
        bond4_energy,
    ])
    mask = np.full_like(fmt_, True, dtype=bool)
    mask[range(6,11)] = en_array > 0.

    header_ = fmt_[mask].shape[0] * "{:>13}"
    if config.initial_energy is None:
        fmt_[-1] = ""

    divide_by = 1.0
    if dump_per_particle:
        for i in range(3, 9):
            fmt_[i] = fmt_[i][:-2] + "E/N"
        fmt_[-1] += "/N"
        divide_by = config.n_particles
    total_energy = kinetic_energy + potential_energy
    if config.initial_energy is not None:
        if config.target_temperature:
            H_tilde = (
                total_energy - config.initial_energy - config.thermostat_work
            )
        else:
            H_tilde = total_energy - config.initial_energy
    else:
        H_tilde = 0.0

    header = header_.format(*fmt_[mask])
    data_fmt = f'{"{:13}"}{(fmt_[mask].shape[0]-1) * "{:13.5g}" }'
    all_data = [
        step,
        time_step * step,
        temperature,
        total_energy / divide_by,
        kinetic_energy / divide_by,
        potential_energy / divide_by,
        field_energy / divide_by,
        field_q_energy / divide_by,
        bond2_energy / divide_by,
        bond3_energy / divide_by,
        bond4_energy / divide_by,
        total_momentum[0] / divide_by,
        total_momentum[1] / divide_by,
        total_momentum[2] / divide_by,
        H_tilde / divide_by,
    ]
    data = data_fmt.format(*[val for i,val in enumerate(all_data) if mask[i]])
    Logger.rank0.log(logging.INFO, ("\n" + header + "\n" + data))


def distribute_input(
    in_file, rank, size, n_particles, max_molecule_size=201,
    comm=MPI.COMM_WORLD
):
    """Assign global arrays onto MPI ranks, attempting load balancing

    Distributes approximately equal numbers of particles (workload) onto each
    independent MPI rank, while respecting the requirement that any molecule
    must be fully contained on a single MPI rank only (no splitting molecules
    across multiple CPUs).

    Parameters
    ----------
    in_file : h5py.File
        HDF5 input file object.
    rank : int
        Local rank number for this MPI rank.
    size : int
        Global size of the MPI communicator (number of total CPUs).
    n_particles : int
        Total number of particles.
    max_molecule_size : int, optional
        Maximum size of any molecule present in the system. Used to initially
        guess where the MPI rank boundaries (start/end indices) in the global
        arrays should be placed. If molecules of size
        :code:`>max_molecule_size` exist in the simulation system, HyMD
        **might** work as expected. Or it might fail spectacularly.
    comm : mpi4py.Comm
        MPI communicator to use for rank commuication.

    Returns
    -------
    rank_range :
        Starting and ending indices in the global arrays for each MPI rank.
    """
    if n_particles is None:
        n_particles = len(in_file["indices"])
    np_per_MPI = n_particles // size

    molecules_flag = False
    if "molecules" in in_file:
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
    # min(max_molecule_size, n_particles // n_MPI_ranks) atoms.
    max_molecule_size += 2
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

    molecules = in_file["molecules"][mpi_range_start:mpi_range_end]
    indices = in_file["indices"][mpi_range_start:mpi_range_end]
    molecule_end_indices = np.nonzero(np.diff(molecules))[0]

    p_mpi_range = [None, None]
    if rank == 0:
        p_mpi_range[0] = 0
        if size == 1:
            p_mpi_range[1] = n_particles
        else:
            p_mpi_range[1] = indices[
                molecule_end_indices[molecule_end_indices >= np_per_MPI][0] + 1
            ]
    elif rank == size - 1:
        p_mpi_range[0] = (
            indices[molecule_end_indices[molecule_end_indices > 0][0]] + 1
        )
        p_mpi_range[1] = n_particles
    else:
        p_mpi_range[0] = (
            indices[molecule_end_indices[molecule_end_indices > 0][0]] + 1
        )
        p_mpi_range[1] = (
            indices[molecule_end_indices[molecule_end_indices > np_per_MPI][0]] + 1  # noqa: E501
        )
    return list(range(p_mpi_range[0], p_mpi_range[1])), molecules_flag
