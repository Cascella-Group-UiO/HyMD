import numpy as np
import h5py
import os
import logging
import getpass
from mpi4py import MPI
from logger import Logger


class OutDataset:
    def __init__(self, dest_directory, config, double_out=False,
                 disable_mpio=False, comm=MPI.COMM_WORLD):
        self.disable_mpio = disable_mpio
        self.config = config
        if double_out:
            self.float_dtype = "float64"
        else:
            self.float_dtype = "float32"

        if disable_mpio:
            self.file = h5py.File(
                os.path.join(
                    dest_directory, f"sim.hdf5-{comm.rank:6d}-of-{comm.size:6d}"
                ),
                "w",
            )
        else:
            self.file = h5py.File(
                os.path.join(dest_directory, "sim.h5"), "w", driver="mpio", comm=comm
            )

    def close_file(self, comm=MPI.COMM_WORLD):
        comm.Barrier()
        self.file.close()

    def flush(self):
        self.file.flush()


def setup_time_dependent_element(
    name, parent_group, n_frames, shape, dtype, units=None
):
    group = parent_group.create_group(name)
    step = group.create_dataset("step", (n_frames,), "int32")
    time = group.create_dataset("time", (n_frames,), "float32")
    value = group.create_dataset("value", (n_frames, *shape), dtype)
    if units is not None:
        group.attrs["units"] = units
    return group, step, time, value


def store_static(
    h5md,
    rank_range,
    names,
    types,
    indices,
    config,
    bonds_2_atom1,
    bonds_2_atom2,
    velocity_out=False,
    force_out=False,
    comm=MPI.COMM_WORLD,
):
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

    # Check if we are in a git repo and grab the commit hash and the branch if
    # we are, append it to the version number in the output specification. Also
    # grab the user email from git config if we can find it.
    try:
        import git

        try:
            repo_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir)
            )
            repo = git.Repo(repo_dir)
            commit = repo.head.commit
            branch = repo.active_branch
            version_add = f"[{branch}-branch commit {commit}]"
            try:
                reader = repo.config_reader()
                user_email = reader.get_value("user", "email")
                author_group.attrs["email"] = np.string_(user_email)
            except KeyError:
                ...
        except git.exc.InvalidGitRepositoryError:
            version_add = ""
    except ModuleNotFoundError:
        version_add = ""
    creator_group.attrs["version"] = np.string_("0.0 " + version_add)

    h5md.particles_group = h5md.file.create_group("/particles")
    h5md.all_particles = h5md.particles_group.create_group("all")
    mass = h5md.all_particles.create_dataset("mass", (config.n_particles,), dtype)
    mass[:].fill(config.mass)

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

    # Time dependent box, fix this later.
    # h5md.box_step = h5md.edges.create_dataset('step', (n_frames,), 'i')
    # h5md.box_time = h5md.edges.create_dataset('time', (n_frames,), 'float32')
    # h5md.box_value = h5md.edges.create_dataset('value', (n_frames, 3), 'float32')  # noqa: E501

    species = h5md.all_particles.create_dataset(
        "species", (config.n_particles,), dtype="i"  # noqa: E501
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
        units="nanometers",
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
            units="nanometers/picosecond",
        )
    if force_out:
        (
            _,
            h5md.forces_step,
            h5md.forces_time,
            h5md.forces
        ) = setup_time_dependent_element(
            'force',
            h5md.all_particles,
            n_frames,
            (config.n_particles, 3),
            dtype,
            units='kJ/mol nanometer'
        )
    (
        _,
        h5md.total_energy_step,
        h5md.total_energy_time,
        h5md.total_energy,
    ) = setup_time_dependent_element(
        "total_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    )
    (
        _,
        h5md.kinetc_energy_step,
        h5md.kinetc_energy_time,
        h5md.kinetc_energy,
    ) = setup_time_dependent_element(
        "kinetic_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    )
    (
        _,
        h5md.potential_energy_step,
        h5md.potential_energy_time,
        h5md.potential_energy,
    ) = setup_time_dependent_element(  # noqa: E501
        "potential_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    )
    (
        _,
        h5md.bond_energy_step,
        h5md.bond_energy_time,
        h5md.bond_energy,
    ) = setup_time_dependent_element(
        "bond_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    )
    (
        _,
        h5md.angle_energy_step,
        h5md.angle_energy_time,
        h5md.angle_energy,
    ) = setup_time_dependent_element(
        "angle_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    )
    (
        _,
        h5md.field_energy_step,
        h5md.field_energy_time,
        h5md.field_energy,
    ) = setup_time_dependent_element(
        "field_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    )
    (
        _,
        h5md.field_q_energy_step,
        h5md.field_q_energy_time,
        h5md.field_q_energy,
    ) = setup_time_dependent_element(
        "field_q_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    ) # <-------- xinmeng 
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
        units="nanometers g/picosecond mol",
    )
    (
        _,
        h5md.temperature_step,
        h5md.temperature_time,
        h5md.temperature,
    ) = setup_time_dependent_element(
        "temperature", h5md.observables, n_frames, (3,), dtype, units="Kelvin"
    )
    (
        _,
        h5md.thermostat_work_step,
        h5md.thermostat_work_time,
        h5md.thermostat_work,
    ) = setup_time_dependent_element(
        "thermostat_work", h5md.observables, n_frames, (1,), "float32", units="kJ/mol"
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
        bonds_from[rank_bond_start + i] = indices[a]
        bonds_to[rank_bond_start + i] = indices[b]



def store_static_with_charge(
    h5md,
    rank_range,
    names,
    types,
    indices,
    charges, #<------- simple add 
    config,
    bonds_2_atom1,
    bonds_2_atom2,
    velocity_out=False,
    force_out=False,
    comm=MPI.COMM_WORLD,
):
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

    # Check if we are in a git repo and grab the commit hash and the branch if
    # we are, append it to the version number in the output specification. Also
    # grab the user email from git config if we can find it.
    try:
        import git

        try:
            repo_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir)
            )
            repo = git.Repo(repo_dir)
            commit = repo.head.commit
            branch = repo.active_branch
            version_add = f"[{branch}-branch commit {commit}]"
            try:
                reader = repo.config_reader()
                user_email = reader.get_value("user", "email")
                author_group.attrs["email"] = np.string_(user_email)
            except KeyError:
                ...
        except git.exc.InvalidGitRepositoryError:
            version_add = ""
    except ModuleNotFoundError:
        version_add = ""
    creator_group.attrs["version"] = np.string_("0.0 " + version_add)
    
    h5md.particles_group = h5md.file.create_group("/particles")
    h5md.all_particles = h5md.particles_group.create_group("all")
    mass = h5md.all_particles.create_dataset("mass", (config.n_particles,), dtype)
    mass[:].fill(config.mass)
    ### WARNING THIS DOES NOT FILL THE config.mass, still all zero
    ### TBF, ok for now 
    
    ### add  charge  
    charge = h5md.all_particles.create_dataset('charge', (config.n_particles,) , dtype='float32') #charges.shape
    charge[indices]= charges 
    #print(indices, len(indices))
    #print('shape of charges inside the static', charges.shape)
    ### NO USE http://pdebuyl.be/blog/2014/pmi-and-h5py.html
    ### NO USE https://github.com/pdebuyl/pmi-h5py/blob/master/test_pmi_mod.py
    ### NO USE https://gist.github.com/apdavison/36126ee26067592ee69bf51b57fd3f31
    
    
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

    # Time dependent box, fix this later.
    # h5md.box_step = h5md.edges.create_dataset('step', (n_frames,), 'i')
    # h5md.box_time = h5md.edges.create_dataset('time', (n_frames,), 'float32')
    # h5md.box_value = h5md.edges.create_dataset('value', (n_frames, 3), 'float32')  # noqa: E501

    species = h5md.all_particles.create_dataset(
        "species", (config.n_particles,), dtype="i"  # noqa: E501
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
        units="nanometers",
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
            units="nanometers/picosecond",
        )
    if force_out:
        (
            _,
            h5md.forces_step,
            h5md.forces_time,
            h5md.forces
        ) = setup_time_dependent_element(
            'force',
            h5md.all_particles,
            n_frames,
            (config.n_particles, 3),
            dtype,
            units='kJ/mol nanometer'
        )
    (
        _,
        h5md.total_energy_step,
        h5md.total_energy_time,
        h5md.total_energy,
    ) = setup_time_dependent_element(
        "total_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    )
    (
        _,
        h5md.kinetc_energy_step,
        h5md.kinetc_energy_time,
        h5md.kinetc_energy,
    ) = setup_time_dependent_element(
        "kinetic_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    )
    (
        _,
        h5md.potential_energy_step,
        h5md.potential_energy_time,
        h5md.potential_energy,
    ) = setup_time_dependent_element(  # noqa: E501
        "potential_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    )
    (
        _,
        h5md.bond_energy_step,
        h5md.bond_energy_time,
        h5md.bond_energy,
    ) = setup_time_dependent_element(
        "bond_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    )
    (
        _,
        h5md.angle_energy_step,
        h5md.angle_energy_time,
        h5md.angle_energy,
    ) = setup_time_dependent_element(
        "angle_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    )
    (
        _,
        h5md.field_energy_step,
        h5md.field_energy_time,
        h5md.field_energy,
    ) = setup_time_dependent_element(
        "field_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    )
    (
        _,
        h5md.field_q_energy_step,
        h5md.field_q_energy_time,
        h5md.field_q_energy,
    ) = setup_time_dependent_element(
        "field_q_energy", h5md.observables, n_frames, (1,), dtype, units="kJ/mol"
    ) # <-------- xinmeng 
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
        units="nanometers g/picosecond mol",
    )
    (
        _,
        h5md.temperature_step,
        h5md.temperature_time,
        h5md.temperature,
    ) = setup_time_dependent_element(
        "temperature", h5md.observables, n_frames, (3,), dtype, units="Kelvin"
    )
    (
        _,
        h5md.thermostat_work_step,
        h5md.thermostat_work_time,
        h5md.thermostat_work,
    ) = setup_time_dependent_element(
        "thermostat_work", h5md.observables, n_frames, (1,), "float32", units="kJ/mol"
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
        bonds_from[rank_bond_start + i] = indices[a]
        bonds_to[rank_bond_start + i] = indices[b]


def store_data(
    h5md,
    step,
    frame,
    indices,
    positions,
    velocities,
    forces,
    box_size,
    temperature,
    kinetic_energy,
    bond2_energy,
    bond3_energy,
    field_energy,
    field_q_energy, ##<----- xinmeng add
    time_step,
    config,
    velocity_out=False,
    force_out=False,
    dump_per_particle=False,
    comm=MPI.COMM_WORLD,
):
    for dset in (
        h5md.positions_step,
        h5md.total_energy_step,
        h5md.potential_energy,
        h5md.kinetc_energy_step,
        h5md.bond_energy_step,
        h5md.angle_energy_step,
        h5md.field_energy_step,
        h5md.field_q_energy_step, ##<----- xinmeng add
        h5md.total_momentum_step,
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
        h5md.field_energy_time,
        h5md.field_q_energy_time, ##<----- xinmeng add
        h5md.total_momentum_time,
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

    # Time dependent box, fix this later.
    # h5md.box_step[frame] = step
    # h5md.box_time[frame] = step * time_step
    # h5md.box_value[frame, ...] = np.array(box_size)

    ind_sort = np.argsort(indices)
    h5md.positions[frame, indices[ind_sort]] = positions[ind_sort]

    if velocity_out:
        h5md.velocities[frame, indices[ind_sort]] = velocities[ind_sort]
    if force_out:
        h5md.forces[frame, indices[ind_sort]] = forces[ind_sort]

    potential_energy = bond2_energy + bond3_energy + field_energy + field_q_energy #<-------- xinmeng
    total_momentum = config.mass * comm.allreduce(np.sum(velocities, axis=0), MPI.SUM)
    h5md.total_energy[frame] = kinetic_energy + potential_energy
    h5md.potential_energy[frame] = potential_energy
    h5md.kinetc_energy[frame] = kinetic_energy
    h5md.bond_energy[frame] = bond2_energy
    h5md.angle_energy[frame] = bond3_energy
    h5md.field_energy[frame] = field_energy
    h5md.field_q_energy[frame] = field_q_energy  #<-------- xinmeng
    h5md.total_momentum[frame, :] = total_momentum
    h5md.temperature[frame] = temperature
    h5md.thermostat_work[frame] = config.thermostat_work

    header_ = 13 * "{:>15}"
    fmt_ = [
        "step",
        "time",
        "temperature",
        "total E",
        "kinetic E",
        "potential E",
        "field E",
        "field q E", #<-------- xinmeng 
        "bond E",
        "angle E",
        "total Px",
        "total Py",
        "total Pz",
        "ΔH tilde" if config.target_temperature else "ΔE",
    ]
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
            H_tilde = total_energy - config.initial_energy - config.thermostat_work
        else:
            H_tilde = total_energy - config.initial_energy
    else:
        H_tilde = 0.0

    header = header_.format(*fmt_)
    data_fmt = f'{"{:15}"}{12 * "{:15.8g}" }'
    data = data_fmt.format(
        step,
        time_step * step,
        temperature,
        total_energy / divide_by,
        kinetic_energy / divide_by,
        potential_energy / divide_by,
        field_energy / divide_by,
        field_q_energy / divide_by, #<-------- 
        bond2_energy / divide_by,
        bond3_energy / divide_by,
        total_momentum[0] / divide_by,
        total_momentum[1] / divide_by,
        total_momentum[2] / divide_by,
        H_tilde / divide_by,
    )  # noqa: E501
    Logger.rank0.log(logging.INFO, ("\n" + header + "\n" + data))


def distribute_input(
    in_file, rank, size, n_particles, max_molecule_size=201, comm=MPI.COMM_WORLD
):
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
    grab_extra = max_molecule_size if np_per_MPI > max_molecule_size else np_per_MPI
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
        p_mpi_range[0] = indices[molecule_end_indices[molecule_end_indices > 0][0]] + 1
        p_mpi_range[1] = n_particles
    else:
        p_mpi_range[0] = indices[molecule_end_indices[molecule_end_indices > 0][0]] + 1
        p_mpi_range[1] = (
            indices[molecule_end_indices[molecule_end_indices > np_per_MPI][0]] + 1
        )
    return list(range(p_mpi_range[0], p_mpi_range[1])), molecules_flag
