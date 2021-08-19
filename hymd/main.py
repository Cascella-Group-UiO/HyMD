from argparse import ArgumentParser
import atexit
import cProfile
import datetime
import h5py
import logging
from mpi4py import MPI
import numpy as np
import os
import pmesh.pm as pmesh
import pstats
import sys
from types import ModuleType as moduleobj
import warnings


from hamiltonian import DefaultNoChi, DefaultWithChi
from field import (
    compute_field_force,
    update_field,
    compute_field_and_kinetic_energy,
    domain_decomposition,
)
from file_io import distribute_input, OutDataset, store_static, store_data
from force import compute_bond_forces__fortran as compute_bond_forces
from force import compute_angle_forces__fortran as compute_angle_forces
from force import prepare_bonds
from input_parser import (
    read_config_toml,
    parse_config_toml,
    check_config,
    convert_CONF_to_config,
)
from integrator import integrate_velocity, integrate_position
from logger import Logger
from thermostat import csvr_thermostat
from pressure import comp_pressure
from barostat import isotropic, semiisotropic


def fmtdt(timedelta):  ### FIX ME (move this somewhere else)
    days = timedelta.days
    hours, rem = divmod(timedelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    microseconds = timedelta.microseconds
    ret_str = ""
    if days != 0:
        ret_str += f"{days} days "
    ret_str += f"{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:06d}"
    return ret_str


def configure_runtime(comm):
    ap = ArgumentParser()
    ap.add_argument(
        "-v",
        "--verbose",
        default=0,
        type=int,
        nargs="?",
        help="Increase logging verbosity",
    )
    ap.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="Profile program execution with cProfile",
    )
    ap.add_argument(
        "--disable-field",
        default=False,
        action="store_true",
        help="Disable field forces",
    )
    ap.add_argument(
        "--disable-bonds",
        default=False,
        action="store_true",
        help="Disable two-particle bond forces",
    )
    ap.add_argument(
        "--disable-angle-bonds",
        default=False,
        action="store_true",
        help="Disable three-particle angle bond forces",
    )
    ap.add_argument(
        "--double-precision",
        default=False,
        action="store_true",
        help="Use double precision positions/velocities",
    )
    ap.add_argument(
        "--double-output",
        default=False,
        action="store_true",
        help="Use double precision in output h5md",
    )
    ap.add_argument(
        "--dump-per-particle",
        default=False,
        action="store_true",
        help="Log energy values per particle, not total",
    )
    ap.add_argument(
        "--force-output",
        default=False,
        action="store_true",
        help="Dump forces to h5md output",
    )
    ap.add_argument(
        "--velocity-output",
        default=False,
        action="store_true",
        help="Dump velocities to h5md output",
    )
    ap.add_argument(
        "--disable-mpio",
        default=False,
        action="store_true",
        help=("Avoid using h5py-mpi, potentially decreasing IO " "performance"),
    )
    ap.add_argument(
        "--destdir", default=".", help="Write output to specified directory"
    )
    ap.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Set the numpy random generator seed for every rank",
    )
    ap.add_argument(
        "--logfile", default=None, help="Redirect event logging to specified file"
    )
    ap.add_argument("config", help="Config .py or .toml input configuration script")
    ap.add_argument("input", help="input.hdf5")
    args = ap.parse_args()

    # Given as '--verbose' or '-v' without a specific value specified,
    # default to 1
    if args.verbose is None:
        args.verbose = 1

    if comm.rank == 0:
        os.makedirs(args.destdir, exist_ok=True)
    comm.barrier()

    if args.seed is not None:
        np.random.seed(args.seed)
    else:
        np.random.seed()

    # Setup logger
    Logger.setup(
        default_level=logging.INFO, log_file=args.logfile, verbose=args.verbose
    )

    if args.profile:
        prof_file_name = "cpu.txt-%05d-of-%05d" % (comm.rank, comm.size)
        output_file = open(os.path.join(args.destdir, prof_file_name), "w")
        pr = cProfile.Profile()

        def profile_atexit():
            pr.disable()
            # Dump results:
            # - for binary dump
            prof_file_bin = "cpu.prof-%05d-of-%05d" % (comm.rank, comm.size)
            pr.dump_stats(os.path.join(args.destdir, prof_file_bin))
            stats = pstats.Stats(pr, stream=output_file)
            stats.sort_stats("time").print_stats()
            output_file.close()

        # TODO: if we have a main function then we can properly do set up and
        # teardown without using atexit.
        atexit.register(profile_atexit)

        pr.enable()

    try:
        Logger.rank0.log(
            logging.INFO, f"Attempting to parse config file {args.config} as .toml"
        )
        toml_config = read_config_toml(args.config)
        config = parse_config_toml(
            toml_config, file_path=os.path.abspath(args.config), comm=comm
        )
        Logger.rank0.log(
            logging.INFO, f"Successfully parsed {args.config} as .toml file"
        )
        config.command_line_full = " ".join(sys.argv)
        Logger.rank0.log(logging.INFO, str(config))
    except ValueError as ve:
        try:
            Logger.rank0.log(
                logging.INFO,
                (
                    f"Attempt to parse {args.config} as .toml failed, trying "
                    "to parse as python file"
                ),
            )
            CONF = {}
            exec(open(args.config).read(), CONF)
            CONF = {
                k: v
                for k, v in CONF.items()
                if (not k.startswith("_") and not isinstance(v, moduleobj))
            }

            Logger.rank0.log(
                logging.INFO, f"Successfully parsed {args.config} as .py file"
            )
            for key, value in sorted(CONF.items()):
                Logger.rank0.log(logging.INFO, f"{key} = {value}")
            config = convert_CONF_to_config(CONF, file_path=args.config)
        except NameError as ne:
            Logger.rank0.log(
                logging.ERROR, (f"Attempt to parse {args.config} as .py failed" f", ")
            )
            raise ValueError(
                f"Unable to parse configuration file {args.config}"
                + "\n\ntoml parse traceback:"
                + repr(ve)
                + "\n\npython parse traceback:"
                + repr(ne)
            )
    return args, config


def cancel_com_momentum(velocities, config, comm=MPI.COMM_WORLD):
    com_velocity = comm.allreduce(np.sum(velocities[...], axis=0), MPI.SUM)
    velocities[...] = velocities[...] - com_velocity / config.n_particles
    return velocities


def generate_initial_velocities(velocities, config, comm=MPI.COMM_WORLD):
    kT_start = config.R * config.start_temperature
    n_particles_ = velocities.shape[0]
    velocities[...] = np.random.normal(
        loc=0, scale=kT_start / config.mass, size=(n_particles_, 3)
    )
    com_velocity = comm.allreduce(np.sum(velocities[...], axis=0), MPI.SUM)
    velocities[...] = velocities[...] - com_velocity / config.n_particles
    kinetic_energy = comm.allreduce(
        0.5 * config.mass * np.sum(velocities ** 2), MPI.SUM
    )
    start_kinetic_energy_target = (
        1.5 * config.R * config.n_particles * config.start_temperature
    )
    factor = np.sqrt(1.5 * config.n_particles * kT_start / kinetic_energy)
    velocities[...] = velocities[...] * factor
    kinetic_energy = comm.allreduce(
        0.5 * config.mass * np.sum(velocities ** 2), MPI.SUM
    )
    Logger.rank0.log(
        logging.INFO,
        (
            f"Initialized {config.n_particles} velocities, target kinetic energy:"
            f" {start_kinetic_energy_target}, actual kinetic energy generated:"
            f" {kinetic_energy}"
        ),
    )
    return velocities

def initialize_pm():

    # Ignore numpy numpy.VisibleDeprecationWarning: Creating an ndarray from
    # ragged nested sequences until it is fixed in pmesh
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=np.VisibleDeprecationWarning,
            message=r"Creating an ndarray from ragged nested sequences",
        )
        # The first argument of ParticleMesh has to be a tuple
        pm = pmesh.ParticleMesh(
            config.mesh_size, BoxSize=config.box_size, dtype="f4", comm=comm
        )

    phi = [pm.create("real", value=0.0) for _ in range(config.n_types)]
    phi_fourier = [
        pm.create("complex", value=0.0) for _ in range(config.n_types)
    ]  # noqa: E501
    force_on_grid = [
        [pm.create("real", value=0.0) for d in range(3)] for _ in range(config.n_types)
    ]
    v_ext_fourier = [pm.create("complex", value=0.0) for _ in range(4)]
    v_ext = [pm.create("real", value=0.0) for _ in range(config.n_types)]

    lap_transfer = [pm.create("complex", value=0.0) for _ in range(3)]
    phi_laplacian = [
        [pm.create("real", value=0.0) for d in range(3)] for _ in range(config.n_types)
    ]
    field_list = [phi, phi_fourier, force_on_grid, v_ext_fourier, v_ext, lap_transfer,
            phi_laplacian]
    return (pm, phi, phi_fourier, force_on_grid, v_ext_fourier, v_ext, lap_transfer,
            phi_laplacian, field_list)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        start_time = datetime.datetime.now()

    args, config = configure_runtime(comm)

    if args.double_precision:
        dtype = np.float64
        if dtype == np.float64:
            from force import (
                                    compute_bond_forces__fortran__double as compute_bond_forces,
            )  # noqa: E501, F811
            from force import (
                compute_angle_forces__fortran__double as compute_angle_forces,
            )  # noqa: E501, F811
    else:
        dtype = np.float32

    driver = "mpio" if not args.disable_mpio else None
    with h5py.File(args.input, "r", driver=driver, comm=comm) as in_file:
        rank_range, molecules_flag = distribute_input(
            in_file,
            rank,
            size,
            config.n_particles,  ### << USE config here, update n_particles if not given
            config.max_molecule_size if config.max_molecule_size else 201,
            comm=comm,
        )
        indices = in_file["indices"][rank_range]
        positions = in_file["coordinates"][-1, rank_range, :]
        positions = positions.astype(dtype)
        if "velocities" in in_file:
            velocities = in_file["velocities"][-1, rank_range, :]
            velocities = velocities.astype(dtype)
        else:
            velocities = np.zeros_like(positions, dtype=dtype)

        names = in_file["names"][rank_range]

        types = None
        bonds = None
        molecules = []
        #molecules = None
        if "types" in in_file:
            types = in_file["types"][rank_range]
        if molecules_flag:
            molecules = in_file["molecules"][rank_range]
            bonds = in_file["bonds"][rank_range]

    config = check_config(config, indices, names, types, comm=comm)
    if config.n_print:
        if config.n_flush is None:
            config.n_flush = 10000 // config.n_print

    if config.start_temperature:
        velocities = generate_initial_velocities(velocities, config, comm=comm)
    elif config.cancel_com_momentum:
        velocities = cancel_com_momentum(velocities, config, comm=comm)

    positions = np.mod(positions, config.box_size[None, :])

    bond_forces = np.zeros(
        shape=(len(positions), 3), dtype=dtype
    )  # , order='F')  # noqa: E501
    angle_forces = np.zeros(
        shape=(len(positions), 3), dtype=dtype
    )  # , order='F')  # noqa: E501
    field_forces = np.zeros(shape=(len(positions), 3), dtype=dtype)

    field_energy = 0.0
    bond_energy = 0.0
    angle_energy = 0.0
    kinetic_energy = 0.0


    if config.hamiltonian.lower() == "defaultnochi":
        hamiltonian = DefaultNoChi(config)
    elif config.hamiltonian.lower() == "defaultwithchi":
        hamiltonian = DefaultWithChi(
            config, config.unique_names, config.type_to_name_map
        )
    else:
        err_str = (
            f"The specified Hamiltonian {config.hamiltonian} was not "
            f"recognized as a valid Hamiltonian."
        )
        Logger.rank0.log(logging.ERROR, err_str)
        if rank == 0:
            raise NotImplementedError(err_str)

    pm_stuff  = initialize_pm()
    (pm, phi, phi_fourier, force_on_grid, v_ext_fourier, v_ext, lap_transfer, phi_laplacian,
    field_list) = pm_stuff
    Logger.rank0.log(logging.INFO, f"pfft-python processor mesh: {str(pm.np)}")
    #print('Creating phi_fourier ',phi_fourier[0].value[0][0][0:2])
    #print('Creating phi_fft ',phi_fft[0].value[0][0][0:2])

    if config.domain_decomposition:
        dd = domain_decomposition(
            positions,
            pm,
            velocities,
            indices,
            bond_forces,
            angle_forces,
            field_forces,
            names,
            types,
            molecules=molecules if molecules_flag else None,
            bonds=bonds if molecules_flag else None,
            verbose=args.verbose,
            comm=comm,
        )
        if molecules_flag:
            (
                positions,
                velocities,
                indices,
                bond_forces,
                angle_forces,
                field_forces,
                names,
                types,
                bonds,
                molecules,
            ) = dd
        else:
            (
                positions,
                velocities,
                indices,
                bond_forces,
                angle_forces,
                field_forces,
                names,
                types,
            ) = dd
    positions = np.asfortranarray(positions)
    velocities = np.asfortranarray(velocities)
    bond_forces = np.asfortranarray(bond_forces)
    angle_forces = np.asfortranarray(angle_forces)

    if not args.disable_field:
        layouts = [pm.decompose(positions[types == t]) for t in range(config.n_types)]
        update_field(
            phi,
            layouts,
            force_on_grid,
            hamiltonian,
            pm,
            positions,
            types,
            config,
            v_ext,
            phi_fourier,
            v_ext_fourier,
            compute_potential=True,
        )
        #print('Updating phi_fourier for t=0',phi_fourier[0].value[0][0][0:2])
        #print('Updating phi_fourier for t=1',phi_fourier[1].value[0][0][0:2])

        field_energy, kinetic_energy = compute_field_and_kinetic_energy(
            phi,
            velocities,
            hamiltonian,
            positions,
            types,
            v_ext,
            config,
            layouts,
            comm=comm,
        )
        compute_field_force(
            layouts, positions, force_on_grid, field_forces, types, config.n_types
        )
    else:
        kinetic_energy = comm.allreduce(0.5 * config.mass * np.sum(velocities ** 2))

    if molecules_flag:
        if not (args.disable_bonds and args.disable_angle_bonds):
            bonds_prep = prepare_bonds(molecules, names, bonds, indices, config)
            (
                bonds_2_atom1,
                bonds_2_atom2,
                bonds_2_equilibrium,
                bonds_2_stength,
                bonds_3_atom1,
                bonds_3_atom2,
                bonds_3_atom3,
                bonds_3_equilibrium,
                bonds_3_stength,
            ) = bonds_prep
        if not args.disable_bonds:
            bond_energy_,bond_pr_= compute_bond_forces(
                bond_forces,
                positions,
                config.box_size,
                bonds_2_atom1,
                bonds_2_atom2,
                bonds_2_equilibrium,
                bonds_2_stength,
            )
            bond_energy = comm.allreduce(bond_energy_, MPI.SUM)
            #bond_pr = comm.allreduce(bond_pr_, MPI.SUM)
            #if(rank == 0):
            #    print('bond_energy:',bond_energy)
            #    print('bond_pr in x:', bond_pr[0])
            #    print('bond_pr in y:', bond_pr[1])
            #    print('bond_pr in z:', bond_pr[2])
        if not args.disable_angle_bonds:
            angle_energy_, angle_pr_ = compute_angle_forces(
                angle_forces,
                positions,
                config.box_size,
                bonds_3_atom1,
                bonds_3_atom2,
                bonds_3_atom3,
                bonds_3_equilibrium,
                bonds_3_stength,
            )
            angle_energy = comm.allreduce(angle_energy_, MPI.SUM)
            angle_pr = comm.allreduce(angle_pr_, MPI.SUM)
            #if(rank == 0):
            #    print('angle_energy:',angle_energy)
            #    print('angle_pr in x:', angle_pr[0])
            #    print('angle_pr in y:', angle_pr[1])
            #    print('angle_pr in z:', angle_pr[2])
        else:
            bonds_2_atom1, bonds_2_atom2 = [], []
    else:
        #bonds_2_atom1, bonds_2_atom2 = None, None
        bonds_2_atom1, bonds_2_atom2 = [], []

    config.initial_energy = field_energy + kinetic_energy + bond_energy + angle_energy
    out_dataset = OutDataset(args.destdir, config,
                             double_out=args.double_output,
                             disable_mpio=args.disable_mpio)
    store_static(
        out_dataset,
        rank_range,
        names,
        types,
        indices,
        config,
        bonds_2_atom1,
        bonds_2_atom2,
        velocity_out=args.velocity_output,
        force_out=args.force_output,
        comm=comm,
    )

    if config.n_print > 0:
        step = 0
        frame = 0
        if not args.disable_field:
            field_energy, kinetic_energy = compute_field_and_kinetic_energy(
                phi,
                velocities,
                hamiltonian,
                positions,
                types,
                v_ext,
                config,
                layouts,
                comm=comm,
            )
        else:
            kinetic_energy = comm.allreduce(0.5 * config.mass * np.sum(velocities ** 2))
        temperature = (2 / 3) * kinetic_energy / (config.R * config.n_particles)  # noqa: E501
        if config.pressure:
            pressure = comp_pressure(
                    phi,
                    hamiltonian,
                    velocities,
                    config,
                    phi_fourier,
                    phi_laplacian,
                    lap_transfer,
                    args,
                    bond_forces,
                    angle_forces,
                    positions,
                    bond_pr_,
                    angle_pr_,
                    comm=comm
            )
            #if rank ==0 : print(pressure[9:12])
            #print('phi_fft after pressure call: phi_fft[d=0]',phi_fft[0].value[0][0][0:2])
        else:
            pressure = 0.0 #0.0 indicates not calculated. To be changed.
        store_data(
            out_dataset,
            step,
            frame,
            indices,
            positions,
            velocities,
            field_forces + bond_forces + angle_forces,
            config.box_size,
            temperature,
            pressure,
            kinetic_energy,
            bond_energy,
            angle_energy,
            field_energy,
            config.time_step,
            config,
            velocity_out=args.velocity_output,
            force_out=args.force_output,
            dump_per_particle=args.dump_per_particle,
            comm=comm,
        )

    if rank == 0:
        loop_start_time = datetime.datetime.now()
        last_step_time = datetime.datetime.now()

    flush_step = 0
    # ======================================================================= #
    # =================  |\/| |¯¯\     |    |¯¯| |¯¯| |¯¯)  ================= #
    # =================  |  | |__/     |___ |__| |__| |¯¯   ================= #
    # ======================================================================= #
    for step in range(config.n_steps):
        current_step_time = datetime.datetime.now()

        if step == 0 and args.verbose > 1:
            Logger.rank0.log(logging.INFO, f"MD step = {step:10d}")
        else:
            log_step = False
            if config.n_steps < 1000:
                log_step = True
            elif (
                np.mod(step, config.n_steps // 1000) == 0
                or np.mod(step, config.n_print) == 0
            ):
                log_step = True
            if rank == 0 and log_step and args.verbose > 2:
                step_t = current_step_time - last_step_time
                tot_t = current_step_time - loop_start_time
                avg_t = (current_step_time - loop_start_time) / (step + 1)
                ns_sim = (step + 1) * config.time_step / 1000

                seconds_per_day = 24 * 60 * 60
                seconds_elapsed = tot_t.days * seconds_per_day
                seconds_elapsed += tot_t.seconds
                seconds_elapsed += 1e-6 * tot_t.microseconds
                minutes_elapsed = seconds_elapsed / 60
                hours_elapsed = minutes_elapsed / 60
                days_elapsed = hours_elapsed / 24

                ns_per_day = ns_sim / days_elapsed
                hours_per_ns = hours_elapsed / ns_sim
                steps_per_s = (step + 1) / seconds_elapsed
                info_str = (
                    f"MD step = {step:10d}   step time: "
                    f"{fmtdt(step_t):22s}   Performance: "
                    f"{ns_per_day:.3f} ns/day   {hours_per_ns:.3f} hours/ns   "
                    f"{steps_per_s:.3f} steps/s"
                )
                Logger.rank0.log(logging.INFO, info_str)

        # Initial rRESPA velocity step
        velocities = integrate_velocity(
            velocities, field_forces / config.mass, config.time_step
        )

        # Inner rRESPA steps
        for inner in range(config.respa_inner):
            velocities = integrate_velocity(
                velocities,
                (bond_forces + angle_forces) / config.mass,
                config.time_step / config.respa_inner,
            )
            positions = integrate_position(
                positions, velocities, config.time_step / config.respa_inner
            )
            positions = np.mod(positions, config.box_size[None, :])

            # Update fast forces
            if molecules_flag:
                if not args.disable_bonds:
                    bond_energy_, bond_pr_ = compute_bond_forces(
                        bond_forces,
                        positions,
                        config.box_size,
                        bonds_2_atom1,
                        bonds_2_atom2,
                        bonds_2_equilibrium,
                        bonds_2_stength,
                    )
                if not args.disable_angle_bonds:
                    angle_energy_, angle_pr_ = compute_angle_forces(
                        angle_forces,
                        positions,
                        config.box_size,
                        bonds_3_atom1,
                        bonds_3_atom2,
                        bonds_3_atom3,
                        bonds_3_equilibrium,
                        bonds_3_stength,
                    )
            velocities = integrate_velocity(
                velocities,
                (bond_forces + angle_forces) / config.mass,
                config.time_step / config.respa_inner,
            )

        # Update slow forces
        if not args.disable_field:
            #should compute_field_energy be here?
            layouts = [
                pm.decompose(positions[types == t]) for t in range(config.n_types)
            ]
            update_field(
                phi,
                layouts,
                force_on_grid,
                hamiltonian,
                pm,
                positions,
                types,
                config,
                v_ext,
                phi_fourier,
                v_ext_fourier,
            )
            compute_field_force(
                layouts, positions, force_on_grid, field_forces, types, config.n_types
            )

        # Second rRESPA velocity step
        velocities = integrate_velocity(
            velocities, field_forces / config.mass, config.time_step
        )

        # Only compute and keep the molecular bond energy from the last rRESPA
        # inner step
        if molecules_flag:
            if not args.disable_bonds:
                bond_energy = comm.allreduce(bond_energy_, MPI.SUM)
            if not args.disable_angle_bonds:
                angle_energy = comm.allreduce(angle_energy_, MPI.SUM)
        if step != 0 and config.domain_decomposition:
            if np.mod(step, config.domain_decomposition) == 0:
                positions = np.ascontiguousarray(positions)
                bond_forces = np.ascontiguousarray(bond_forces)
                angle_forces = np.ascontiguousarray(angle_forces)

                dd = domain_decomposition(
                    positions,
                    pm,
                    velocities,
                    indices,
                    bond_forces,
                    angle_forces,
                    field_forces,
                    names,
                    types,
                    molecules=molecules if molecules_flag else None,
                    bonds=bonds if molecules_flag else None,
                    verbose=args.verbose,
                    comm=comm,
                )
                if molecules_flag:
                    (
                        positions,
                        velocities,
                        indices,
                        bond_forces,
                        angle_forces,
                        field_forces,
                        names,
                        types,
                        bonds,
                        molecules,
                    ) = dd
                else:
                    (
                        positions,
                        velocities,
                        indices,
                        bond_forces,
                        angle_forces,
                        field_forces,
                        names,
                        types,
                    ) = dd
                positions = np.asfortranarray(positions)
                bond_forces = np.asfortranarray(bond_forces)
                angle_forces = np.asfortranarray(angle_forces)

                layouts = [
                    pm.decompose(positions[types == t]) for t in range(config.n_types)
                ]
                if molecules_flag:
                    bonds_prep = prepare_bonds(molecules, names, bonds, indices, config)
                    (
                        bonds_2_atom1,
                        bonds_2_atom2,
                        bonds_2_equilibrium,
                        bonds_2_stength,
                        bonds_3_atom1,
                        bonds_3_atom2,
                        bonds_3_atom3,
                        bonds_3_equilibrium,
                        bonds_3_stength,
                    ) = bonds_prep

        for t in range(config.n_types):
            if args.verbose > 2:
                exchange_cost = layouts[t].get_exchange_cost()
                Logger.all_ranks.log(
                    logging.INFO,
                    (
                        f"(GHOSTS: Total number of particles of type "
                        f"{config.type_to_name_map[t]} to be "
                        f"exchanged = {exchange_cost[rank]}"
                    ),
                )

        # Thermostat
        if config.target_temperature:
            csvr_thermostat(velocities, names, config, comm=comm)

        # Berendsen Barostat
        if config.barostat:
            if config.barostat.lower() == 'isotropic':
                positions = isotropic(
                     phi,
                     hamiltonian,
                     positions,
                     velocities,
                     config,
                     phi_fourier,
                     phi_laplacian,
                     lap_transfer,
                     bond_forces,
                     angle_forces,
                     args,
                     bond_pr_,
                     angle_pr_,
                     comm=comm
                )

            elif config.barostat.lower() == 'semiisotropic':
                positions = semiisotropic(
                     phi,
                     hamiltonian,
                     positions,
                     velocities,
                     config,
                     phi_fourier,
                     phi_laplacian,
                     lap_transfer,
                     bond_forces,
                     angle_forces,
                     args,
                     bond_pr_,
                     angle_pr_,
                     comm=comm
                )

            #pmesh repair attempt: recreate all
            pm_stuff  = initialize_pm()
            (pm, phi, phi_fourier, force_on_grid, v_ext_fourier, v_ext, lap_transfer, phi_laplacian,
            field_list) = pm_stuff
            if not args.disable_field:
                layouts = [pm.decompose(positions[types == t]) for t in range(config.n_types)]
                update_field(
                    phi,
                    layouts,
                    force_on_grid,
                    hamiltonian,
                    pm,
                    positions,
                    types,
                    config,
                    v_ext,
                    phi_fourier,
                    v_ext_fourier,
                    compute_potential=True,
                )
                field_energy, kinetic_energy = compute_field_and_kinetic_energy(
                    phi,
                    velocities,
                    hamiltonian,
                    positions,
                    types,
                    v_ext,
                    config,
                    layouts,
                    comm=comm,
                )
                compute_field_force(
                    layouts, positions, force_on_grid, field_forces, types, config.n_types
                )
            else:
                kinetic_energy = comm.allreduce(0.5 * config.mass * np.sum(velocities ** 2))
        
        # Print trajectory
        if config.n_print > 0:
            if np.mod(step, config.n_print) == 0 and step != 0:
                frame = step // config.n_print
                if not args.disable_field:
                    (
                        field_energy,
                        kinetic_energy,
                    ) = compute_field_and_kinetic_energy(  # noqa: E501
                        phi,
                        velocities,
                        hamiltonian,
                        positions,
                        types,
                        v_ext,
                        config,
                        layouts,
                        comm=comm,
                    )
                else:
                    kinetic_energy = comm.allreduce(
                        0.5 * config.mass * np.sum(velocities ** 2)
                    )
                temperature = (
                    (2 / 3) * kinetic_energy / (config.R * config.n_particles)
                )
                if args.disable_field:
                    field_energy = 0.0
                if config.pressure:
                    pressure = comp_pressure(
                            phi,
                            hamiltonian,
                            velocities,
                            config,
                            phi_fourier,
                            phi_laplacian,
                            lap_transfer,
                            args,
                            bond_forces,
                            angle_forces,
                            positions,
                            bond_pr_,
                            angle_pr_,
                            comm=comm
                    )
                else:
                    pressure = 0.0 #0.0 indicates not calculated. To be changed.

                store_data(
                    out_dataset,
                    step,
                    frame,
                    indices,
                    positions,
                    velocities,
                    field_forces + bond_forces + angle_forces,
                    config.box_size,
                    temperature,
                    pressure,
                    kinetic_energy,
                    bond_energy,
                    angle_energy,
                    field_energy,
                    config.time_step,
                    config,
                    velocity_out=args.velocity_output,
                    force_out=args.force_output,
                    dump_per_particle=args.dump_per_particle,
                    comm=comm,
                )
                if np.mod(step, config.n_print * config.n_flush) == 0:
                    out_dataset.flush()
        last_step_time = current_step_time

    # End simulation
    if rank == 0:
        end_time = datetime.datetime.now()
        sim_time = end_time - start_time
        setup_time = loop_start_time - start_time
        loop_time = end_time - loop_start_time
        Logger.rank0.log(
            logging.INFO,
            (
                f"Elapsed time: {fmtdt(sim_time)}   "
                f"Setup time: {fmtdt(setup_time)}   "
                f"MD loop time: {fmtdt(loop_time)}"
            ),
        )

    if config.n_print > 0 and np.mod(config.n_steps - 1, config.n_print) != 0:
        if not args.disable_field:
            update_field(
                phi,
                layouts,
                force_on_grid,
                hamiltonian,
                pm,
                positions,
                types,
                config,
                v_ext,
                phi_fourier,
                v_ext_fourier,
                compute_potential=True,
            )
            field_energy, kinetic_energy = compute_field_and_kinetic_energy(
                phi,
                velocities,
                hamiltonian,
                positions,
                types,
                v_ext,
                config,
                layouts,
                comm=comm,
            )
        else:
            kinetic_energy = comm.allreduce(0.5 * config.mass * np.sum(velocities ** 2))
        frame = (step + 1) // config.n_print
        temperature = (2 / 3) * kinetic_energy / (config.R * config.n_particles)  # noqa: E501
        if args.disable_field:
            field_energy = 0.0
        if config.pressure:
            pressure = comp_pressure(
                    phi,
                    hamiltonian,
                    velocities,
                    config,
                    phi_fourier,
                    phi_laplacian,
                    lap_transfer,
                    args,
                    bond_forces,
                    angle_forces,
                    positions,
                    bond_pr_,
                    angle_pr_,
                    comm=comm
            )
        else:
            pressure = 0.0 #0.0 indicates not calculated. To be changed.
        store_data(
            out_dataset,
            step,
            frame,
            indices,
            positions,
            velocities,
            field_forces + bond_forces + angle_forces,
            config.box_size,
            temperature,
            pressure,
            kinetic_energy,
            bond_energy,
            angle_energy,
            field_energy,
            config.time_step,
            config,
            velocity_out=args.velocity_output,
            force_out=args.force_output,
            dump_per_particle=args.dump_per_particle,
            comm=comm,
        )
    out_dataset.close_file()
