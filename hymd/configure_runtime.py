from argparse import ArgumentParser
import os
import sys
import numpy as np
import atexit
import cProfile
import logging
import pstats
from .logger import Logger
from .input_parser import read_config_toml, parse_config_toml


def configure_runtime(comm):
    ap = ArgumentParser()

    ap.add_argument(
        "-v", "--verbose", default=0, type=int, nargs="?",
        help="Increase logging verbosity",
    )
    ap.add_argument(
        "--profile", default=False, action="store_true",
        help="Profile program execution with cProfile",
    )
    ap.add_argument(
        "--disable-field", default=False, action="store_true",
        help="Disable field forces",
    )
    ap.add_argument(
        "--disable-bonds", default=False, action="store_true",
        help="Disable two-particle bond forces",
    )
    ap.add_argument(
        "--disable-angle-bonds", default=False, action="store_true",
        help="Disable three-particle angle bond forces",
    )
    ap.add_argument(
        "--disable-dihedrals", default=False, action="store_true",
        help="Disable four-particle dihedral forces",
    )
    ap.add_argument(
        "--disable-dipole", default=False, action="store_true",
        help="Disable BB dipole calculation",
    )
    ap.add_argument(
        "--double-precision", default=False, action="store_true",
        help="Use double precision positions/velocities",
    )
    ap.add_argument(
        "--double-output", default=False, action="store_true",
        help="Use double precision in output h5md",
    )
    ap.add_argument(
        "--dump-per-particle", default=False, action="store_true",
        help="Log energy values per particle, not total",
    )
    ap.add_argument(
        "--force-output", default=False, action="store_true",
        help="Dump forces to h5md output",
    )
    ap.add_argument(
        "--velocity-output", default=False, action="store_true",
        help="Dump velocities to h5md output",
    )
    ap.add_argument(
        "--disable-mpio", default=False, action="store_true",
        help=(
            "Avoid using h5py-mpi, potentially decreasing IO " "performance"
        ),
    )
    ap.add_argument(
        "--destdir", default=".", help="Write output to specified directory"
    )
    ap.add_argument(
        "--seed", default=None, type=int,
        help="Set the numpy random generator seed for every rank",
    )
    ap.add_argument(
        "--logfile", default="sim.log",
        help="Redirect event logging to specified file",
    )
    ap.add_argument(
        "config", help="Config .py or .toml input configuration script"
    )
    ap.add_argument("input", help="input.hdf5")
    args = ap.parse_args()

    # Given as '--verbose' or '-v' without a specific value specified,
    # default to 1
    if args.verbose is None:
        args.verbose = 1

    if comm.rank == 0:
        os.makedirs(args.destdir, exist_ok=True)
    comm.barrier()

    # Is this used anywhere?
    if args.seed is not None:
        np.random.seed(args.seed)
    else:
        np.random.seed()

    # Setup logger
    Logger.setup(
        default_level=logging.INFO,
        log_file=f"{args.destdir}/{args.logfile}",
        verbose=args.verbose,
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
            logging.INFO,
            f"Attempting to parse config file {args.config} as "".toml",
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
        raise ValueError(
            f"Unable to parse configuration file {args.config}"
            f"\n\ntoml parse traceback:" + repr(ve)
        )
    return args, config
