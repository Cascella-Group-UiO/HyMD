"""Parses and handles the configuration information provided for the simulation
"""
import copy
import tomli
import datetime
import logging
import warnings
import numpy as np
from mpi4py import MPI
from dataclasses import dataclass, field
from typing import List, Union, ClassVar
from .force import Bond, Angle, Dihedral, Chi
from .logger import Logger


@dataclass
class Config:
    """Configuration object

    Handles and verifies the simulation configuration specified in the
    configuration file.

    Attributes
    ----------
    gas_constant : float
        Constant value of the gas constant, R (equivalently the Boltzmann
        constant) in the units used internally in HyMD.
    coulomb_constant : float
        Constant value of the Coulomb constant which converts electric field
        values to forces and electric potential values to energies in the units
        used internally in HyMD.
    n_steps: int
        Number of time steps in the simulation.
    time_step: float
        *Outer* time step used in the simulation. If the rRESPA intergrator is
        used, the *inner* time step (the time step used in the integration of
        intramolecular bending, stretching, and torsional forces) is
        :code:`time_step / respa_inner`.
    box_size : list[float] or (D,) numpy.ndarray
        Simulation box size of simulation in :code:`D` dimensions in units of
        nanometers.
    mesh_size : list[int] or int or numpy.ndarray
        Number of grid points used for the discrete density grid.
    sigma : float
        Filter width representing the effective coarse-graining level of the
        particles in the simulation.
    kappa : float
        Compressibility parameter used in the relaxed incompressibility term in
        the interaction energy functional.
    n_print : int, optional
        Frequency of trajectory/energy output to the H5MD trajectory/energy
        output file (in units of number of time steps).
    tau : float, optional
        The time scale of the CSVR thermostat coupling.
    start_temperature : float, optional
        Generate starting temperature by assigning all particle velocities
        randomly according to the Maxwell-Boltzmann distribution at
        :code:`start_temperature` Kelvin prior to starting the simulation.
    target_temperature : float, optional
        Couple the system to a heat bath at :code:`target_temperature` Kelvin.
    mass : float, optional
        Mass of the particles in the simulation.
    hamiltonian : str, optional
        Specifies the interaction energy functional :math:`W[\\tilde\\phi]`
        for use with the particle-field interactions. Options:
        :code:`SquaredPhi`, :code:`DefaultNohChi`, or :code:`DefaultWithChi`.
    domain_decomposition : int, optional
        Specifies the interval (in time steps) of domain decomposition
        exchange, involving all MPI ranks sending and receiving particles
        according to the particles’ positions in the integration box and the
        MPI ranks’ assigned domain.
    integrator : str, optional
        Specifies the time integrator used in the simulation. Options:
        :code:`velocity-verlet` or :code:`respa`.
    respa_inner : int, optional
        The number of inner time steps in the rRESPA integrator. This denotes
        the number of intramolecular force calculations (stretching, bending,
        torsional) are performed between each impulse applied from the field
        forces.
    file_name : str, optional
        File path of the parsed configuration file.
    name : str, optional
        Name for the simulation.
    tags : list[str], optional
        Tags for the simulation.
    bonds : list[Bond], optional
        Specifies harmonic stretching potentials between particles in the
        same molecule.
    angle_bonds : list[Angle], optional
        Specifies harmonic angular bending potentials between particles in the
        same molecule.
    dihedrals : list[Dihedral], optional
        Specifies four-particle torsional potentials by cosine series.
    chi : list[Chi], optional
        Specifies :math:`\\chi`-interaction parameters between particle
        species.
    n_particles : int, optional
        Specifies the total number of particles in the input. Optional keyword
        for validation, ensuring the input HDF5 topology has the correct number
        of particles and molecules.
    max_molecule_size : int, optional
        Maximum size of any single molecule in the system. Used to speed up
        distribution of particles onto MPI ranks in a parallel fashion.
    n_flush : int, optional
        Frequency of HDF5 write buffer flush, forcing trajectory/energy to be
        written to disk (in units of number of :code:`n_print`).
    thermostat_work : float
        Work performed by the thermostat on the system.
    thermostat_coupling_groups : list[str], optional
        Specifies individual groups coupling independently to the CSVR
        thermostat. E.g. in a system containing :code:`"A"`, :code:`"B"`, and
        :code:`"C"` type particles,
        :code:`thermostat_coupling_groups = [["A", "B"], ["C"],]` would
        thermalise types :code:`"A"` and :code:`"B"` together and couple
        :code:`"C"` type particles to a different thermostat (all individual
        thermostats are at the same temperature, i.e. target_temperature
        Kelvin).
    initial_energy : float
        Value of the total energy prior to the start of the simulation.
    cancel_com_momentum : int, optional
        If :code:`True`, the total linear momentum of the center of mass is
        removed before starting the simulation. If an integer is specifed, the
        total linear momentum of the center of mass is removed every
        :code:`remove_com_momentum` time steps. If :code:`False`, the linear
        momentum is never removed.
    coulombtype : str, optional
        Specifies the type of electrostatic Coulomb interactions in the system.
        The strength of the electrostatic forces is modulated by the relative
        dielectric constant of the simulation medium, specified with the
        :code:`dielectric_const` keyword. Charges for individual particles are
        specified in the structure/topology HDF5 input file, not in the
        configuration file. If no charges (or peptide backbone dipoles) are
        present, the electrostatic forces will not be calculated even if this
        keyword is set to :code:`"PIC_Spectral"`.
    dielectric_const : float, optional
        Specifies the relative dielectric constant of the simulation medium
        which regulates the strength of the electrostatic interactions. When
        using helical propensity dihedrals, this keyword must be specified—even
        if electrostatics are not included with the :code:`coulombtype`
        keyword.

    See also
    --------
    hymd.input_parser.Bond :
        Two-particle bond type dataclass.
    hymd.input_parser.Angle :
        Three-particle bond type dataclass.
    hymd.input_parser.Dihedral :
        Four-particle bond type dataclass.
    """
    gas_constant: ClassVar[float] = 0.0083144621  # kJ mol-1 K-1
    coulomb_constant: ClassVar[float] = 138.935458  # kJ mn mol-1 e-2

    n_steps: int
    time_step: float
    box_size: Union[List[float], np.ndarray]
    mesh_size: Union[Union[List[int], np.ndarray], int]
    sigma: float
    kappa: float

    n_print: int = None
    tau: float = None
    start_temperature: Union[float, bool] = None
    target_temperature: Union[float, bool] = None
    mass: float = None
    hamiltonian: str = None
    domain_decomposition: Union[int, bool] = None
    integrator: str = None
    respa_inner: int = 1
    file_name: str = "<config file path unknown>"
    name: str = None
    tags: List[str] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    angle_bonds: List[Angle] = field(default_factory=list)
    dihedrals: List[Dihedral] = field(default_factory=list)
    chi: List[Chi] = field(default_factory=list)
    n_particles: int = None
    max_molecule_size: int = None
    n_flush: int = None
    thermostat_work: float = 0.0
    thermostat_coupling_groups: List[List[str]] = field(default_factory=list)
    initial_energy: float = None
    cancel_com_momentum: Union[int, bool] = False
    coulombtype: str = None
    dielectric_const: float = None

    def __str__(self):
        bonds_str = "\tbonds:\n" + "".join(
            [
                (
                    f"\t\t{k.atom_1} {k.atom_2}: "
                    f"{k.equilibrium}, {k.strength}\n"
                )
                for k in self.bonds
            ]
        )
        angle_str = "\tangle_bonds:\n" + "".join(
            [
                (
                    f"\t\t{k.atom_1} {k.atom_2} {k.atom_3}: "
                    + f"{k.equilibrium}, {k.strength}\n"
                )
                for k in self.angle_bonds
            ]
        )
        dihedrals_str = "\tdihedrals:\n" + "".join(
            [
                (
                    f"\t\t{k.atom_1} {k.atom_2} {k.atom_3} {k.atom_4}: "
                    # This might need to be fixed/made prettier, probably
                    # there's an easier way
                    + (
                        "\n\t\t"
                        + " " * len(
                            f"{k.atom_1} {k.atom_2} {k.atom_3} {k.atom_4}: "
                        )
                    ).join(
                        map(
                            str,
                            [
                                [round(num, 3) for num in c_in]
                                if isinstance(c_in, list)
                                else c_in
                                for c_in in k.coeffs
                            ],
                        )
                    )
                    + (
                        "\n\t\t"
                        + " " * len(
                            f"{k.atom_1} {k.atom_2} {k.atom_3} {k.atom_4}: "
                        )
                    )
                    + f"dih_type = {k.dih_type}\n"
                )
                for k in self.dihedrals
            ]
        )
        chi_str = "\tchi:\n" + "".join(
            [
                (f"\t\t{k.atom_1} {k.atom_2}: " + f"{k.interaction_energy}\n")
                for k in self.chi
            ]
        )
        thermostat_coupling_groups_str = ""
        if any(self.thermostat_coupling_groups):
            thermostat_coupling_groups_str = (
                "\tthermostat_coupling_groups:\n"
                + "".join(
                    [
                        "\t\t" + ", ".join([f"{n}" for n in ng]) + "\n"
                        for ng in self.thermostat_coupling_groups
                    ]
                )
            )

        ret_str = f'\n\n\tConfig: {self.file_name}\n\t{50 * "-"}\n'
        for k, v in self.__dict__.items():
            if k not in (
                "bonds",
                "angle_bonds",
                "dihedrals",
                "chi",
                "thermostat_coupling_groups",
            ):
                ret_str += f"\t{k}: {v}\n"
        ret_str += (
            bonds_str
            + angle_str
            + dihedrals_str
            + chi_str
            + thermostat_coupling_groups_str
        )
        return ret_str


def read_config_toml(file_path):
    with open(file_path, "r") as in_file:
        toml_content = in_file.read()
    return toml_content


def propensity_potential_coeffs(x: float, comm):
    alpha_coeffs = np.array(
        [
            [7.406, -5.298, -2.570, 1.336, 0.739],
            [-0.28632126, 1.2099146, 1.18122138, 0.49075168, 0.98495911],
        ]
    )
    beta_coeffs = np.array(
        [
            [3.770, 5.929, -4.151, -0.846, 0.190],
            [-0.2300693, -0.0583289, 0.99342396, 1.03237971, 2.90160988],
        ]
    )
    coil_coeffs = np.array(
        [
            [1.416, -0.739, 0.990, -0.397, 0.136],
            [1.3495933, 0.45649087, 2.30441057, -0.12274901, -0.26179939],
        ]
    )

    zero_add = np.zeros((2, 5))
    if x == -1:
        return np.concatenate((alpha_coeffs, zero_add))
    elif x == 0:
        return np.concatenate((coil_coeffs, zero_add))
    elif x == 1:
        return np.concatenate((beta_coeffs, zero_add))

    abs_x = np.abs(x)
    if abs_x > 1:
        err_str = (
            f"The provided value of λ = {x} is out of λ definition range, "
            f"[-1.0, 1.0]."
        )
        Logger.rank0.log(logging.ERROR, err_str)
        if comm.Get_rank() == 0:
            raise ValueError(err_str)

    else:
        coil_coeffs[0] *= 1 - abs_x
        if x < 0:
            alpha_coeffs[0] *= 0.5 * (abs_x - x)
            return np.concatenate((alpha_coeffs, coil_coeffs))
        else:
            beta_coeffs[0] *= 0.5 * (abs_x + x)
            return np.concatenate((beta_coeffs, coil_coeffs))


def parse_config_toml(toml_content, file_path=None, comm=MPI.COMM_WORLD):
    parsed_toml = tomli.loads(toml_content)
    config_dict = {}

    # Defaults = None
    for n in (
        "n_print",
        "tau",
        "start_temperature",
        "target_temperature",
        "mass",
        "hamiltonian",
        "domain_decomposition",
        "integrator",
        "name",
        "n_particles",
        "max_molecule_size",
        "coulombtype",
        "dielectric_const",
        "n_flush",
    ):
        config_dict[n] = None

    # Defaults = []
    for n in ("bonds", "angle_bonds", "dihedrals", "chi", "tags"):
        config_dict[n] = []

    # Flatten the .toml dictionary, ignoring the top level [tag] directives (if
    # any).
    for k, v in parsed_toml.items():
        if isinstance(v, dict):
            for nested_k, nested_v in v.items():
                config_dict[nested_k] = nested_v
        else:
            config_dict[k] = v
    for k, v in config_dict.items():
        if k == "bonds":
            config_dict["bonds"] = [None] * len(v)
            for i, b in enumerate(v):
                config_dict["bonds"][i] = Bond(
                    atom_1=b[0],
                    atom_2=b[1],
                    equilibrium=b[2],
                    strength=b[3],
                )
        if k == "angle_bonds":
            config_dict["angle_bonds"] = [None] * len(v)
            for i, b in enumerate(v):
                config_dict["angle_bonds"][i] = Angle(
                    atom_1=b[0],
                    atom_2=b[1],
                    atom_3=b[2],
                    equilibrium=b[3],
                    strength=b[4],
                )
        if k == "dihedrals":
            config_dict["dihedrals"] = [None] * len(v)
            for i, b in enumerate(v):
                try:
                    dih_type = int(b[2][0])
                except IndexError:
                    Logger.rank0.log(
                        logging.WARNING,
                        "Dihedral type not provided, defaulting to 0."
                    )
                    dih_type = 0

                    # Probably it's better to move this in check_dihedrals?
                    wrong_len = len(b[1]) not in (1, 2)
                    wrong_type_1 = len(b[1]) == 1 and not isinstance(
                        b[1][0], float
                    )
                    wrong_type_2 = len(b[1]) == 2 and not isinstance(
                        b[1][0], list
                    )
                    if wrong_len or wrong_type_1 or wrong_type_2:
                        err_str = (
                            "The coefficients specified for the dihedral type "
                            "(0) do not match the correct structure. Either "
                            "use [lambda] or [[cn_prop], [dn_prop]], or select"
                            " the correct dihedral type."
                        )
                        Logger.rank0.log(logging.ERROR, err_str)
                        if comm.Get_rank() == 0:
                            raise RuntimeError(err_str)

                # FIXME: this is messy af, I don't like it
                if dih_type == 0 and isinstance(b[1][0], (float, int)):
                    coeff = propensity_potential_coeffs(b[1][0], comm)
                elif dih_type == 1 and len(b[1]) == 3:
                    coeff = np.array(
                        propensity_potential_coeffs(b[1][0][0], comm).tolist()
                        + b[1][1:]
                    )
                elif dih_type == 2:
                    coeff = np.array(b[1])
                else:
                    coeff = np.insert(
                        np.array(b[1]), 2, np.zeros((2, 5)), axis=0
                    )

                config_dict["dihedrals"][i] = Dihedral(
                    atom_1=b[0][0],
                    atom_2=b[0][1],
                    atom_3=b[0][2],
                    atom_4=b[0][3],
                    coeffs=coeff,
                    dih_type=dih_type,
                )
        # if k == "improper dihedrals":
        #     config_dict["improper dihedrals"] = [None] * len(v)
        # ...
        if k == "chi":
            config_dict["chi"] = [None] * len(v)
            for i, c in enumerate(v):
                c_ = sorted([c[0], c[1]])
                config_dict["chi"][i] = Chi(
                    atom_1=c_[0], atom_2=c_[1], interaction_energy=c[2]
                )

    if file_path is not None:
        config_dict["file_name"] = file_path

    for n in (
        "n_steps", "time_step", "box_size", "mesh_size", "sigma", "kappa"
    ):
        if n not in config_dict:
            err_str = (
                f"No {n} specified in config file {file_path}. Unable to start"
                f" simulation."
            )
            Logger.rank0.log(logging.ERROR, err_str)
            if comm.Get_rank() == 0:
                raise ValueError(err_str)
    return Config(**config_dict)


def check_n_particles(config, indices, comm=MPI.COMM_WORLD):
    n_particles = comm.allreduce(len(indices), MPI.SUM)
    if config.n_particles is None:
        config = copy.deepcopy(config)
        config.n_particles = n_particles
        info_str = (
            f"No n_particles found in toml file {config.file_name}, defaulting"
            f" to indices.shape ({n_particles})"
        )
        Logger.rank0.log(logging.INFO, info_str)
        return config

    if n_particles != config.n_particles:
        warn_str = (
            f"n_particles in {config.file_name} ({config.n_particles}) does "
            "not match the shape of the indices array in the .HDF5 file "
            f"({n_particles}). Defaulting to using indices.shape "
            f"({n_particles})"
        )
        Logger.rank0.log(logging.WARNING, warn_str)
        if comm.Get_rank() == 0:
            warnings.warn(warn_str)
        config = copy.deepcopy(config)
        config.n_particles = n_particles
    return config


def check_max_molecule_size(config, comm=MPI.COMM_WORLD):
    if config.max_molecule_size is None:
        info_str = (
            f"No max_molecule_size found in toml file {config.file_name}, "
            f"defaulting to 201"
        )
        Logger.rank0.log(logging.INFO, info_str)
        config = copy.deepcopy(config)
        config.max_molecule_size = 201
        return config

    if config.max_molecule_size < 1:
        warn_str = (
            f"max_molecule_size in {config.file_name} must be a positive "
            f"integer, not {config.max_molecule_size}, defaulting to 201"
        )
        Logger.rank0.log(logging.WARNING, warn_str)
        if comm.Get_rank() == 0:
            warnings.warn(warn_str)
        config = copy.deepcopy(config)
        config.max_molecule_size = 201
        return config
    return config


def _find_unique_names(config, names, comm=MPI.COMM_WORLD):
    unique_names = np.unique(names)
    receive_buffer = comm.gather(unique_names, root=0)

    gathered_unique_names = None
    if comm.Get_rank() == 0:
        gathered_unique_names = np.unique(np.concatenate(receive_buffer))
    unique_names = comm.bcast(gathered_unique_names, root=0)
    unique_names = sorted([n.decode("UTF-8") for n in unique_names])
    config.unique_names = unique_names
    config.n_types = len(unique_names)
    return config


def _setup_type_to_name_map(config, names, types, comm=MPI.COMM_WORLD):
    if not hasattr(config, "unique_names"):
        config = _find_unique_names(config, names)
    name_to_type_ = {}
    for n, t in zip(names, types):
        n = n.decode("utf-8")
        if n not in name_to_type_:
            name_to_type_[n] = t
    receive_buffer = comm.gather(name_to_type_, root=0)
    gathered_dict = None
    if comm.Get_rank() == 0:
        gathered_dict = {}
        for d in receive_buffer:
            for k, v in d.items():
                if k not in gathered_dict:
                    gathered_dict[k] = v
                else:
                    assert v == gathered_dict[k]
    name_to_type_map = comm.bcast(gathered_dict, root=0)
    config.name_to_type_map = name_to_type_map
    config.type_to_name_map = {v: k for k, v in name_to_type_map.items()}
    return config


def check_bonds(config, names, comm=MPI.COMM_WORLD):
    if not hasattr(config, "unique_names"):
        config = _find_unique_names(config, names)
    unique_names = config.unique_names

    for b in config.bonds:
        if b.atom_1 not in unique_names or b.atom_2 not in unique_names:
            missing_str = ""
            if b.atom_1 not in unique_names:
                if b.atom_2 not in unique_names:
                    if b.atom_1 == b.atom_2:
                        missing_str = f"no {b.atom_1} atoms"
                    else:
                        missing_str = f"neither {b.atom_1}, {b.atom_2} atoms"
                else:
                    missing_str = f"no {b.atom_1} atoms"
            else:
                missing_str = f"no {b.atom_2} atoms"

            warn_str = (
                f"Bond type {b.atom_1}--{b.atom_2} specified in "
                f"{config.file_name} but {missing_str} are present in the "
                f"specified system (names array)"
            )
            Logger.rank0.log(logging.WARNING, warn_str)
            if comm.Get_rank() == 0:
                warnings.warn(warn_str)
    return config


def check_angles(config, names, comm=MPI.COMM_WORLD):
    if not hasattr(config, "unique_names"):
        config = _find_unique_names(config, names)
    unique_names = config.unique_names

    for a in config.angle_bonds:
        if (
            a.atom_1 not in unique_names
            or a.atom_2 not in unique_names
            or a.atom_3 not in unique_names
        ):
            missing = [
                a.atom_1 not in unique_names,
                a.atom_2 not in unique_names,
                a.atom_3 not in unique_names,
            ]
            missing_names = [
                atom
                for i, atom in enumerate([a.atom_1, a.atom_2, a.atom_3])
                if missing[i]
            ]
            missing_str = ", ".join(np.unique(missing_names))

            warn_str = (
                f"Angle bond type {a.atom_1}--{a.atom_2}--{a.atom_3} "
                f"specified in {config.file_name} but no {missing_str} atoms "
                f"are present in the specified system (names array)"
            )
            Logger.rank0.log(logging.WARNING, warn_str)
            if comm.Get_rank() == 0:
                warnings.warn(warn_str)
    return config


def check_dihedrals(config, names, comm=MPI.COMM_WORLD):
    if not hasattr(config, "unique_names"):
        config = _find_unique_names(config, names)
    unique_names = config.unique_names

    for d in config.dihedrals:
        if (
            d.atom_1 not in unique_names
            or d.atom_2 not in unique_names
            or d.atom_3 not in unique_names
            or d.atom_4 not in unique_names
        ):
            missing = [
                d.atom_1 not in unique_names,
                d.atom_2 not in unique_names,
                d.atom_3 not in unique_names,
                d.atom_4 not in unique_names,
            ]
            missing_names = [
                atom
                for i, atom in enumerate(
                    [d.atom_1, d.atom_2, d.atom_3, d.atom_4]
                )
                if missing[i]
            ]
            missing_str = ", ".join(np.unique(missing_names))

            warn_str = (
                f"Dihedral type {d.atom_1}--{d.atom_2}--{d.atom_3}--{d.atom_4}"
                f" specified in {config.file_name} but no {missing_str} atoms "
                f"are present in the specified system (names array)"
            )
            Logger.rank0.log(logging.WARNING, warn_str)
            if comm.Get_rank() == 0:
                warnings.warn(warn_str)
    return config


def check_chi(config, names, comm=MPI.COMM_WORLD):
    if not hasattr(config, "unique_names"):
        config = _find_unique_names(config, names)
    unique_names = config.unique_names

    for c in config.chi:
        if c.atom_1 not in unique_names or c.atom_2 not in unique_names:
            missing_str = ""
            if c.atom_1 not in unique_names:
                if c.atom_2 not in unique_names:
                    if c.atom_1 == c.atom_2:
                        missing_str = f"no {c.atom_1} atoms"
                    else:
                        missing_str = f"neither {c.atom_1}, {c.atom_2} atoms"
                else:
                    missing_str = f"no {c.atom_1} atoms"
            else:
                missing_str = f"no {c.atom_2} atoms"

            warn_str = (
                f"Chi interaction {c.atom_1}--{c.atom_2} specified in "
                f"{config.file_name} but {missing_str} are present in the "
                f"specified system (names array)"
            )
            Logger.rank0.log(logging.WARNING, warn_str)
            if comm.Get_rank() == 0:
                warnings.warn(warn_str)

    for i, n in enumerate(unique_names):
        for m in unique_names[i+1:]:
            found = False
            for c in config.chi:
                if (c.atom_1 == n and c.atom_2 == m) or (
                    c.atom_1 == m and c.atom_2 == n
                ):
                    found = True
            if not found:
                config.chi.append(
                    Chi(atom_1=n, atom_2=m, interaction_energy=0.0)
                )
                warn_str = (
                    f"Atom types {n} and {m} found in the "
                    f"system, but no chi interaction {n}--{m} "
                    f"specified in {config.file_name}. Defaulting to "
                    f"chi[{n}, {m}] = 0"
                )
                Logger.rank0.log(logging.WARNING, warn_str)
                if comm.Get_rank() == 0:
                    warnings.warn(warn_str)
    return config


def check_box_size(config, comm=MPI.COMM_WORLD):
    for b in config.box_size:
        if b <= 0.0:
            err_str = (
                f"Invalid box size specified in {config.file_name}: "
                f"{config.box_size}"
            )
            Logger.rank0.log(logging.ERROR, err_str)
            if comm.Get_rank() == 0:
                raise ValueError(err_str)
    config.box_size = np.array(config.box_size, dtype=np.float64)
    return config


def check_integrator(config, comm=MPI.COMM_WORLD):
    if config.integrator.lower() not in ("velocity-verlet", "respa"):
        err_str = (
            f"Invalid integrator specified in {config.file_name}: "
            f'{config.integrator}. Available options "velocity-verlet" or '
            f'"respa".'
        )
        Logger.rank0.log(logging.ERROR, err_str)
        if comm.Get_rank() == 0:
            raise ValueError(err_str)

    if config.integrator.lower() == "respa":
        if isinstance(config.respa_inner, float):
            warn_str = (
                f"Number of inner rRESPA time steps in "
                f"{config.file_name}: {config.respa_inner} specified "
                f"as float, using {int(config.respa_inner)}"
            )
            Logger.rank0.log(logging.WARNING, warn_str)
            if comm.Get_rank() == 0:
                warnings.warn(warn_str)
            config.respa_inner = int(config.respa_inner)
        elif not isinstance(config.respa_inner, int):
            err_str = (
                f"Invalid number of inner rRESPA time steps in "
                f"{config.file_name}: {config.respa_inner}. Must be "
                f"positive integer"
            )
            Logger.rank0.log(logging.ERROR, err_str)
            raise TypeError(err_str)

        if config.respa_inner <= 0:
            err_str = (
                f"Invalid number of inner rRESPA time steps in "
                f"{config.file_name}: {config.respa_inner}. Must be "
                f"positive integer"
            )
            Logger.rank0.log(logging.ERROR, err_str)
            raise ValueError(err_str)

    if (
        config.integrator.lower() == "velocity-verlet"
        and config.respa_inner != 1
    ):
        warn_str = (
            f"Integrator type Velocity-Verlet specified in {config.file_name} "
            f"and inner rRESPA time steps set to {config.respa_inner}. "
            f"Using respa_inner = 1"
        )
        Logger.rank0.log(logging.WARNING, warn_str)
        if comm.Get_rank() == 0:
            warnings.warn(warn_str)
        config.respa_inner = 1

    return config


def check_hamiltonian(config, comm=MPI.COMM_WORLD):
    valid_hamiltonians = ["defaultnochi", "defaultwithchi", "squaredphi"]

    if config.hamiltonian is None:
        if len(config.chi) == 0:
            warn_str = (
                f"No hamiltonian form and no chi interactions specified in "
                f"{config.file_name}, defaulting to DefaultNoChi hamiltonian"
            )
            Logger.rank0.log(logging.WARNING, warn_str)
            if comm.Get_rank() == 0:
                warnings.warn(warn_str)
            config.hamiltonian = "DefaultNoChi"
        else:
            warn_str = (
                f"No hamiltonian form specified in {config.file_name}, but "
                f"chi interactions are specified, defaulting to "
                f"DefaultWithChi hamiltonian"
            )
            Logger.rank0.log(logging.WARNING, warn_str)
            if comm.Get_rank() == 0:
                warnings.warn(warn_str)
            config.hamiltonian = "DefaultWithChi"
    elif config.hamiltonian.lower() not in valid_hamiltonians:
        err_str = (
            f"The specified Hamiltonian {config.hamiltonian} was not "
            f"recognized as a valid Hamiltonian."
        )
        Logger.rank0.log(logging.ERROR, err_str)
        raise NotImplementedError(err_str)

    return config


def check_n_flush(config, comm=MPI.COMM_WORLD):
    if config.n_print:
        if config.n_flush is None:
            config.n_flush = 10000 // config.n_print
    return config


def check_n_print(config, comm=MPI.COMM_WORLD):
    if (not isinstance(config.n_print, int) and
        not isinstance(config.n_print, float) and
        config.n_print is not None):
        err_str = (
            f"invalid value for n_print ({config.n_print})"
        )
        Logger.rank0.log(logging.ERROR, err_str)
        raise RuntimeError(err_str)        
    
    if config.n_print is None or config.n_print <= 0:
        config.n_print = False
    elif not isinstance(config.n_print, int):
        if isinstance(config.n_print, float):
            warn_str = (
                f"n_print is a float ({config.n_print}), not int, using "
                f"{int(round(config.n_print))}"
            )
            Logger.rank0.log(logging.WARNING, warn_str)
            if comm.Get_rank() == 0:
                warnings.warn(warn_str)

            config.n_print = int(round(config.n_print))
        else:
            err_str = (
                f"invalid value for n_print ({config.n_print})"
            )
            Logger.rank0.log(logging.ERROR, err_str)
            raise RuntimeError(err_str)
    return config


def check_tau(config, comm=MPI.COMM_WORLD):
    if config.tau is None and config.target_temperature is not None:
        warn_str = "target temp specified but no tau, defaulting 0.7"
        config.tau = 0.7
        Logger.rank0.log(logging.WARNING, warn_str)
        if comm.Get_rank() == 0:
            warnings.warn(warn_str)
    return config


def check_start_and_target_temperature(config, comm=MPI.COMM_WORLD):
    """Validate provided starting and target thermostat temperatures

    Assesses the provided temperature target and ensures it is a non-negative
    floating point number or :code:`False`. Ensures the starting temperature is
    a non-negative floating point number or :code:`False`.

    If the value for either is :code:`None`, the returned configuration object
    has the values defaulted to :code:`False` in each case.

    Parameters
    ----------
    config : Config
        Configuration object.

    Returns
    -------
    validated_config : Config
        Configuration object with validated :code:`target_temperature` and
        :code:`start_temperature`.
    """
    for t in ("start_temperature", "target_temperature"):
        if getattr(config, t) is not None:
            try:
                if getattr(config, t) < 0:
                    warn_str = (
                        f"{t} set to negative value ({getattr(config, t)}), "
                        f"defaulting to False"
                    )
                    setattr(config, t, False)
                    Logger.rank0.log(logging.WARNING, warn_str)
                    if comm.Get_rank() == 0:
                        warnings.warn(warn_str)
            except TypeError as e:
                err_str = (
                    f"Could not interpret {t} = {repr(getattr(config, t))} as "
                    f"a number."
                )
                raise TypeError(err_str) from e

    if config.start_temperature is None:
        config.start_temperature = False
    if config.target_temperature is None:
        config.target_temperature = False
    return config


def check_mass(config, comm=MPI.COMM_WORLD):
    if config.mass is None:
        info_str = "no mass specified, defaulting to 72.0"
        config.mass = 72.0
        Logger.rank0.log(logging.INFO, info_str)
    elif not isinstance(config.mass, int) and not isinstance(config.mass, float):
        err_str = (
            f"specified mass is invalid type {config.mass}, "
            f"({type(config.mass)})"
        )
        Logger.rank0.log(logging.ERROR, err_str)
        raise TypeError(err_str)
    elif config.mass < 0:
        err_str = f"invalid mass specified, {config.mass}"
        Logger.rank0.log(logging.ERROR, err_str)
        raise ValueError(err_str)
    else:
        config.mass = float(config.mass)

    return config


def check_domain_decomposition(config, comm=MPI.COMM_WORLD):
    dd = config.domain_decomposition
    if dd is None or dd is False:
        config.domain_decomposition = False
        return config

    if isinstance(dd, int):
        if dd < 0:
            warn_str = "negative domain_decomposition specified, using False"
            config.domain_decomposition = False
            Logger.rank0.log(logging.WARNING, warn_str)
            if comm.Get_rank() == 0:
                warnings.warn(warn_str)
    elif isinstance(dd, float):
        if dd < 0.0:
            warn_str = "negative domain_decomposition specified, using False"
            config.domain_decomposition = False
        else:
            warn_str = (
                f"domain_decomposition ({config.domain_decomposition}) is not "
                f"an integer, using {int(round(dd))}"
            )
            config.domain_decomposition = int(round(dd))
        Logger.rank0.log(logging.WARNING, warn_str)
        if comm.Get_rank() == 0:
            warnings.warn(warn_str)
    else:
        err_str = (f"invalid value for domain_decomposition "
                   f"({config.domain_decomposition}) use an integer")
        Logger.rank0.log(logging.ERROR, err_str)
        raise ValueError(err_str)

    return config


def check_name(config, comm=MPI.COMM_WORLD):
    if config.name is None:
        root_current_time = ""
        if comm.Get_rank() == 0:
            root_current_time = datetime.datetime.now().strftime(
                "%m/%d/%Y, %H:%M:%S"
            )
        current_time = comm.bcast(root_current_time, root=0)

        if config.name is None:
            config.name = "sim" + current_time
    else:
        config.name = str(config.name)
    return config


def check_thermostat_coupling_groups(config, comm=MPI.COMM_WORLD):
    if any(config.thermostat_coupling_groups):
        found = [0 for _ in config.unique_names]
        unique_names = [n for n in config.unique_names]
        for i, group in enumerate(config.thermostat_coupling_groups):
            for species in group:
                try:
                    ind = unique_names.index(species)
                    found[ind] += 1
                except ValueError as e:
                    err_str = (
                        f"Particle species {species} specified in thermostat "
                        f"coupling group {i}, but no {species} particles were "
                        "found in the system."
                    )
                    raise ValueError(err_str) from e
        if any([True if f > 1 else False for f in found]):
            for i, f in enumerate(found):
                if f > 1:
                    species = unique_names[i]
                    err_str = (
                        f"Particle species {species} specified in multiple "
                        "thermostat coupling groups; "
                        f"{config.thermostat_coupling_groups}."
                    )
                    raise ValueError(err_str)
        if not all([True if f == 1 else False for f in found]):
            for i, f in enumerate(found):
                if f != 1:
                    species = unique_names[i]
                    err_str = (
                        f"Particle species {species} not specified in any "
                        f"thermostat coupling group, but {species} particles "
                        "were found in the system"
                    )
                    raise ValueError(err_str)
    return config


def check_cancel_com_momentum(config, comm=MPI.COMM_WORLD):
    if isinstance(config.cancel_com_momentum, int):
        if config.cancel_com_momentum == 0 or config.cancel_com_momentum < 0:
            config.cancel_com_momentum = False
    elif not isinstance(config.cancel_com_momentum, bool):
        err_str = (
            f"Could not interpret {config.cancel_com_momentum} as an integer "
            f"or boolean."
        )
        raise ValueError(err_str)
    return config


def check_config(config, indices, names, types, comm=MPI.COMM_WORLD):
    """Performs various checks on the specfied config to ensure consistency

    Parameters
    ----------
    config : Config
        Configuration object.
    indices : (N,) numpy.ndarray
        Array of integer indices for :code:`N` particles.
    names : (N,) numpy.ndarray
        Array of string names for :code:`N` particles.
    types : (N,) numpy.ndarray
        Array of integer type indices for :code:`N` particles.
    comm : mpi4py.Comm, optional
        MPI communicator, defaults to :code:`mpi4py.COMM_WORLD`.

    Returns
    -------
    config : Config
        Validated configuration object.
    """
    config.box_size = np.array(config.box_size)
    config = _find_unique_names(config, names, comm=comm)
    if types is not None:
        config = _setup_type_to_name_map(config, names, types, comm=comm)
    config = check_box_size(config, comm=comm)
    config = check_integrator(config, comm=comm)
    config = check_max_molecule_size(config, comm=comm)
    config = check_tau(config, comm=comm)
    config = check_start_and_target_temperature(config, comm=comm)
    config = check_mass(config, comm=comm)
    config = check_domain_decomposition(config, comm=comm)
    config = check_name(config, comm=comm)
    config = check_n_particles(config, indices, comm=comm)
    config = check_chi(config, names, comm=comm)
    config = check_bonds(config, names, comm=comm)
    config = check_angles(config, names, comm=comm)
    config = check_dihedrals(config, names, comm=comm)
    config = check_hamiltonian(config, comm=comm)
    config = check_thermostat_coupling_groups(config, comm=comm)
    config = check_cancel_com_momentum(config, comm=comm)
    config = check_n_print(config, comm=comm)
    config = check_n_flush(config, comm=comm)
    return config


def check_charges(charges, comm=MPI.COMM_WORLD):
    """Check if charges across ranks sum to zero.

    Parameters
    ----------
    charges : (N,) numpy.ndarray
        Array of floats with charges for :code:`N` particles.
    comm : mpi4py.Comm, optional
        MPI communicator, defaults to :code:`mpi4py.COMM_WORLD`.
    """  
    total_charge = comm.allreduce(np.sum(charges), MPI.SUM)

    if not np.isclose(total_charge, 0.):
        warn_str = (
            f"Charges in the input file do not sum to zero. "
            f"Total charge is {total_charge}."
        )
        Logger.rank0.log(logging.WARNING, warn_str)
        if comm.Get_rank() == 0:
            warnings.warn(warn_str)

