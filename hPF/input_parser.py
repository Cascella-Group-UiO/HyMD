import copy
import toml
import logging
import warnings
import numpy as np
from mpi4py import MPI
from dataclasses import dataclass, field
from typing import List, Union
from force import Bond, Angle, Chi
from logger import Logger


@dataclass
class Config:
    mass: float
    n_steps: int
    n_print: int
    time_step: float
    box_size: Union[List[float], np.ndarray]
    integrator: str
    mesh_size: Union[Union[List[int], np.ndarray], int]

    kappa: float = 0.05
    respa_inner: int = 1
    file_name: str = '<config-file>'
    name: str = None
    tags: List[str] = field(default_factory=list)
    chi: List[Chi] = field(default_factory=list)
    angle_bonds: List[Angle] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    n_particles: int = None
    max_molecule_size: int = None

    def __str__(self):
        bonds_str = '\tbonds:\n' + ''.join([
            (f'\t\t{k.atom_1} {k.atom_2}: ' +
             f'{k.equilibrium}, {k.strenght}\n')
            for k in self.bonds
        ])
        angle_str = '\tangle_bonds:\n' + ''.join([
            (f'\t\t{k.atom_1} {k.atom_2} {k.atom_3}: ' +
             f'{k.equilibrium}, {k.strenght}\n')
            for k in self.angle_bonds
        ])
        chi_str = '\tchi:\n' + ''.join([
            (f'\t\t{k.atom_1} {k.atom_2}: ' +
             f'{k.interaction_energy}\n')
            for k in self.chi
        ])

        ret_str = f'\n\n\tConfig: {self.file_name}\n\t{50 * "-"}\n'
        for k, v in self.__dict__.items():
            if k not in ('bonds', 'angle_bonds', 'chi'):
                ret_str += f'\t{k}: {v}\n'
        ret_str += bonds_str + angle_str + chi_str
        return ret_str


def convert_CONF_to_config(CONF, file_path=None):
    # Name in CONF.py, name in class Config, default if not present in CONF.py
    vars_names_defaults = [
        ('mass', 'mass', 72.0),
        ('NSTEPS', 'n_steps', 1),
        ('nprint', 'n_print', -1),
        ('dt', 'time_step', 0.03),
        ('L', 'box_size', [1.0, 1.0, 1.0]),
        ('Nv', 'mesh_size', 50),
        ('Np', 'n_particles', -1)
    ]
    config_dict = {}
    for x in vars_names_defaults:
        CONF_name = x[0]
        config_name = x[1]
        default = x[2]
        config_dict[config_name] = (
            CONF[CONF_name] if CONF_name in CONF else default
        )
        CONF.pop(CONF_name)

    if file_path is not None:
        config_dict['file_path'] = file_path
    if 'respa' in CONF or 'RESPA' in CONF:
        config_dict['integrator'] = 'respa'
        config_dict['respa_inner'] = (
            CONF['respa_inner'] if 'respa_inner' in CONF else 1
        )




def read_config_toml(file_path):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("read_config")
        print(logging.getLogger('root'))
        print(hex(id(logging.getLogger('root'))))
    with open(file_path, 'r') as in_file:
        toml_content = in_file.read()
    return toml_content


def parse_config_toml(toml_content, file_path=None):
    parsed_toml = toml.loads(toml_content)
    config_dict = {}

    # Flatten the .toml dictionary, ignoring the top level [tag] directives (if
    # any).
    for k, v in parsed_toml.items():
        if isinstance(v, dict):
            for nested_k, nested_v in v.items():
                config_dict[nested_k] = nested_v

    bonds = []
    angle_bonds = []
    chi = []
    for k, v in config_dict.items():
        if k == 'bonds':
            bonds = [None] * len(v)
            for i, b in enumerate(v):
                bonds[i] = Bond(atom_1=b[0][0], atom_2=b[0][1],
                                equilibrium=b[1][0], strenght=b[1][1])
        if k == 'angle_bonds':
            angle_bonds = [None] * len(v)
            for i, b in enumerate(v):
                angle_bonds[i] = Angle(atom_1=b[0][0], atom_2=b[0][1],
                                       atom_3=b[0][2], equilibrium=b[1][0],
                                       strenght=b[1][1])
        if k == 'chi':
            chi = [None] * len(v)
            for i, c in enumerate(v):
                chi[i] = Chi(atom_1=c[0][0], atom_2=c[0][1],
                             interaction_energy=c[1][0])
    for k in ('bonds', 'angle_bonds', 'chi'):
        if k in config_dict:
            config_dict.pop(k)
    if file_path is not None:
        config_dict['file_path'] = file_path
    return Config(bonds=bonds, angle_bonds=angle_bonds, chi=chi, **config_dict)


def check_n_particles(config, indices):
    n_particles = MPI.COMM_WORLD.allreduce(len(indices), MPI.SUM)
    if config.n_particles is None:
        config = copy.deepcopy(config)
        config.n_particles = n_particles
        info_str = (
            f'No n_particles found in toml file {config.file_name}, defaulting'
            f' to indices.shape ({n_particles})'
        )
        Logger.rank0.log(logging.INFO, info_str)
        return config

    if n_particles != config.n_particles:
        warn_str = (
            f'n_particles in {config.file_name} ({config.n_particles}) does '
            'not match the shape of the indices array in the .HDF5 file '
            f'({n_particles}). Defaulting to using indices.shape '
            f'({n_particles})')
        Logger.rank0.log(logging.WARNING, warn_str)
        if MPI.COMM_WORLD.Get_rank() == 0:
            warnings.warn(warn_str)
        config = copy.deepcopy(config)
        config.n_particles = n_particles
    return config


def check_max_molecule_size(config):
    if config.max_molecule_size is None:
        info_str = (
            f'No max_molecule_size found in toml file {config.file_name}, '
            f'defaulting to 201'
        )
        Logger.rank0.log(logging.INFO, info_str)
        config = copy.deepcopy(config)
        config.max_molecule_size = 201
        return config

    if config.max_molecule_size < 1:
        warn_str = (
            f'max_molecule_size in {config.file_name} must be a positive '
            f'integer, not {config.max_molecule_size}, defaulting to 201'
        )
        Logger.rank0.log(logging.WARNING, warn_str)
        if MPI.COMM_WORLD.Get_rank() == 0:
            warnings.warn(warn_str)
        config = copy.deepcopy(config)
        config.max_molecule_size = 201
        return config
    return config


def _find_unique_names(config, names):
    unique_names = np.unique(names)
    receive_buffer = MPI.COMM_WORLD.gather(unique_names, root=0)

    gathered_unique_names = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        gathered_unique_names = np.unique(np.concatenate(receive_buffer))
    unique_names = MPI.COMM_WORLD.bcast(gathered_unique_names, root=0)
    unique_names = [n.decode('UTF-8') for n in unique_names]
    config.unique_names = unique_names
    return config


def check_bonds(config, names):
    if not hasattr(config, 'unique_names'):
        config = _find_unique_names(config, names)
    unique_names = config.unique_names

    for b in config.bonds:
        if b.atom_1 not in unique_names or b.atom_2 not in unique_names:
            missing_str = ''
            if b.atom_1 not in unique_names:
                if b.atom_2 not in unique_names:
                    if b.atom_1 == b.atom_2:
                        missing_str = f'no {b.atom_1} atoms'
                    else:
                        missing_str = (
                            f'neither {b.atom_1}, nor {b.atom_2} atoms'
                        )
                else:
                    missing_str = f'no {b.atom_1} atoms'
            else:
                missing_str = f'no {b.atom_2} atoms'

            warn_str = (
                f'Bond type {b.atom_1}--{b.atom_2} specified in '
                f'{config.file_name} but {missing_str} are present in the '
                f'specified system (names array)'
            )
            Logger.rank0.log(logging.WARNING, warn_str)
            if MPI.COMM_WORLD.Get_rank() == 0:
                warnings.warn(warn_str)
    return config


def check_angles(config, names):
    if not hasattr(config, 'unique_names'):
        config = _find_unique_names(config, names)
    unique_names = config.unique_names

    for a in config.angle_bonds:
        if (a.atom_1 not in unique_names or
                a.atom_2 not in unique_names or
                a.atom_3 not in unique_names):
            missing = [a.atom_1 not in unique_names,
                       a.atom_2 not in unique_names,
                       a.atom_3 not in unique_names]
            missing_names = [atom for i, atom in enumerate([a.atom_1,
                                                            a.atom_2,
                                                            a.atom_3]) if
                             missing[i]]
            missing_str = ', '.join(np.unique(missing_names))

            warn_str = (
                f'Angle bond type {a.atom_1}--{a.atom_2}--{a.atom_3} '
                f'specified in {config.file_name} but no {missing_str} atoms '
                f'are present in the specified system (names array)'
            )
            Logger.rank0.log(logging.WARNING, warn_str)
            if MPI.COMM_WORLD.Get_rank() == 0:
                warnings.warn(warn_str)
    return config


def check_chi(config, names):
    if not hasattr(config, 'unique_names'):
        config = _find_unique_names(config, names)
    unique_names = config.unique_names

    for c in config.chi:
        if c.atom_1 not in unique_names or c.atom_2 not in unique_names:
            missing_str = ''
            if c.atom_1 not in unique_names:
                if c.atom_2 not in unique_names:
                    if c.atom_1 == c.atom_2:
                        missing_str = f'no {c.atom_1} atoms'
                    else:
                        missing_str = (
                            f'neither {c.atom_1}, nor {c.atom_2} atoms'
                        )
                else:
                    missing_str = f'no {c.atom_1} atoms'
            else:
                missing_str = f'no {c.atom_2} atoms'

            warn_str = (
                f'Chi interaction {c.atom_1}--{c.atom_2} specified in '
                f'{config.file_name} but {missing_str} are present in the '
                f'specified system (names array)'
            )
            Logger.rank0.log(logging.WARNING, warn_str)
            if MPI.COMM_WORLD.Get_rank() == 0:
                warnings.warn(warn_str)

    for i, n in enumerate(unique_names):
        for m in unique_names[i+1:]:
            found = False
            for c in config.chi:
                if ((c.atom_1 == n and c.atom_2 == m) or
                        (c.atom_1 == m and c.atom_2 == n)):
                    found = True
            if not found:
                warn_str = (
                    f'Atom types {n} and {m} found in the '
                    f'system, but no chi interaction {n}--{m} '
                    f'specified in {config.file_name}. Defaulting to '
                    f'chi[{n}, {m}] = 0'
                )
                Logger.rank0.log(logging.WARNING, warn_str)
                if MPI.COMM_WORLD.Get_rank() == 0:
                    warnings.warn(warn_str)
    return config


def check_box_size(config):
    for b in config.box_size:
        if b <= 0.0:
            err_str = (
                f'Invalid box size specified in {config.file_name}: '
                f'{config.box_size}'
            )
            Logger.rank0.log(logging.ERROR, err_str)
            raise ValueError(err_str)
    return config


def check_integrator(config):
    if config.integrator.lower() not in ('velocity-verlet', 'respa'):
        err_str = (
            f'Invalid integrator specified in {config.file_name}: '
            f'{config.integrator}. Available options "velocity-verlet" or '
            f'"respa".'
        )
        Logger.rank0.log(logging.ERROR, err_str)
        raise ValueError(err_str)

    if config.integrator.lower() == 'respa':
        if not isinstance(config.respa_inner, int):
            if isinstance(config.respa_inner, float):
                if config.respa_inner.is_int():
                    warn_str = (
                        f'Number of inner rRESPA time steps in '
                        f'{config.file_name}: {config.respa_inner} specified '
                        f'as float, using {int(config.respa_inner)}'
                    )
                    Logger.rank0.log(logging.WARNING, err_str)
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        warnings.warn(warn_str)
                    config.respa_inner = int(config.respa_inner)
                else:
                    err_str = (
                        f'Invalid number of inner rRESPA time steps in '
                        f'{config.file_name}: {config.respa_inner}. Must be '
                        f'positive integer'
                    )
                    Logger.rank0.log(logging.ERROR, err_str)
                    raise ValueError(err_str)
            else:
                err_str = (
                    f'Invalid number of inner rRESPA time steps in '
                    f'{config.file_name}: {config.respa_inner}. Must be '
                    f'positive integer'
                )
                Logger.rank0.log(logging.ERROR, err_str)
                raise TypeError(err_str)
        else:
            err_str = (
                f'Invalid number of inner rRESPA time steps in '
                f'{config.file_name}: {config.respa_inner}. Must be positive'
                f'integer'
            )
            Logger.rank0.log(logging.ERROR, err_str)
            raise TypeError(err_str)

    if (config.integrator.lower() == 'velocity-verlet' and
            config.respa_inner != 1):
        warn_str = (
            f'Integrator type Velocity-Verlet specified in {config.file_name} '
            f'and inner rRESPA time steps set to {config.respa_inner}. '
            f'Using respa_inner = 1'
        )
        Logger.rank0.log(logging.WARNING, warn_str)
        if MPI.COMM_WORLD.Get_rank() == 0:
            warnings.warn(warn_str)
        config.respa_inner = 1

    return config
