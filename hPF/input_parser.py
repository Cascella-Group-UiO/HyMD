import copy
import toml
import logging
import warnings
import numpy as np
from mpi4py import MPI
from dataclasses import dataclass, field
from typing import List, Union
from hPF.force import Bond, Angle, Chi
from hPF.logger import clog


@dataclass
class Config:
    mass: float
    n_steps: int
    n_print: int
    time_step: float
    box_size: Union[List[float], np.ndarray]
    integrator: str
    respa_inner: int
    mesh_size: Union[Union[List[int], np.ndarray], int]

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


def read_config_toml(file_path):
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
        clog(logging.INFO, info_str, comm=MPI.COMM_WORLD)
        return config

    if n_particles != config.n_particles:
        warn_str = (
            f'n_particles in {config.file_name} ({config.n_particles}) does '
            'not match the shape of the indices array in the .HDF5 file '
            f'({n_particles}). Defaulting to using indices.shape '
            f'({n_particles})')
        clog(logging.WARNING, warn_str, comm=MPI.COMM_WORLD)
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
        clog(logging.INFO, info_str, comm=MPI.COMM_WORLD)
        config = copy.deepcopy(config)
        config.max_molecule_size = 201
        return config

    if config.max_molecule_size < 1:
        warn_str = (
            f'max_molecule_size in {config.file_name} must be a positive '
            f'integer, not {config.max_molecule_size}, defaulting to 201'
        )
        clog(logging.WARNING, warn_str, comm=MPI.COMM_WORLD)
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


def check_bonds(config, names):
    if not hasattr(config, 'unique_names'):
        _find_unique_names(config, names)
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
            clog(logging.WARNING, warn_str, comm=MPI.COMM_WORLD)
            if MPI.COMM_WORLD.Get_rank() == 0:
                warnings.warn(warn_str)
    return config


def check_chi(config, names):
    if not hasattr(config, 'unique_names'):
        _find_unique_names(config, names)
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
            clog(logging.WARNING, warn_str, comm=MPI.COMM_WORLD)
            if MPI.COMM_WORLD.Get_rank() == 0:
                warnings.warn(warn_str)
    return config
