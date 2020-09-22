import toml
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union
from hPF.force import Bond, Angle, Chi


@dataclass
class Config:
    file_name: str
    mass: float
    n_steps: int
    n_print: int
    time_step: float
    box_size: Union[List[float], np.ndarray]
    integrator: str
    respa_inner: int
    mesh_size: Union[Union[List[int], np.ndarray], int]

    name: str = None
    tags: List[str] = field(default_factory=list)
    chi: List[Chi] = field(default_factory=list)
    angle_bonds: List[Angle] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    n_particles: int = None
    max_molecule_size: int = None


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
                chi[i] = Chi(atom_1=v[0][0], atom_2=v[0][1],
                             interaction_energy=v[1][0])
    for k in ('bonds', 'angle_bonds', 'chi'):
        if k in config_dict:
            config_dict.pop(k)
    return Config(file_name=file_path, bonds=bonds, angle_bonds=angle_bonds,
                  chi=chi, **config_dict)
