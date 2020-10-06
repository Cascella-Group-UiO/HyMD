import pytest
import numpy as np
from hPF.hamiltonian import DefaultNoChi, DefaultWithChi
from hPF.input_parser import parse_config_toml, _find_unique_names


def test_DefaultNoChi(dppc_single, config_toml):
    indices, _, names, _, r, _ = dppc_single
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    config = _find_unique_names(config, names)
