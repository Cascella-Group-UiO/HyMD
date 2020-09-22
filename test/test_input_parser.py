import pytest
import re
import logging
import numpy as np
from mpi4py import MPI
from hPF.input_parser import (Config, read_config_toml, parse_config_toml,
                              check_n_particles, check_max_molecule_size)


def test_input_parser_read_config_toml(config_toml):
    config_toml_file, config_toml_str = config_toml
    file_content = read_config_toml(config_toml_file)
    assert config_toml_str == file_content


def test_input_parser_file(config_toml):
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    assert isinstance(config, Config)


def test_input_parser_file_check_n_particles(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    n_particles_config = config.n_particles
    indices = np.empty((n_particles_config + 1,))

    with pytest.warns(Warning) as recorded_warning:
        config_ = check_n_particles(config, indices)
        assert config_.n_particles == 10001
        if MPI.COMM_WORLD.Get_rank() == 0:
            message = recorded_warning[0].message.args[0]
            log = caplog.text
            assert all([(s in message) for s in ('10000', 'not', '10001')])
            assert all([(s in log) for s in ('10000', 'not', '10001')])

    caplog.clear()
    indices = np.empty((n_particles_config,))
    with pytest.warns(None) as recorded_warning:
        check_n_particles(config, indices)
        assert not recorded_warning
        assert not caplog.text


def test_input_parser_check_optionals(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config_toml_no_n_particles = re.sub(
        'n_particles[ \t]*=[ \t]*[0-9]+', '', config_toml_str
    )
    config = parse_config_toml(config_toml_no_n_particles)
    assert config.n_particles is None
    indices = np.empty((100,))
    config = check_n_particles(config, indices)
    assert config.n_particles == 100
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert all([(s in caplog.text) for s in
                    ('No', 'n_particles', '100')])
    caplog.clear()

    config_toml_no_max_molecule_size = re.sub(
        'max_molecule_size[ \t]*=[ \t]*[0-9]+', '', config_toml_str
    )
    config = parse_config_toml(config_toml_no_max_molecule_size)
    assert config.max_molecule_size is None
    config = check_max_molecule_size(config)
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert all([(s in caplog.text) for s in
                    ('No', 'max_molecule_size', '201')])
    caplog.clear()

    config_toml_wrong_type_max_molecule_size = re.sub(
        'max_molecule_size[ \t]*=[ \t]*[0-9]+',
        'max_molecule_size = 0', config_toml_str
    )
    with pytest.warns(Warning) as recorded_warning:
        config = parse_config_toml(config_toml_wrong_type_max_molecule_size)
        config = check_max_molecule_size(config)
        assert config.max_molecule_size == 201
        if MPI.COMM_WORLD.Get_rank() == 0:
            message = recorded_warning[0].message.args[0]
            log = caplog.text
            assert all([(s in message) for s in ('must be', 'integer', '201')])
            assert all([(s in log) for s in ('must be', 'integer', '201')])
