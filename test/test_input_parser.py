import pytest
import re
import io
import logging
import numpy as np
from mpi4py import MPI
from hPF.input_parser import (Config, read_config_toml, parse_config_toml,
                              check_n_particles, check_max_molecule_size,
                              check_bonds, check_chi)


def test_input_parser_read_config_toml(config_toml):
    config_toml_file, config_toml_str = config_toml
    file_content = read_config_toml(config_toml_file)
    assert config_toml_str == file_content


def test_input_parser_file(config_toml):
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    assert isinstance(config, Config)


@pytest.mark.mpi()
def test_input_parser_file_check_n_particles(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    n_particles_config = config.n_particles
    indices = np.empty((n_particles_config + 1,))
    indices_slices = np.array_split(indices, MPI.COMM_WORLD.Get_size())
    indices_take = indices_slices[MPI.COMM_WORLD.Get_rank()]

    if MPI.COMM_WORLD.Get_rank() == 0:
        warning = Warning
    else:
        warning = None
    with pytest.warns(warning) as recorded_warning:
        config_ = check_n_particles(config, indices_take)
        assert config_.n_particles == 10001
        if MPI.COMM_WORLD.Get_rank() == 0:
            message = recorded_warning[0].message.args[0]
            log = caplog.text
            assert all([(s in message) for s in ('10000', 'not', '10001')])
            assert all([(s in log) for s in ('10000', 'not', '10001')])

    caplog.clear()
    indices = np.empty((n_particles_config,))
    indices_slices = np.array_split(indices, MPI.COMM_WORLD.Get_size())
    indices_take = indices_slices[MPI.COMM_WORLD.Get_rank()]

    with pytest.warns(None) as recorded_warning:
        check_n_particles(config, indices_take)
        assert not recorded_warning
        assert not caplog.text


@pytest.mark.mpi()
def test_input_parser_check_optionals(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config_toml_no_n_particles = re.sub(
        'n_particles[ \t]*=[ \t]*[0-9]+', '', config_toml_str
    )
    config = parse_config_toml(config_toml_no_n_particles)
    assert config.n_particles is None
    indices = np.empty((100,))
    indices_slices = np.array_split(indices, MPI.COMM_WORLD.Get_size())
    indices_take = indices_slices[MPI.COMM_WORLD.Get_rank()]

    config = check_n_particles(config, indices_take)
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

    config_toml_wrong_max_molecule_size = re.sub(
        'max_molecule_size[ \t]*=[ \t]*[0-9]+',
        'max_molecule_size = 0', config_toml_str
    )

    if MPI.COMM_WORLD.Get_rank() == 0:
        warning = Warning
    else:
        warning = None
    with pytest.warns(warning) as recorded_warning:
        config = parse_config_toml(config_toml_wrong_max_molecule_size)
        config = check_max_molecule_size(config)
        assert config.max_molecule_size == 201
        if MPI.COMM_WORLD.Get_rank() == 0:
            message = recorded_warning[0].message.args[0]
            log = caplog.text
            assert all([(s in message) for s in ('must be', 'integer', '201')])
            assert all([(s in log) for s in ('must be', 'integer', '201')])


def _add_to_config(config_str, new_str, header_str):
    sio = io.StringIO(config_str)
    sio_new = []
    header_flag = False
    for line in sio:
        if line.strip().startswith(f'{header_str} ='):
            header_flag = True
        if header_flag and line.strip() == ']':
            header_flag = False
            sio_new.append(new_str)
        sio_new.append(line.rstrip())
    return '\n'.join(s for s in sio_new)


def _remove_from_config(config_str, remove_str):
    sio = io.StringIO(config_str)
    sio_new = []
    for line in sio:
        if remove_str not in line.strip():
            sio_new.append(line.rstrip())
    return '\n'.join(s for s in sio_new)


@pytest.mark.mpi()
def test_input_parser_check_bonds(config_toml, dppc_single, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    _, _, names, _, _, _ = dppc_single
    solvent_names = np.empty((8), dtype='a5')
    solvent_names.fill('W')
    names = np.concatenate((names, solvent_names))

    name_slices = np.array_split(names, MPI.COMM_WORLD.Get_size())
    names_take = name_slices[MPI.COMM_WORLD.Get_rank()]

    add_bonds = ['  [["A", "C"], [0.47, 1250.0]],',
                 '  [["A", "A"], [0.47, 1250.0]],',
                 '  [["P", "B"], [0.47, 1250.0]],',
                 '  [["A", "A"], [0.47, 1250.0]],',
                 '  [["A", "B"], [0.47, 1250.0]],']
    warn_strs = [['A--C', 'no A'],
                 ['A--A', 'no A'],
                 ['P--B', 'no B'],
                 ['A--A', 'no A'],
                 ['A--B', 'neither A, nor B']]

    for a, w in zip(add_bonds, warn_strs):
        added_bonds_toml_str = _add_to_config(config_toml_str, a, 'bonds')
        config = parse_config_toml(added_bonds_toml_str)

        if MPI.COMM_WORLD.Get_rank() == 0:
            warning = Warning
        else:
            warning = None
        with pytest.warns(warning) as recorded_warning:
            config = check_bonds(config, names_take)
            if MPI.COMM_WORLD.Get_rank() == 0:
                message = recorded_warning[0].message.args[0]
                log = caplog.text
                assert all([(s in message) for s in w])
                assert all([(s in log) for s in w])
        caplog.clear()


@pytest.mark.mpi()
def test_input_parser_check_chi(config_toml, dppc_single, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    _, _, names, _, _, _ = dppc_single
    solvent_names = np.empty((8), dtype='a5')
    solvent_names.fill('W')
    names = np.concatenate((names, solvent_names))

    name_slices = np.array_split(names, MPI.COMM_WORLD.Get_size())
    names_take = name_slices[MPI.COMM_WORLD.Get_rank()]

    add_chi = ['  [["A", "C"], [1.2398]],',
               '  [["A", "A"], [-9.0582]],',
               '  [["P", "B"], [-8.8481]],',
               '  [["A", "A"], [3.1002]],',
               '  [["A", "B"], [2.7815]],']
    warn_strs = [['A--C', 'no A'],
                 ['A--A', 'no A'],
                 ['P--B', 'no B'],
                 ['A--A', 'no A'],
                 ['A--B', 'neither A, nor B']]

    for a, w in zip(add_chi, warn_strs):
        added_chi_toml_str = _add_to_config(config_toml_str, a, 'chi')
        config = parse_config_toml(added_chi_toml_str)

        if MPI.COMM_WORLD.Get_rank() == 0:
            warning = Warning
        else:
            warning = None
        with pytest.warns(warning) as recorded_warning:
            config = check_chi(config, names_take)
            if MPI.COMM_WORLD.Get_rank() == 0:
                message = recorded_warning[0].message.args[0]
                log = caplog.text
                assert all([(s in message) for s in w])
                assert all([(s in log) for s in w])
        caplog.clear()

    remove_chi = ['[["C", "W"], [42.24]],',
                  '[["N", "P"], [-9.34]],',
                  '[["N", "C"], [13.56]],',
                  '[["P", "C"], [14.72]],']
    warn_strs = [['C and W', 'no chi interaction C--W', 'Defaulting'],
                 ['N and P', 'no chi interaction N--P', 'Defaulting'],
                 ['C and N', 'no chi interaction C--N', 'Defaulting'],
                 ['C and P', 'no chi interaction C--P', 'Defaulting']]

    for r, w in zip(remove_chi, warn_strs):
        removed_chi_toml_str = _remove_from_config(config_toml_str, r)
        config = parse_config_toml(removed_chi_toml_str)

        if MPI.COMM_WORLD.Get_rank() == 0:
            warning = Warning
        else:
            warning = None
        with pytest.warns(warning) as recorded_warning:
            config = check_chi(config, names_take)
            if MPI.COMM_WORLD.Get_rank() == 0:
                message = recorded_warning[0].message.args[0]
                log = caplog.text
                print("------------------------")
                print("------------------------")
                print(r)
                print(w)
                print(message)
                print(log)
                print("------------------------")
                print("------------------------")
                assert all([(s in message) for s in w])
                assert all([(s in log) for s in w])
        caplog.clear()
