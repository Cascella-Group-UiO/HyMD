import pytest
import re
import io
import logging
from types import ModuleType
import numpy as np
from mpi4py import MPI
from hymd.input_parser import (
    Config,
    read_config_toml,
    parse_config_toml,
    check_n_particles,
    check_max_molecule_size,
    check_bonds,
    check_angles,
    check_chi,
    check_box_size,
    check_integrator,
    check_thermostat_coupling_groups,
    check_cancel_com_momentum,
    check_start_and_target_temperature,
)


def test_input_parser_read_config_toml(config_toml):
    config_toml_file, config_toml_str = config_toml
    file_content = read_config_toml(config_toml_file)
    assert config_toml_str == file_content


def test_input_parser_file(config_toml):
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    assert isinstance(config, Config)
    assert isinstance(str(config), str)


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
            assert all([(s in message) for s in ("10000", "not", "10001")])
            assert all([(s in log) for s in ("10000", "not", "10001")])

    caplog.clear()
    indices = np.empty((n_particles_config,))
    indices_slices = np.array_split(indices, MPI.COMM_WORLD.Get_size())
    indices_take = indices_slices[MPI.COMM_WORLD.Get_rank()]

    with pytest.warns(None) as recorded_warning:
        check_n_particles(config, indices_take)
        assert not recorded_warning
        assert not caplog.text
    MPI.COMM_WORLD.Barrier()


@pytest.mark.mpi()
def test_input_parser_check_optionals(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config_toml_no_n_particles = re.sub(
        "n_particles[ \t]*=[ \t]*[0-9]+", "", config_toml_str
    )
    config = parse_config_toml(config_toml_no_n_particles)
    assert config.n_particles is None
    indices = np.empty((100,))
    indices_slices = np.array_split(indices, MPI.COMM_WORLD.Get_size())
    indices_take = indices_slices[MPI.COMM_WORLD.Get_rank()]

    config = check_n_particles(config, indices_take)
    assert config.n_particles == 100
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert all([(s in caplog.text) for s in ("No", "n_particles", "100")])
    caplog.clear()

    config_toml_no_max_molecule_size = re.sub(
        "max_molecule_size[ \t]*=[ \t]*[0-9]+", "", config_toml_str
    )
    config = parse_config_toml(config_toml_no_max_molecule_size)
    assert config.max_molecule_size is None
    config = check_max_molecule_size(config)
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert all([(s in caplog.text) for s in ("No", "max_molecule_size", "201")])  # noqa: E501
    caplog.clear()

    config_toml_wrong_max_molecule_size = re.sub(
        "max_molecule_size[ \t]*=[ \t]*[0-9]+", "max_molecule_size = 0",
        config_toml_str
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
            assert all([(s in message) for s in ("must be", "integer", "201")])
            assert all([(s in log) for s in ("must be", "integer", "201")])
    MPI.COMM_WORLD.Barrier()


def _add_to_config(config_str, new_str, header_str):
    sio = io.StringIO(config_str)
    sio_new = []
    header_flag = False
    for line in sio:
        if line.strip().startswith(f"{header_str} ="):
            header_flag = True
        if header_flag and line.strip() == "]":
            header_flag = False
            sio_new.append(new_str)
        sio_new.append(line.rstrip())
    return "\n".join(s for s in sio_new)


def _remove_from_config(config_str, remove_str):
    sio = io.StringIO(config_str)
    sio_new = []
    for line in sio:
        if remove_str not in line.strip():
            sio_new.append(line.rstrip())
    return "\n".join(s for s in sio_new)


def _change_in_config(config_str, old_line, new_line):
    sio = io.StringIO(config_str)
    sio_new = []
    for line in sio:
        if line.strip().startswith(old_line.strip()):
            sio_new.append(new_line.rstrip())
        else:
            sio_new.append(line.rstrip())
    return "\n".join(s for s in sio_new)


@pytest.mark.mpi()
def test_input_parser_check_bonds(config_toml, dppc_single, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    _, _, names, _, _, _ = dppc_single
    solvent_names = np.empty((8), dtype="a5")
    solvent_names.fill("W")
    names = np.concatenate((names, solvent_names))

    name_slices = np.array_split(names, MPI.COMM_WORLD.Get_size())
    names_take = name_slices[MPI.COMM_WORLD.Get_rank()]

    add_bonds = [
        '  ["A", "C", 0.47, 1250.0],',
        '  ["A", "A", 0.47, 1250.0],',
        '  ["P", "B", 0.47, 1250.0],',
        '  ["A", "A", 0.47, 1250.0],',
        '  ["A", "B", 0.47, 1250.0],',
    ]
    warn_strs = [
        ["A--C", "no A"],
        ["A--A", "no A"],
        ["P--B", "no B"],
        ["A--A", "no A"],
        ["A--B", "neither A, B"],
    ]

    for a, w in zip(add_bonds, warn_strs):
        added_bonds_toml_str = _add_to_config(config_toml_str, a, "bonds")
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
    MPI.COMM_WORLD.Barrier()


@pytest.mark.mpi()
def test_input_parser_check_angles(config_toml, dppc_single, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    _, _, names, _, _, _ = dppc_single
    solvent_names = np.empty((8), dtype="a5")
    solvent_names.fill("W")
    names = np.concatenate((names, solvent_names))

    name_slices = np.array_split(names, MPI.COMM_WORLD.Get_size())
    names_take = name_slices[MPI.COMM_WORLD.Get_rank()]

    add_angles = [
        '  ["A", "C", "C", 115.9, 25.0],',
        '  ["A", "A", "A", 158.5, 25.0],',
        '  ["P", "B", "C",   1.5, 25.0],',
        '  ["A", "A", "B",  99.8, 25.0],',
        '  ["A", "B", "K", 119.8, 25.0],',
    ]
    warn_strs = [
        ["A--C--C", "no A"],
        ["A--A--A", "no A"],
        ["P--B--C", "no B"],
        ["A--A--B", "no A, B"],
        ["A--B--K", "no A, B, K"],
    ]

    for a, w in zip(add_angles, warn_strs):
        added_angles_toml_str = _add_to_config(
            config_toml_str, a, "angle_bonds"
        )
        config = parse_config_toml(added_angles_toml_str)

        if MPI.COMM_WORLD.Get_rank() == 0:
            warning = Warning
        else:
            warning = None
        with pytest.warns(warning) as recorded_warning:
            config = check_angles(config, names_take)
            if MPI.COMM_WORLD.Get_rank() == 0:
                message = recorded_warning[0].message.args[0]
                log = caplog.text
                assert all([(s in message) for s in w])
                assert all([(s in log) for s in w])
        caplog.clear()
    MPI.COMM_WORLD.Barrier()


@pytest.mark.mpi()
def test_input_parser_check_chi(config_toml, dppc_single, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    for c in config.chi:
        if c.atom_1 == "G" and c.atom_2 == "W":
            assert c.interaction_energy == pytest.approx(4.53, abs=1e-2)
        elif c.atom_1 == "P" and c.atom_2 == "C":
            assert c.interaction_energy == pytest.approx(14.72, abs=1e-2)

    _, _, names, _, _, _ = dppc_single
    solvent_names = np.empty((8), dtype="a5")
    solvent_names.fill("W")
    names = np.concatenate((names, solvent_names))

    name_slices = np.array_split(names, MPI.COMM_WORLD.Get_size())
    names_take = name_slices[MPI.COMM_WORLD.Get_rank()]

    add_chi = [
        '  ["A", "C", 1.2398],',
        '  ["A", "A", -9.0582],',
        '  ["P", "B", -8.8481],',
        '  ["A", "A", 3.1002],',
        '  ["A", "B", 2.7815],',
    ]
    warn_strs = [
        ["A--C", "no A"],
        ["A--A", "no A"],
        ["B--P", "no B"],
        ["A--A", "no A"],
        ["A--B", "neither A, B"],
    ]

    for a, w in zip(add_chi, warn_strs):
        added_chi_toml_str = _add_to_config(config_toml_str, a, "chi")
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

    remove_chi = [
        '["C", "W", 42.24],',
        '["N", "P", -9.34],',
        '["N", "C", 13.56],',
        '["P", "C", 14.72],',
    ]
    warn_strs = [
        ["C and W", "no chi interaction C--W", "Defaulting"],
        ["N and P", "no chi interaction N--P", "Defaulting"],
        ["C and N", "no chi interaction C--N", "Defaulting"],
        ["C and P", "no chi interaction C--P", "Defaulting"],
    ]

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
                assert all([(s in message) for s in w])
                assert all([(s in log) for s in w])
        caplog.clear()
    MPI.COMM_WORLD.Barrier()


def test_input_parser_check_box_size(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    assert np.allclose(np.array([2.1598, 11.2498, 5.1009]), config.box_size, atol=1e-4)  # noqa: E501

    changed_box_toml_str = _change_in_config(
        config_toml_str, "box_size = [", "box_size = [2.25, -3.91, 4.11]"
    )
    config = parse_config_toml(changed_box_toml_str)

    with pytest.raises(ValueError) as recorded_error:
        _ = check_box_size(config)
        log = caplog.text
        assert all([(s in log) for s in ("Invalid", "box")])
    message = str(recorded_error.value)
    assert all([(s in message) for s in ("Invalid", "box")])
    caplog.clear()


def test_input_parser_check_integrator(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    assert config.integrator == "respa"
    assert config.respa_inner == 5

    changed_integrator_toml_str = _change_in_config(
        config_toml_str, "integrator = ", 'integrator = "rsjgyu"'
    )
    config = parse_config_toml(changed_integrator_toml_str)

    with pytest.raises(ValueError) as recorded_error:
        _ = check_integrator(config)
        if MPI.COMM_WORLD.Get_rank() == 0:
            log = caplog.text
            assert all([(s in log) for s in ("Invalid", "integrator")])
    message = str(recorded_error.value)
    assert all([(s in message) for s in ("Invalid", "integrator")])
    caplog.clear()


def test_input_parser_thermostat_coupling_groups(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    nn = (("N", "P"), ("G", "C"), ("W"))
    for i in range(3):
        for n in nn[i]:
            assert n in config.thermostat_coupling_groups[i]

    thermostat_coupling_groups_toml_str = _remove_from_config(
        config_toml_str, '["N", "P"],'
    )
    thermostat_coupling_groups_toml_str = _remove_from_config(
        thermostat_coupling_groups_toml_str, '["G", "C"],'
    )
    thermostat_coupling_groups_toml_str = _remove_from_config(
        thermostat_coupling_groups_toml_str, '["W"],'
    )
    thermostat_coupling_groups_toml_str = _add_to_config(
        thermostat_coupling_groups_toml_str,
        '        ["N", "P", "G", "C"],',
        "thermostat_coupling_groups",
    )
    thermostat_coupling_groups_toml_str = _add_to_config(
        thermostat_coupling_groups_toml_str,
        '        ["W"],',
        "thermostat_coupling_groups",
    )
    config = parse_config_toml(thermostat_coupling_groups_toml_str)
    nn = (("N", "P", "G", "C"), ("W"))
    for i in range(2):
        for n in nn[i]:
            assert n in config.thermostat_coupling_groups[i]

    thermostat_coupling_groups_toml_str = _remove_from_config(
        thermostat_coupling_groups_toml_str, '["N", "P", "G", "C"],'
    )
    thermostat_coupling_groups_toml_str = _add_to_config(
        thermostat_coupling_groups_toml_str,
        '        ["N", "P", "G"],',
        "thermostat_coupling_groups",
    )

    config = parse_config_toml(thermostat_coupling_groups_toml_str)
    config.unique_names = sorted(["N", "P", "G", "C", "W"])
    with pytest.raises(ValueError) as recorded_error:
        _ = check_thermostat_coupling_groups(config)
        log = caplog.text
        assert all([(s in log) for s in ("species C", "not specified")])

    message = str(recorded_error.value)
    assert all([(s in message) for s in ("species C", "not specified")])
    caplog.clear()

    thermostat_coupling_groups_toml_str = _remove_from_config(
        thermostat_coupling_groups_toml_str, '["N", "P", "G"],'
    )
    thermostat_coupling_groups_toml_str = _remove_from_config(
        thermostat_coupling_groups_toml_str, '["W"],'
    )
    thermostat_coupling_groups_toml_str = _add_to_config(
        thermostat_coupling_groups_toml_str,
        '        ["N", "P", "G", "C"],',
        "thermostat_coupling_groups",
    )
    thermostat_coupling_groups_toml_str = _add_to_config(
        thermostat_coupling_groups_toml_str,
        '        ["W", "P"],',
        "thermostat_coupling_groups",
    )
    config = parse_config_toml(thermostat_coupling_groups_toml_str)
    config.unique_names = sorted(["N", "P", "G", "C", "W"])
    with pytest.raises(ValueError) as recorded_error:
        _ = check_thermostat_coupling_groups(config)
        log = caplog.text
        assert all([(s in log) for s in ("species P", "specified", "multiple")])  # noqa: E501

    message = str(recorded_error.value)
    print(message)
    assert all([(s in message) for s in ("species P", "specified", "multiple")])  # noqa: E501
    caplog.clear()


def test_input_parser_check_cancel_com_momentum(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    for t in (
        1.1,
        "hello",
        MPI.COMM_WORLD,
        config_toml,
        [1],
    ):
        config.cancel_com_momentum = t
        with pytest.raises(ValueError) as recorded_error:
            _ = check_cancel_com_momentum(config)
            log = caplog.text
            assert all([(s in log) for s in ("not interpret", "an integer")])

        message = str(recorded_error.value)
        assert all([(s in message) for s in ("not interpret", "an integer")])
        caplog.clear()

    for t in (-1, -100, 0, False):
        config.cancel_com_momentum = t
        config_ = check_cancel_com_momentum(config)
        assert config_.cancel_com_momentum is False
    caplog.clear()


def test_input_parser_check_start_and_target_temperature(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    for t in (
        "hello",
        MPI.COMM_WORLD,
        config_toml,
        [1],
    ):
        config.start_temperature = t
        with pytest.raises(TypeError) as recorded_error:
            _ = check_start_and_target_temperature(config)
            log = caplog.text
            assert all([(s in log) for s in ("not interpret", "a number")])
        message = str(recorded_error.value)
        assert all([(s in message) for s in ("not interpret", "a number")])
        caplog.clear()

        config.target_temperature = t
        with pytest.raises(TypeError) as recorded_error:
            _ = check_start_and_target_temperature(config)
            log = caplog.text
            assert all([(s in log) for s in ("not interpret", "a number")])
        message = str(recorded_error.value)
        assert all([(s in message) for s in ("not interpret", "a number")])
        caplog.clear()

    config = parse_config_toml(config_toml_str)

    config.start_temperature = None
    assert check_start_and_target_temperature(config).start_temperature is False  # noqa: E501
    config.target_temperature = None
    assert check_start_and_target_temperature(config).target_temperature is False  # noqa: E501

    with pytest.warns(Warning) as recorded_warning:
        config.start_temperature = -6.2985252885781357
        config = check_start_and_target_temperature(config)
        assert config.start_temperature is False
        message = recorded_warning[0].message.args[0]
        log = caplog.text
        assert all([(s in message) for s in ("to negative", "defaulting to")])
        assert all([(s in log) for s in ("to negative", "defaulting to")])
    caplog.clear()

    with pytest.warns(Warning) as recorded_warning:
        config.target_temperature = -0.000025892857873
        config = check_start_and_target_temperature(config)
        assert config.target_temperature is False
        message = recorded_warning[0].message.args[0]
        log = caplog.text
        assert all([(s in message) for s in ("to negative", "defaulting to")])
        assert all([(s in log) for s in ("to negative", "defaulting to")])
    caplog.clear()
