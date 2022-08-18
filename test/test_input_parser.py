import pytest
import warnings
import re
import io
import logging
import warnings
from types import ModuleType
import numpy as np
from mpi4py import MPI
from hymd.input_parser import (
    Config,
    _find_unique_names,
    _setup_type_to_name_map,
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
    check_n_print,
    check_tau,
    check_mass,
    check_domain_decomposition,
    check_name,
    check_config,
    check_hamiltonian,
    check_charges,
    check_n_flush,
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
        with pytest.warns(Warning) as recorded_warning:
            config_ = check_n_particles(config, indices_take)
            assert config_.n_particles == 10001
            message = recorded_warning[0].message.args[0]
            log = caplog.text
            assert all([(s in message) for s in ("10000", "not", "10001")])
            assert all([(s in log) for s in ("10000", "not", "10001")])
    else:
        with warnings.catch_warnings() as recorded_warning:
            warnings.simplefilter("error")
            config_ = check_n_particles(config, indices_take)
            assert config_.n_particles == 10001

    caplog.clear()
    indices = np.empty((n_particles_config,))
    indices_slices = np.array_split(indices, MPI.COMM_WORLD.Get_size())
    indices_take = indices_slices[MPI.COMM_WORLD.Get_rank()]

    with warnings.catch_warnings() as recorded_warning:
        warnings.simplefilter("error")
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
        with pytest.warns(Warning) as recorded_warning:
            config = parse_config_toml(config_toml_wrong_max_molecule_size)
            config = check_max_molecule_size(config)
            assert config.max_molecule_size == 201
            message = recorded_warning[0].message.args[0]
            log = caplog.text
            assert all([(s in message) for s in ("must be", "integer", "201")])
            assert all([(s in log) for s in ("must be", "integer", "201")])
    else:
        with warnings.catch_warnings() as recorded_warning:
            warnings.simplefilter("error")
            config = parse_config_toml(config_toml_wrong_max_molecule_size)
            config = check_max_molecule_size(config)
            assert config.max_molecule_size == 201

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
            with pytest.warns(Warning) as recorded_warning:
                config = check_bonds(config, names_take)
                message = recorded_warning[0].message.args[0]
                log = caplog.text
                assert all([(s in message) for s in w])
                assert all([(s in log) for s in w])
        else:
            with warnings.catch_warnings() as recorded_warning:
                warnings.simplefilter("error")
                config = check_bonds(config, names_take)

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
            with pytest.warns(Warning) as recorded_warning:
                config = check_angles(config, names_take)
                message = recorded_warning[0].message.args[0]
                log = caplog.text
                assert all([(s in message) for s in w])
                assert all([(s in log) for s in w])
        else:
            with warnings.catch_warnings() as recorded_warning:
                warnings.simplefilter("error")
                config = check_angles(config, names_take)

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
            with pytest.warns(Warning) as recorded_warning:
                config = check_chi(config, names_take)
                message = recorded_warning[0].message.args[0]
                log = caplog.text
                assert all([(s in message) for s in w])
                assert all([(s in log) for s in w])
        else:
            with warnings.catch_warnings() as recorded_warning:
                warnings.simplefilter("error")
                config = check_chi(config, names_take)

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
            with pytest.warns(Warning) as recorded_warning:
                config = check_chi(config, names_take)
                message = recorded_warning[0].message.args[0]
                log = caplog.text
                assert all([(s in message) for s in w])
                assert all([(s in log) for s in w])
        else:
            with warnings.catch_warnings() as recorded_warning:
                warnings.simplefilter("error")
                config = check_chi(config, names_take)

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

    with pytest.warns(Warning) as recorded_warning:
        config.respa_inner = 1.0
        config = check_integrator(config)
        assert config.respa_inner == 1
        log = caplog.text
        message = recorded_warning[0].message.args[0]
        cmp_strings = ("Number of inner rRESPA time steps",
                       "specified as float")
        assert all([(s in message) for s in cmp_strings])
        assert all([(s in log) for s in cmp_strings])

    with pytest.raises(TypeError) as recorded_error:
        config.respa_inner = "test"
        _ = check_integrator(config)
    message = str(recorded_error.value)
    cmp_strings = ("Invalid number of inner rRESPA time steps", 
                   "Must be positive integer")
    assert all([(s in message) for s in cmp_strings])

    with pytest.raises(ValueError) as recorded_error:
        config.respa_inner = -1
        _ = check_integrator(config)
    message = str(recorded_error.value)
    cmp_strings = ("Invalid number of inner rRESPA time steps", 
                   "Must be positive integer")
    assert all([(s in message) for s in cmp_strings])

    with pytest.warns(Warning) as recorded_warning:
        config.integrator = "velocity-verlet"
        config.respa_inner = 2
        config = check_integrator(config)
        assert config.respa_inner == 1
        log = caplog.text
        message = recorded_warning[0].message.args[0]
        cmp_strings = ("Integrator type Velocity-Verlet specified",
                       "and inner rRESPA time steps set to",
                       "Using respa_inner = 1")
        assert all([(s in message) for s in cmp_strings])
        assert all([(s in log) for s in cmp_strings])

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


def test_input_parser_check_n_print(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)

    config.n_print = None
    config = check_n_print(config)
    assert config.n_print == False

    with pytest.warns(Warning) as recorded_warning:
        config.n_print = 1.27
        config = check_n_print(config)
        assert config.n_print == 1
        message = recorded_warning[0].message.args[0]
        log = caplog.text
        cmp_string = "n_print is a float"
        assert cmp_string in message
        assert cmp_string in log

    with pytest.raises(RuntimeError) as recorded_error:
        config.n_print = "test"
        _ = check_n_print(config)        
        log = caplog.text
        cmp_string = "invalid value for n_print"
        assert cmp_string in log
    message = str(recorded_error.value)
    cmp_string = "invalid value for n_print"
    assert cmp_string in message

    caplog.clear()


def test_input_parser_check_tau(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)

    with pytest.warns(Warning) as recorded_warning:
        config.tau = None
        config = check_tau(config)
        assert config.tau == pytest.approx(0.7)
        message = recorded_warning[0].message.args[0]
        log = caplog.text
        cmp_string = "target temp specified but no tau"
        assert cmp_string in message
        assert cmp_string in log

    caplog.clear()

   
def test_input_parser_check_mass(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)

    config = check_mass(config)
    assert config.mass == pytest.approx(72.0)

    config.mass = None
    config = check_mass(config)
    log = caplog.text
    assert config.mass == pytest.approx(72.0)
    cmp_string = "no mass specified, defaulting to 72.0"
    assert cmp_string in log

    with pytest.raises(TypeError) as recorded_error:
        config.mass = "test"
        _ = check_mass(config)
    message = str(recorded_error.value)
    cmp_string = "specified mass is invalid type"
    assert cmp_string in message

    with pytest.raises(ValueError) as recorded_error:
        config.mass = -1.0
        _ = check_mass(config)
    message = str(recorded_error.value)
    cmp_string = "invalid mass specified"
    assert cmp_string in message

    caplog.clear()


def test_input_parser_check_domain_decomposition(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)

    config = check_domain_decomposition(config)
    assert config.domain_decomposition == False

    with pytest.warns(Warning) as recorded_warning:
        config.domain_decomposition = -1
        config = check_domain_decomposition(config)
        assert config.domain_decomposition == False
        message = recorded_warning[0].message.args[0]
        log = caplog.text
        cmp_string = "negative domain_decomposition specified, using False"
        assert cmp_string in message
        assert cmp_string in log

    with pytest.warns(Warning) as recorded_warning:
        config.domain_decomposition = -1.1
        config = check_domain_decomposition(config)
        assert config.domain_decomposition == False
        message = recorded_warning[0].message.args[0]
        log = caplog.text
        cmp_string = "negative domain_decomposition specified, using False"
        assert cmp_string in message
        assert cmp_string in log
    
    with pytest.warns(Warning) as recorded_warning:
        config.domain_decomposition = 1.0
        config = check_domain_decomposition(config)
        assert config.domain_decomposition == 1
        message = recorded_warning[0].message.args[0]
        log = caplog.text
        cmp_strings = ("domain_decomposition", "is not an integer, using")
        assert all([(s in message) for s in cmp_strings])
        assert all([(s in log) for s in cmp_strings])

    with pytest.raises(ValueError) as recorded_error:
        config.domain_decomposition = "test"
        _ = check_domain_decomposition(config)
    message = str(recorded_error.value)
    cmp_strings = ("invalid value for domain_decomposition", 
                   "use an integer")
    assert all([(s in message) for s in cmp_strings])

    caplog.clear()


def test_input_parser_check_name(config_toml):
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)

    config = check_name(config)
    assert "example config.toml" in config.name

    config.name = None
    config = check_name(config)
    assert "sim" in config.name


def test_input_parser_check_config(config_toml, dppc_single):
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    config.n_particles = 13

    indices, _, names, _, _, _ = dppc_single

    indices = np.append(indices, [12])
    names = np.append(names, [b"W"])
    names_to_types = {"N":0, "P": 1, "G": 2, "C": 3, "W": 4}
    types = np.array([names_to_types[n.decode('UTF-8')] for n in names],
                     dtype=int)

    config = check_config(config, indices, names, types)
    assert isinstance(config, Config)


def test_input_parser_check_hamiltonian(config_toml, caplog):
    caplog.set_level(logging.INFO)
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)

    config = check_hamiltonian(config)
    assert config.hamiltonian == "DefaultWithChi"

    with pytest.warns(Warning) as recorded_warning:
        config.hamiltonian = None
        config = check_hamiltonian(config)
        assert config.hamiltonian == "DefaultWithChi"
        message = recorded_warning[0].message.args[0]
        log = caplog.text
        cmp_strings = ("No hamiltonian form specified", 
                       "defaulting to DefaultWithChi")
        assert all([(s in message) for s in cmp_strings])
        assert all([(s in log) for s in cmp_strings])

    with pytest.warns(Warning) as recorded_warning:
        config.hamiltonian = None
        config.chi = []
        config = check_hamiltonian(config)
        assert config.hamiltonian == "DefaultNoChi"
        message = recorded_warning[0].message.args[0]
        log = caplog.text
        cmp_strings = ("No hamiltonian form and no chi", 
                       "defaulting to DefaultNoChi")
        assert all([(s in message) for s in cmp_strings])
        assert all([(s in log) for s in cmp_strings])

    with pytest.raises(NotImplementedError) as recorded_error:
        config.hamiltonian = "test"
        _ = check_hamiltonian(config)
    message = str(recorded_error.value)
    cmp_strings = ("The specified Hamiltonian", 
                   "was not recognized as a valid Hamiltonian")
    assert all([(s in message) for s in cmp_strings])

    caplog.clear()


def test_input_parser__find_unique_names(config_toml, dppc_single):
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)

    _, _, names, _, _, _ = dppc_single

    config = _find_unique_names(config, names)
    assert hasattr(config, "unique_names")
    assert isinstance(config.unique_names, list)
    assert all([(n in config.unique_names) for n in ["N","P","G","C"]])
    assert hasattr(config, "n_types")
    assert config.n_types == 4


def test_input_parser__setup_type_to_name_map(config_toml, dppc_single):
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)

    _, _, names, _, _, _ = dppc_single

    names_to_types = {"N":0, "P": 1, "G": 2, "C": 3}
    types = np.array([names_to_types[n.decode('UTF-8')] for n in names],
                     dtype=int)

    config = _setup_type_to_name_map(config, names, types)
    assert hasattr(config, "name_to_type_map")
    assert isinstance(config.name_to_type_map, dict)
    assert all([(k in config.name_to_type_map.keys()) 
                 for k in names_to_types.keys()])
    assert hasattr(config, "type_to_name_map")
    assert isinstance(config.type_to_name_map, dict)
    assert all([(v in config.type_to_name_map.keys()) 
                 for v in names_to_types.values()])


@pytest.mark.mpi()
def test_input_parser_check_charges(caplog):
    caplog.set_level(logging.WARNING)
    charges = np.array(
        [1.0, 0.0, 0.5, 0.2, -0.3, 0.0, 0.3, 0.0, -0.5, 0.0, -0.5, 0.99, -0.2,
        0.3, 0.5, 0.0, 0.0, -0.3, 0.0, 0.0, -0.99, -1.0],
        dtype=float
    )
    # split charges across ranks
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_charges = charges.shape[0] // size
    ind_rank = {}
    for i in range(size):
        ind_rank[i] = i * n_charges
    ind_rank[size] = size * n_charges + np.mod(charges.shape[0], size)

    rank_charges = charges[ind_rank[rank]:ind_rank[rank+1]]

    check_charges(rank_charges, comm=comm) # warnings are not expected
    assert len(caplog.text) == 0

    # change charges to generate a warning
    charges[1] = 1000.
    rank_charges = charges[ind_rank[rank]:ind_rank[rank+1]]

    with pytest.warns(Warning) as recorded_warning:
        config = check_charges(rank_charges)
        # only rank 0 gives the warning
        if rank == 0:
            message = recorded_warning[0].message.args[0]
            log = caplog.text
            cmp_strings = ("Charges in the input file do not sum to zero.", 
                           "Total charge is ")
            assert all([(s in message) for s in cmp_strings])
            assert all([(s in log) for s in cmp_strings])
            total_charge = float(log.split()[-1][:-1])
            assert total_charge == pytest.approx(charges[1])
        # but others should give a warn so pytest.warns does not fail with:
        # Failed: DID NOT WARN. No warnings of type..
        else:
            warnings.warn("give a warning so other ranks do not fail")
            assert len(recorded_warning) == 1


def test_input_parser_check_n_flush(config_toml):
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)

    config = check_n_flush(config)

    assert config.n_flush == (10000 // config.n_print)

    config.n_flush = 1234

    config = check_n_flush(config)

    assert config.n_flush == 1234
