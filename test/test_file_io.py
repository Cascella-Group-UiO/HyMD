import pytest
from mpi4py import MPI
import numpy as np
import collections
import itertools
import h5py
from hymd.file_io import (
    OutDataset, setup_time_dependent_element, store_static, store_data,
    distribute_input
)
from hymd.input_parser import Config, parse_config_toml, _setup_type_to_name_map
from hymd.force import prepare_bonds

def test_OutDataset(config_toml, tmp_path):
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)

    out = OutDataset(tmp_path, config)
    assert hasattr(out, "config")
    assert hasattr(out, "disable_mpio")
    assert not out.disable_mpio
    assert hasattr(out, "float_dtype")
    assert out.float_dtype == "float32"
    assert hasattr(out, "file")
    assert isinstance(out.file, h5py.File)
    assert out.file # check if open

    out.flush()
    out.close_file()
    assert not out.file # check if closed

    out = OutDataset(tmp_path, config, double_out=True)
    assert out.float_dtype == "float64"
    out.close_file()

    out = OutDataset(tmp_path, config, disable_mpio=True)
    assert out.disable_mpio
    assert isinstance(out.file, h5py.File)
    assert out.file # check if open
    out.close_file()
    assert not out.file


def test_setup_time_dependent_element(config_toml, tmp_path):
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)

    out = OutDataset(tmp_path, config)
    test_group = out.file.create_group("/test")

    (
        group, step, time, value
    ) = setup_time_dependent_element(
        "position",
        test_group,
        1,
        (config.n_particles, 3),
        "float32",
        units="nm"
    )

    assert isinstance(group, h5py.Group)
    assert group.name == "/test/position"
    assert group.attrs["units"] == "nm"

    assert isinstance(step, h5py.Dataset)
    assert isinstance(time, h5py.Dataset)
    assert isinstance(value, h5py.Dataset)

    out.close_file()


@pytest.mark.mpi()
def test_distribute_input():
    # create fake indices and molecules
    indices = np.arange(0, 10000)
    molecules = np.zeros_like(indices)
    molecules[400:450] = 1
    molecules[450:] = np.arange(2, 9552)
    assert indices.shape == molecules.shape

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    in_file = {'indices': indices}
    rank_range, molecules_flag = distribute_input(
        in_file, rank, size,  None, comm=comm
    )
    assert not molecules_flag
    gather_range = comm.allgather(rank_range)
    gather_range = list(itertools.chain.from_iterable(gather_range))
    assert np.array_equal(gather_range, indices)

    in_file = {'indices': indices, 'molecules': molecules}
    rank_range, molecules_flag = distribute_input(
        in_file, rank, size, indices.shape[0], max_molecule_size=1000, comm=comm
    )
    assert molecules_flag
    gather_range = comm.allgather(rank_range)
    gather_range = list(itertools.chain.from_iterable(gather_range))
    assert np.array_equal(gather_range, indices)


@pytest.mark.mpi()
def test_store_static(molecules_with_solvent, tmp_path):
    indices, positions, molecules, velocities, bonds, names, types = molecules_with_solvent
    box_size = np.array([10, 10, 10], dtype=np.float64)
    config = Config(n_steps=0, n_print=1, time_step=0.03, box_size=box_size,
                    mesh_size=[5, 5, 5], sigma=0.5, kappa=0.05,
                    n_particles=len(indices))
    Bond = collections.namedtuple(
        "Bond", ["atom_1", "atom_2", "equilibrium", "strength"]
    )
    cbonds = (
        Bond("A", "A", 0.27, 10000),
        Bond("A", "B", 0.27, 10000),
        Bond("A", "C", 0.27, 10000),
        Bond("B", "B", 0.27, 10000),
        Bond("B", "C", 0.27, 10000),
    )
    config.bonds = cbonds
    config = _setup_type_to_name_map(config, names, types)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    test_dir = comm.bcast(tmp_path, root=0)
    out = OutDataset(test_dir, config, comm=comm)

    # Test stub acting like a hdf5 file for distribute_input
    in_file = {'molecules': molecules, 'indices': indices}
    rank_range, molecules_flag = distribute_input(
        in_file, rank, size,  config.n_particles, 6, comm=comm
    )
    positions_ = positions[rank_range]
    molecules_ = molecules[rank_range]
    indices_ = indices[rank_range]
    velocities_ = velocities[rank_range]
    bonds_ = bonds[rank_range]
    types_ = types[rank_range]
    names_ = names[rank_range]

    bonds_prep = prepare_bonds(
        molecules_, names_, bonds_, indices_, config
    )
    (
        # two-particle bonds
        bonds_2_atom1, bonds_2_atom2, bonds_2_equilibrium,
        bonds_2_stength,
        # three-particle bonds
        bonds_3_atom1, bonds_3_atom2, bonds_3_atom3,
        bonds_3_equilibrium, bonds_3_stength,
        # four-particle bondsl
        bonds_4_atom1, bonds_4_atom2, bonds_4_atom3, bonds_4_atom4,
        bonds_4_coeff, bonds_4_type, bonds_4_last,
    ) = bonds_prep


    store_static(
        out, rank_range, names_, types_, indices_, config, bonds_2_atom1,
        bonds_2_atom2, molecules=molecules_
    )

    groups = ['connectivity', 'h5md', 'observables', 'parameters', 'particles']
    assert all((k in out.file.keys()) for k in groups)
    groups = ['box', 'mass', 'position', 'species']
    assert all((k in out.file['particles/all']) for k in groups)
    groups = [
        'angle_energy', 'angular_momentum', 'bond_energy', 
        'dihedral_energy', 'field_energy', 'kinetic_energy', 
        'potential_energy', 'temperature', 'thermostat_work', 
        'torque', 'total_energy', 'total_momentum'
    ]
    assert all((k in out.file['observables']) for k in groups)
    assert 'vmd_structure' in out.file['parameters'].keys()

    out.close_file()


@pytest.mark.mpi()
def test_store_data(molecules_with_solvent, tmp_path):
    indices, positions, molecules, velocities, bonds, names, types = molecules_with_solvent
    box_size = np.array([10, 10, 10], dtype=np.float64)
    config = Config(n_steps=100, n_print=1, time_step=0.03, box_size=box_size,
                    mesh_size=[5, 5, 5], sigma=0.5, kappa=0.05,
                    n_particles=len(indices), mass=72.)
    Bond = collections.namedtuple(
        "Bond", ["atom_1", "atom_2", "equilibrium", "strength"]
    )
    cbonds = (
        Bond("A", "A", 0.27, 10000),
        Bond("A", "B", 0.27, 10000),
        Bond("A", "C", 0.27, 10000),
        Bond("B", "B", 0.27, 10000),
        Bond("B", "C", 0.27, 10000),
    )
    config.bonds = cbonds
    config = _setup_type_to_name_map(config, names, types)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    test_dir = comm.bcast(tmp_path, root=0)
    out = OutDataset(test_dir, config, comm=comm)

    # Test stub acting like a hdf5 file for distribute_input
    in_file = {'molecules': molecules, 'indices': indices}
    rank_range, molecules_flag = distribute_input(
        in_file, rank, size,  config.n_particles, 6, comm=comm
    )
    positions_ = positions[rank_range]
    molecules_ = molecules[rank_range]
    indices_ = indices[rank_range]
    velocities_ = velocities[rank_range]
    bonds_ = bonds[rank_range]
    types_ = types[rank_range]
    names_ = names[rank_range]

    bonds_prep = prepare_bonds(
        molecules_, names_, bonds_, indices_, config
    )
    (
        # two-particle bonds
        bonds_2_atom1, bonds_2_atom2, bonds_2_equilibrium,
        bonds_2_stength,
        # three-particle bonds
        bonds_3_atom1, bonds_3_atom2, bonds_3_atom3,
        bonds_3_equilibrium, bonds_3_stength,
        # four-particle bondsl
        bonds_4_atom1, bonds_4_atom2, bonds_4_atom3, bonds_4_atom4,
        bonds_4_coeff, bonds_4_type, bonds_4_last,
    ) = bonds_prep


    # call it to prepare OutDataset
    store_static(
        out, rank_range, names_, types_, indices_, config, bonds_2_atom1,
        bonds_2_atom2, molecules=molecules_
    )

    forces = np.copy(positions_)

    store_data(
        out, 0, 0, indices_, positions_, velocities_, forces,
        box_size, 300., 1., 2., 3., 4., 5., 6., 0.02, config
    )

    outdataset_step = [
        out.positions_step,
        out.total_energy_step,
        out.potential_energy_step,
        out.kinetc_energy_step,
        out.bond_energy_step,
        out.angle_energy_step,
        out.dihedral_energy_step,
        out.field_energy_step,
        out.total_momentum_step,
        out.angular_momentum_step,
        out.torque_step,
        out.temperature_step,
        out.thermostat_work_step,
    ]

    outdataset_time = [
        out.positions_time,
        out.total_energy_time,
        out.potential_energy_time,
        out.kinetc_energy_time,
        out.bond_energy_time,
        out.angle_energy_time,
        out.dihedral_energy_time,
        out.field_energy_time,
        out.total_momentum_time,
        out.angular_momentum_time,
        out.torque_time,
        out.temperature_time,
        out.thermostat_work_time,
    ]

    for dset in outdataset_step:
        assert isinstance(dset, h5py.Dataset)
        print(dset)
        assert dset[0] == 0

    for dset in outdataset_time:
        assert isinstance(dset, h5py.Dataset)
        assert dset[0] == 0

    assert out.potential_energy[0] == pytest.approx(20.)
    assert out.kinetc_energy[0] == pytest.approx(1.)
    assert out.temperature[0] == pytest.approx(300.)
    assert out.bond_energy[0] == pytest.approx(2.)
    assert out.angle_energy[0] == pytest.approx(3.)
    assert out.dihedral_energy[0] == pytest.approx(4.)
    assert out.field_energy[0] == pytest.approx(5.)

    out.close_file()
