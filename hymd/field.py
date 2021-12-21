import logging
import numpy as np
from mpi4py import MPI
from .logger import Logger


def compute_field_force(layouts, r, force_mesh, force, types, n_types):
    for t in range(n_types):
        ind = types == t
        for d in range(3):
            force[ind, d] = force_mesh[t][d].readout(r[ind], layout=layouts[t])


def compute_field_energy_q(
    config, phi_q_fourier, elec_energy_field, field_q_energy,
    comm=MPI.COMM_WORLD,
):
    elec_conversion_factor = config.coulomb_constant / config.dielectric_const

    def transfer_energy(k, v):
        return (
            4.0 * np.pi * elec_conversion_factor * np.abs(v)**2
            / k.normp(p=2, zeromode=1)
        )

    phi_q_fourier.apply(
        transfer_energy, kind="wavenumber", out=elec_energy_field
    )
    V = np.prod(config.box_size)
    field_q_energy = 0.5 * V * comm.allreduce(np.sum(elec_energy_field.value))
    return field_q_energy.real


def update_field_force_q(
    charges, phi_q, phi_q_fourier, elec_field_fourier, elec_field,
    elec_forces, layout_q, pm, positions, config,
):
    V = np.prod(config.box_size)
    n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh_cells
    pm.paint(positions, layout=layout_q, mass=charges, out=phi_q)
    phi_q /= volume_per_cell
    phi_q.r2c(out=phi_q_fourier)

    def phi_transfer_function(k, v):
        return v * np.exp(-0.5 * config.sigma**2 * k.normp(p=2, zeromode=1))

    phi_q_fourier.apply(phi_transfer_function, out=phi_q_fourier)
    phi_q_fourier.c2r(out=phi_q)
    n_dimensions = config.box_size.size
    elec_conversion_factor = config.coulomb_constant / config.dielectric_const

    for _d in np.arange(n_dimensions):

        def poisson_transfer_function(k, v, d=_d):
            return (
                -1j * k[d] * 4.0 * np.pi * elec_conversion_factor * v
                / k.normp(p=2, zeromode=1)
            )
        phi_q_fourier.apply(
            poisson_transfer_function, out=elec_field_fourier[_d]
        )
        elec_field_fourier[_d].c2r(out=elec_field[_d])

    for _d in np.arange(n_dimensions):
        elec_forces[:, _d] = charges * (
            elec_field[_d].readout(positions, layout=layout_q)
        )


def update_field_force_energy_q(
    charges, phi_q, phi_q_fourier, elec_field_fourier, elec_field, elec_forces,
    elec_energy_field, field_q_energy, layout_q, pm, positions, config,
    compute_energy=False, comm=MPI.COMM_WORLD,
):
    V = np.prod(config.box_size)
    n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh_cells
    pm.paint(positions, layout=layout_q, mass=charges, out=phi_q)
    phi_q /= volume_per_cell
    phi_q.r2c(out=phi_q_fourier)

    def phi_transfer_function(k, v):
        return v * np.exp(-0.5 * config.sigma ** 2 * k.normp(p=2, zeromode=1))

    phi_q_fourier.apply(phi_transfer_function, out=phi_q_fourier)
    phi_q_fourier.c2r(out=phi_q)
    n_dimensions = config.box_size.size
    elec_conversion_factor = config.coulomb_constant / config.dielectric_const

    for _d in np.arange(n_dimensions):
        def poisson_transfer_function(k, v, d=_d):
            return (
                -1j * k[d] * 4.0 * np.pi * elec_conversion_factor * v
                / k.normp(p=2, zeromode=1)
            )

        phi_q_fourier.apply(
            poisson_transfer_function, out=elec_field_fourier[_d]
        )
        elec_field_fourier[_d].c2r(out=elec_field[_d])

    for _d in np.arange(n_dimensions):
        elec_forces[:, _d] = charges * (
            elec_field[_d].readout(positions, layout=layout_q)
        )

    if compute_energy:
        def transfer_energy(k, v):
            return (
                4.0 * np.pi * elec_conversion_factor * np.abs(v)**2
                / k.normp(p=2, zeromode=1)
            )
        phi_q_fourier.apply(
            transfer_energy, kind="wavenumber", out=elec_energy_field
        )
        field_q_energy = 0.5 * comm.allreduce(np.sum(elec_energy_field.value))

    return field_q_energy.real


def update_field(
    phi, layouts, force_mesh, hamiltonian, pm, positions, types, config, v_ext,
    phi_fourier, v_ext_fourier, compute_potential=False,
):
    V = np.prod(config.box_size)
    n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh_cells
    for t in range(config.n_types):
        pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
        phi[t] /= volume_per_cell
        phi[t].r2c(out=phi_fourier[t])
        phi_fourier[t].apply(hamiltonian.H, out=Ellipsis)
        phi_fourier[t].c2r(out=phi[t])

    # External potential
    for t in range(config.n_types):
        hamiltonian.v_ext[t](phi).r2c(out=v_ext_fourier[0])
        v_ext_fourier[0].apply(hamiltonian.H, out=Ellipsis)
        np.copyto(
            v_ext_fourier[1].value, v_ext_fourier[0].value, casting="no",
            where=True,
        )
        np.copyto(
            v_ext_fourier[2].value, v_ext_fourier[0].value, casting="no",
            where=True,
        )
        if compute_potential:
            np.copyto(
                v_ext_fourier[3].value, v_ext_fourier[0].value, casting="no",
                where=True,
            )

        # Differentiate the external potential in fourier space
        for d in range(3):
            def force_transfer_function(k, v, d=d):
                return -k[d] * 1j * v

            v_ext_fourier[d].apply(force_transfer_function, out=Ellipsis)
            v_ext_fourier[d].c2r(out=force_mesh[t][d])

        if compute_potential:
            v_ext_fourier[3].c2r(out=v_ext[t])


def compute_field_and_kinetic_energy(
    phi, velocity, hamiltonian, positions, types, v_ext, config, layouts,
    comm=MPI.COMM_WORLD,
):
    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells

    w = hamiltonian.w(phi) * volume_per_cell
    field_energy = w.csum()
    kinetic_energy = comm.allreduce(0.5 * config.mass * np.sum(velocity ** 2))
    return field_energy, kinetic_energy


def domain_decomposition(
    positions, pm, *args, molecules=None, bonds=None, verbose=0,
    comm=MPI.COMM_WORLD
):
    if molecules is not None:
        assert bonds is not None, "bonds must be provided with molecules"
        unique_molecules = np.sort(np.unique(molecules))
        molecules_com = np.empty_like(positions)
        for m in unique_molecules:
            ind = molecules == m
            r = positions[ind, :][0, :]
            molecules_com[ind, :] = r
        layout = pm.decompose(molecules_com, smoothing=0)
        args = (*args, bonds, molecules)
    else:
        layout = pm.decompose(positions, smoothing=0)
    if verbose > 1:
        Logger.rank0.log(
            logging.INFO,
            "DOMAIN_DECOMP: Total number of particles to be exchanged = %d",
            np.sum(layout.get_exchange_cost()),
        )
    return layout.exchange(positions, *args)
