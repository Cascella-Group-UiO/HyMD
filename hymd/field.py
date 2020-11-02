import logging
import numpy as np
from mpi4py import MPI
from logger import Logger


def compute_field_force(layouts, r, force_mesh, force, types, n_types):
    for t in range(n_types):
        for d in range(3):
            force_mesh[t][d].readout(r[types == t], layout=layouts[t],
                                     out=force[types == t, d])


def update_field(phi, layouts, force_mesh, hamiltonian, pm, positions, types,
                 config, v_ext, phi_fourier, v_ext_fourier,
                 compute_potential=False):
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
        np.copyto(v_ext_fourier[1].value, v_ext_fourier[0].value, casting='no',
                  where=True)
        np.copyto(v_ext_fourier[2].value, v_ext_fourier[0].value, casting='no',
                  where=True)
        if compute_potential:
            np.copyto(v_ext_fourier[3].value, v_ext_fourier[0].value,
                      casting='no', where=True)

        # Differentiate the external potential in fourier space
        for d in range(3):
            def force_transfer_function(k, v, d=d):
                return -k[d] * 1j * v
            v_ext_fourier[d].apply(force_transfer_function, out=Ellipsis)
            v_ext_fourier[d].c2r(out=force_mesh[t][d])

        if compute_potential:
            v_ext_fourier[3].c2r(out=v_ext[t])


def compute_field_and_kinetic_energy(phi, velocity, hamiltonian, positions,
                                     types, v_ext, config, layouts,
                                     comm=MPI.COMM_WORLD):
    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells

    w = hamiltonian.w(phi) * volume_per_cell
    field_energy = w.csum()

    """
    v = 0.0
    for t in range(config.n_types):
        v += comm.allreduce(
            np.sum(v_ext[t].readout(positions[types == t], layout=layouts[t]))
        )
        v += v_ext[t].csum() * volume_per_cell
    field_energy = 0.5 * v
    """
    kinetic_energy = comm.allreduce(0.5 * config.mass * np.sum(velocity**2))
    return field_energy, kinetic_energy


def domain_decomposition(positions, molecules, pm, *args, verbose=0,
                         comm=MPI.COMM_WORLD):
    unique_molecules = np.sort(np.unique(molecules))
    molecules_com = np.empty_like(positions)
    for m in unique_molecules:
        ind = molecules == m
        r = positions[ind, :][0, :]
        molecules_com[ind, :] = r
    layout = pm.decompose(molecules_com, smoothing=0)

    if verbose > 1:
        Logger.rank0.log(
            logging.INFO,
            "DOMAIN_DECOMP: Total number of particles to be exchanged = %d",
            np.sum(layout.get_exchange_cost())
        )
    return layout.exchange(positions, molecules, *args)
