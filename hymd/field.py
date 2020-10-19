import logging
import numpy as np
from logger import Logger


def compute_field_force(layouts, r, force_mesh, types, n_types, out=None):
    if out is None:
        out = np.zeros((len(r), 3))
    else:
        assert out.shape == r.shape
    for t in range(n_types):
        for d in range(3):
            out[types == t, d] = (force_mesh[t][d]
                                  .readout(r[types == t], layout=layouts[t]))
    return out


def update_field(phi, layouts, force_mesh, hamiltonian, pm, positions, types,
                 config, potential, compute_potential=False):
    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells
    for t in range(config.n_types):
        phi[t] = pm.paint(positions[types == t], layout=layouts[t]) / volume_per_cell  # noqa: E501
        phi[t] = (phi[t].r2c(out=Ellipsis)
                        .apply(hamiltonian.H, out=Ellipsis)
                        .c2r(out=Ellipsis))

    # External potential
    for t in range(config.n_types):
        v_ext_fourier_space = (hamiltonian.v_ext[t](phi)
                               .r2c(out=Ellipsis)
                               .apply(hamiltonian.H, out=Ellipsis))

        # Differentiate the external potential in fourier space
        for d in range(3):
            def force_transfer_function(k, v, d=d):
                return -k[d] * 1j * v

            force_mesh[t][d] = (v_ext_fourier_space.copy()
                                .apply(force_transfer_function, out=Ellipsis)
                                .c2r(out=Ellipsis))

        if compute_potential:
            potential[t] = v_ext_fourier_space.c2r(out=Ellipsis)


def compute_field_and_kinetic_energy(phi, velocity, hamiltonian, pm, config):
    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells

    w = hamiltonian.w(phi) * volume_per_cell
    field_energy = w.csum()
    kinetic_energy = pm.comm.allreduce(0.5 * config.mass * np.sum(velocity**2))

    return field_energy, kinetic_energy


def domain_decomposition(positions, velocities, forces, indices, types, pm):
    layout = pm.decompose(positions, smoothing=0)
    Logger.rank0.log(
        logging.INFO,
        "DOMAIN_DECOMP: Total number of particles to be exchanged = %d",
        np.sum(layout.get_exchange_cost())
    )
    return layout.exchange(positions, velocities, forces, indices, types)
