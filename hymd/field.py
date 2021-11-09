import logging
import numpy as np
from mpi4py import MPI
from logger import Logger
import warnings

def initialize_pm(pmesh, config, comm=MPI.COMM_WORLD):

    # Ignore numpy numpy.VisibleDeprecationWarning: Creating an ndarray from
    # ragged nested sequences until it is fixed in pmesh
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=np.VisibleDeprecationWarning,
            message=r"Creating an ndarray from ragged nested sequences",
        )
        # The first argument of ParticleMesh has to be a tuple
        pm = pmesh.ParticleMesh(
            config.mesh_size, BoxSize=config.box_size, dtype="f4", comm=comm
        )
    phi = [pm.create("real", value=0.0) for _ in range(config.n_types)]
    phi_fourier = [
        pm.create("complex", value=0.0) for _ in range(config.n_types)
    ]  # noqa: E501
    force_on_grid = [
        [pm.create("real", value=0.0) for d in range(3)] for _ in range(config.n_types)
    ]
    v_ext_fourier = [pm.create("complex", value=0.0) for _ in range(4)]
    v_ext = [pm.create("real", value=0.0) for _ in range(config.n_types)]

    phi_transfer = [pm.create("complex", value=0.0) for _ in range(3)]

    phi_gradient = None; phi_lap_filtered_fourier = None; phi_lap_filtered = None
    phi_grad_lap_fourier = None; phi_grad_lap = None; v_ext1 = None
    if config.squaregradient:
        phi_gradient = [
            [pm.create("real", value=0.0) for d in range(3)] for _ in range(config.n_types)
        ]
        phi_lap_filtered_fourier = [ pm.create("complex", value=0.0) for _ in range(config.n_types) ]
        phi_lap_filtered = [pm.create("real", value=0.0) for _ in range(config.n_types)]
        phi_grad_lap_fourier = [pm.create("complex", value=0.0) for _ in range(3)]
        phi_grad_lap = [
            [ [pm.create("real", value=0.0) for d_grad in range(3)] for d_lap in range(3) ] for _ in range(config.n_types)
        ]
        v_ext1 = [pm.create("real", value=0.0) for _ in range(config.n_types)]
    phi_laplacian = [
        [pm.create("real", value=0.0) for d in range(3)] for _ in range(config.n_types)
    ]
    field_list = [phi, phi_fourier, force_on_grid, v_ext_fourier, v_ext, phi_transfer,
            phi_laplacian, phi_lap_filtered, v_ext1]
    return (pm, phi, phi_fourier, force_on_grid, v_ext_fourier, v_ext, phi_transfer,
            phi_gradient, phi_laplacian, phi_lap_filtered_fourier, phi_lap_filtered,
            phi_grad_lap_fourier, phi_grad_lap, v_ext1, field_list)

def compute_field_force(layouts, r, force_mesh, force, types, n_types):
    for t in range(n_types):
        ind = types == t
        for d in range(3):
            force[ind, d] = force_mesh[t][d].readout(r[ind], layout=layouts[t])


def comp_gradient(phi_fourier, phi_transfer, phi_gradient, config,):
    for t in range(config.n_types):
        np.copyto(
            phi_transfer[0].value, phi_fourier[t].value, casting="no", where=True
        )
        np.copyto(
            phi_transfer[1].value, phi_fourier[t].value, casting="no", where=True
        )
        np.copyto(
            phi_transfer[2].value, phi_fourier[t].value, casting="no", where=True
        )

        # Evaluate laplacian of phi in fourier space
        for d in range(3):

            def gradient_transfer(k, v, d=d):
                return 1j * k[d] * v

            phi_transfer[d].apply(gradient_transfer, out=Ellipsis)
            phi_transfer[d].c2r(out=phi_gradient[t][d])

def comp_laplacian(
        phi_fourier,
        phi_transfer,
        phi_laplacian,
        phi_grad_lap_fourier,
        phi_grad_lap,
        hamiltonian,
        config,
        phi_lap_filtered_fourier = None,
):
    for t in range(config.n_types):
        np.copyto(
            phi_transfer[0].value, phi_fourier[t].value, casting="no", where=True
        )
        np.copyto(
            phi_transfer[1].value, phi_fourier[t].value, casting="no", where=True
        )
        np.copyto(
            phi_transfer[2].value, phi_fourier[t].value, casting="no", where=True
        )

        # Evaluate laplacian of phi in fourier space
        for d in range(3):

            def laplacian_transfer(k, v, d=d):
                return -k[d]**2 * v

            phi_transfer[d].apply(laplacian_transfer, out=Ellipsis)
            phi_transfer[d].c2r(out=phi_laplacian[t][d])

        # filter laplacian of phi
        if config.squaregradient and phi_lap_filtered_fourier:
            (phi_transfer[0] + phi_transfer[1] + phi_transfer[2]).apply(hamiltonian.H, out=phi_lap_filtered_fourier[t])

        # gradient of laplacian
        if config.squaregradient:
            for d_lap in range(3):
                #make copies of phi_transfer[d_lap] into phi_grad_lap_fourier[0:3]
                np.copyto(
                    phi_grad_lap_fourier[0].value, phi_transfer[d_lap].value, casting="no", where=True
                )
                np.copyto(
                    phi_grad_lap_fourier[1].value, phi_transfer[d_lap].value, casting="no", where=True
                )
                np.copyto(
                    phi_grad_lap_fourier[2].value, phi_transfer[d_lap].value, casting="no", where=True
                )
                for d_grad in range(3):
                    def gradient_transfer(k, v, d=d_grad):
                        return 1j * k[d] * v
                    phi_grad_lap_fourier[d_grad].apply(gradient_transfer, out = Ellipsis)
                    phi_grad_lap_fourier[d_grad].c2r(out = phi_grad_lap[t][d_lap][d_grad])


def update_field(
    phi,
    phi_gradient,
    phi_laplacian,
    phi_transfer,
    phi_grad_lap_fourier,
    phi_grad_lap,
    layouts,
    force_mesh,
    hamiltonian,
    pm,
    positions,
    types,
    config,
    v_ext,
    phi_fourier,
    phi_lap_filtered_fourier,
    v_ext_fourier,
    phi_lap_filtered,
    v_ext1,
    m,
    compute_potential=False,
):

    V = np.prod(config.box_size)
    n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh_cells
    for t in range(config.n_types):
        pm.paint(positions[types == t],mass=m[t], layout=layouts[t], out=phi[t])
        phi[t] /= volume_per_cell
        phi[t].r2c(out=phi_fourier[t])
        phi_fourier[t].apply(hamiltonian.H, out=Ellipsis)
        phi_fourier[t].c2r(out=phi[t])

    if config.squaregradient:
        # gradient
        comp_gradient(phi_fourier, phi_transfer, phi_gradient, config)
        # laplacian        
        comp_laplacian(phi_fourier, phi_transfer, phi_laplacian, phi_grad_lap_fourier, phi_grad_lap, hamiltonian, config, phi_lap_filtered_fourier)

    

    # External potential
    if config.squaregradient:
        for t in range(config.n_types):
            phi_lap_filtered_fourier[t].c2r(out=phi_lap_filtered[t])
        hamiltonian.v_ext1(phi_lap_filtered, v_ext1)
    
    for t in range(config.n_types):
        if config.squaregradient:
            v = hamiltonian.v_ext[t](phi) + v_ext1[t]
        else:
            v = hamiltonian.v_ext[t](phi)

        v.r2c(out=v_ext_fourier[0])
        v_ext_fourier[0].apply(hamiltonian.H, out=Ellipsis)
        np.copyto(
            v_ext_fourier[1].value, v_ext_fourier[0].value, casting="no", where=True
        )
        np.copyto(
            v_ext_fourier[2].value, v_ext_fourier[0].value, casting="no", where=True
        )
        if compute_potential:
            np.copyto(
                v_ext_fourier[3].value, v_ext_fourier[0].value, casting="no", where=True
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
    phi,
    phi_gradient,
    velocity,
    hamiltonian,
    positions,
    types,
    v_ext,
    config,
    layouts,
    comm=MPI.COMM_WORLD,
):
    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells

    w = hamiltonian.w(phi) * volume_per_cell
    w1 = 0.0
    if config.squaregradient:
        w1 = hamiltonian.w1(phi_gradient) * volume_per_cell
    field_energy = (w + w1).csum() #w to W
    kinetic_energy = comm.allreduce(0.5 * config.mass * np.sum(velocity ** 2))
    return field_energy, kinetic_energy


def domain_decomposition(
    positions,
    pm,
    *args,
    molecules=None,
    bonds=None,
    verbose=0,
    comm=MPI.COMM_WORLD
):
    if molecules is not None:
        assert bonds is not None, "bonds must be provided when molecules are present"
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
    if verbose > 2:
        Logger.rank0.log(
            logging.INFO,
            "DOMAIN_DECOMP: Total number of particles to be exchanged = %d",
            np.sum(layout.get_exchange_cost()),
        )
    return layout.exchange(positions, *args)
