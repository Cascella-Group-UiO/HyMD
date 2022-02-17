"""Forces and energies from the discretized particle-field grid interactions
"""
import logging
import numpy as np
from mpi4py import MPI
from .logger import Logger


def compute_field_force(layouts, r, force_mesh, force, types, n_types):
    """Interpolate particle-field forces from the grid onto particle positions

    Backmaps the forces calculated on the grid to particle positions using the
    window function :math:`P` (by default cloud-in-cell [CIC]). In the
    following, let :math:`\\mathbf{F}_j` denote the force acting on the
    particle with index :math:`j` and position :math:`\\mathbf{r}_j`. The
    interpolated force is

    .. math::

        \\mathbf{F}_j = -\\sum_k\\nabla V_{j_k}
            P(\\mathbf{r}_{j_k}-\\mathbf{r}_j)h^3,

    where :math:`V_{j_k}` is the discretized external potential at grid vertex
    :math:`j_k`, :math:`\\mathbf{r}_{j_k}` is the position of the grid vertex
    with index :math:`j_k`, and :math:`h^3` is the volume of each grid voxel.
    The sum is taken over all closest neighbour vertices, :math:`j_k`.

    Parameters
    ----------
    layouts : list[pmesh.domain.Layout]
        Pmesh communication layout objects for domain decompositions of each
        particle type. Used as blueprint by :code:`pmesh.pm.readout` for
        exchange of particle information across MPI ranks as necessary.
    r : (N,D) numpy.ndarray
        Array of positions for :code:`N` particles in :code:`D` dimensions.
        Local for each MPI rank.
    force_mesh : list[pmesh.pm.RealField]
        Pmesh :code:`RealField` objects containing discretized particle-field
        force density values on the computational grid; :code:`D` fields in D
        dimensions for each particle type. Local for each MPI rank--the full
        computaional grid is represented by the collective fields of all MPI
        ranks.
    force : (N,D) numpy.ndarray
        Array of forces for :code:`N` particles in :code:`D` dimensions. Local
        for each MPI rank.
    types : (N,) numpy.ndarray
        Array of type indices for each of :code:`N` particles. Local for each
        MPI rank.
    n_types : int
        Number of different unique types present in the simulation system.
        :code:`n_types` is global, i.e. the same for all MPI ranks even if some
        ranks contain zero particles of certain types.
    """
    for t in range(n_types):
        ind = types == t
        for d in range(3):
            force[ind, d] = force_mesh[t][d].readout(r[ind], layout=layouts[t])


def compute_field_energy_q(
    config, phi_q_fourier, elec_energy_field, field_q_energy,
    comm=MPI.COMM_WORLD,
):
    """Calculate the electrostatic energy from a field configuration

    From the definition of the elecrostatic potential :math:`\\Psi`, the energy
    is

    .. math::

        E = \\frac{1}{2}\\int\\mathrm{d}\\mathbf{r}\\,
            \\rho(\\mathbf{r}) \\Psi(\\mathbf{r}),

    where :math:`\\rho(\\mathbf{r})` denotes the charge density at position
    :math:`\\mathbf{r}`. Through application of Gauss' law and writing
    :math:`\\Psi` in terms of the electric field :math:`\\mathbf{E}`, this
    becomes

    .. math::

        E = \\frac{\\varepsilon}{2}\\int\\mathrm{d}\\mathbf{r}\\,
            \\mathbf{E}\\cdot \\mathbf{E},

    where :math:`\\varepsilon` is the relative dielectric of the simulation
    medium.

    Parameters
    ----------
    config : Config
        Configuration object.
    phi_q_fourier : pmesh.pm.ComplexField
        Pmesh :code:`ComplexField` object containing the discretized
        electrostatic potential values in reciprocal space on the
        computational grid. Local for each MPI rank--the full computaional grid
        is represented by the collective fields of all MPI ranks.
    elec_energy_field : pmesh.pm.ComplexField
        Pmesh :code:`ComplexField` object for storing calculated discretized
        electrostatic energy density values in reciprocal space on the
        computational grid. Pre-allocated, but empty; any values in this field
        are discarded. Changed in-place. Local for each MPI rank--the full
        computaional grid is represented by the collective fields of all MPI
        ranks.
    field_q_energy : float
        Total elecrostatic energy.

    Returns
    -------
    elec_field_energy : float
        Total electrostatic energy.
    """
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
    """Calculate the electrostatic particle-field forces on the grid

    Computes the electrostatic potential :math:`\\Psi` from particle charges
    through the smoothed charge density :math:`\\tilde\\rho`. With :math:`P`
    being the cloud-in-cell (CIC) window function, the charge density and
    filtered charge densities are computed as

    .. math::

        \\rho(\\mathbf{r}) = \\sum_i q_i P(\\mathbf{r}-\\mathbf{r}_i),

    and

    .. math::

        \\tilde\\rho(\\mathbf{r}) = \\int\\mathrm{x}\\mathbf{r}\\,
            \\rho(\\mathbf{x})H(\\mathbf{r}-\\mathbf{x}),

    where :math:`H` is the grid-independent filtering function. The
    electrostatic potential is computed in reciprocal space as

    .. math::

        \\Phi = \\mathrm{FFT}^{-1}\\left[
            \\frac{4\\pi k_e}{\\varepsilon \\vert\\mathbf{k}\\vert^2}
            \\mathrm{FFT}(\\rho)\\mathrm{FFT}(H)
        \\right],

    with the electric field

    .. math::

        \\mathbf{E} = \\mathrm{FFT}^{-1}\\left[
            -i\\mathbf{k}\\,\\mathrm{FFT}(\\Psi)
        \\right].

    In the following, let :math:`\\mathbf{F}_j` denote the electrostatic force
    acting on the particle with index :math:`j` and position
    :math:`\\mathbf{r}_j`. The interpolated electrostatic force is

    .. math::

        \\mathbf{F}_j = \\sum_k q_j\\mathbf{E}_{j_k}
            P(\\mathbf{r}_{j_k}-\\mathbf{r}_j)h^3,

    where :math:`\\mathbf{E}_{j_k}` is the discretized electric field at grid
    vertex :math:`j_k`, :math:`\\mathbf{r}_{j_k}` is the position of the grid
    vertex with index :math:`j_k`, and :math:`h^3` is the volume of each grid
    voxel. The sum is taken over all closest neighbour vertices, :math:`j_k`.

    Parameters
    ----------
    charges : (N,) numpy.ndarray
        Array of particle charge values for :code:`N` particles. Local for each
        MPI rank.
    phi_q : pmesh.pm.RealField
        Pmesh :code:`RealField` object for storing calculated discretized
        charge density density values on the computational grid. Pre-allocated,
        but empty; any values in this field are discarded. Changed in-place.
        Local for each MPI rank--the full computaional grid is represented by
        the collective fields of all MPI ranks.
    phi_q_fourier : pmesh.pm.ComplexField
        Pmesh :code:`ComplexField` object for storing calculated discretized
        Fourier transformed charge density values in reciprocal space on the
        computational grid. Pre-allocated, but empty; any values in this field
        are discarded. Changed in-place. Local for each MPI rank--the full
        computaional grid is represented by the collective fields of all MPI
        ranks.
    elec_field_fourier : pmesh.pm.ComplexField
        Pmesh :code:`ComplexField` object for storing calculated discretized
        electric field values in reciprocal space on the computational grid.
        Pre-allocated, but empty; any values in this field are discarded.
        Changed in-place. Local for each MPI rank--the full computaional grid
        is represented by the collective fields of all MPI ranks.
    elec_field : pmesh.pm.RealField
        Pmesh :code:`RealField` object for storing calculated discretized
        electric field values on the computational grid. Pre-allocated,
        but empty; any values in this field are discarded. Changed in-place.
        Local for each MPI rank--the full computaional grid is represented by
        the collective fields of all MPI ranks.
    elec_forces : (N,D) numpy.ndarray
        Array of electrostatic forces on :code:`N` particles in :code:`D`
        dimensions.
    layout_q : pmesh.domain.Layout
        Pmesh communication layout object for domain decomposition of the full
        system. Used as blueprint by :code:`pmesh.pm.paint` and
        :code:`pmesh.pm.readout` for exchange of particle information across
        MPI ranks as necessary.
    pm : pmesh.pm.ParticleMesh
        Pmesh :code:`ParticleMesh` object interfacing to the CIC window
        function and the PFFT discrete Fourier transform library.
    positions : (N,D) numpy.ndarray
        Array of positions for :code:`N` particles in :code:`D` dimensions.
        Local for each MPI rank.
    config : hymd.input_parser.Config
        Configuration object.
    """
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
    """Calculate the electrostatic particle-field energy and forces on the grid

    .. deprecated:: 1.0.0
        :code:`update_field_force_energy_q` was deprecated in favour of
        :code:`update_field_force_q` and independent calls to
        :code:`compute_field_energy_q` prior to 1.0.0 release.

    Parameters
    ----------
    charges : (N,) numpy.ndarray
        Array of particle charge values for :code:`N` particles. Local for each
        MPI rank.
    phi_q : pmesh.pm.RealField
        Pmesh :code:`RealField` object for storing calculated discretized
        charge density density values on the computational grid. Pre-allocated,
        but empty; any values in this field are discarded. Changed in-place.
        Local for each MPI rank--the full computaional grid is represented by
        the collective fields of all MPI ranks.
    phi_q_fourier : pmesh.pm.ComplexField
        Pmesh :code:`ComplexField` object for storing calculated discretized
        Fourier transformed charge density values in reciprocal space on the
        computational grid. Pre-allocated, but empty; any values in this field
        are discarded. Changed in-place. Local for each MPI rank--the full
        computaional grid is represented by the collective fields of all MPI
        ranks.
    elec_field_fourier : pmesh.pm.ComplexField
        Pmesh :code:`ComplexField` object for storing calculated discretized
        electric field values in reciprocal space on the computational grid.
        Pre-allocated, but empty; any values in this field are discarded.
        Changed in-place. Local for each MPI rank--the full computaional grid
        is represented by the collective fields of all MPI ranks.
    elec_field : pmesh.pm.RealField
        Pmesh :code:`RealField` object for storing calculated discretized
        electric field values on the computational grid. Pre-allocated,
        but empty; any values in this field are discarded. Changed in-place.
        Local for each MPI rank--the full computaional grid is represented by
        the collective fields of all MPI ranks.
    elec_forces : (N,D) numpy.ndarray
        Array of electrostatic forces on :code:`N` particles in :code:`D`
        dimensions.
    elec_energy_field : pmesh.pm.ComplexField
        Pmesh :code:`ComplexField` object for storing calculated discretized
        electrostatic energy density values in reciprocal space on the
        computational grid. Pre-allocated, but empty; any values in this field
        are discarded. Changed in-place. Local for each MPI rank--the full
        computaional grid is represented by the collective fields of all MPI
        ranks.
    field_q_energy : float
        Total elecrostatic energy.
    layout_q : pmesh.domain.Layout
        Pmesh communication layout object for domain decomposition of the full
        system. Used as blueprint by :code:`pmesh.pm.paint` and
        :code:`pmesh.pm.readout` for exchange of particle information across
        MPI ranks as necessary.
    pm : pmesh.pm.ParticleMesh
        Pmesh :code:`ParticleMesh` object interfacing to the CIC window
        function and the PFFT discrete Fourier transform library.
    positions : (N,D) numpy.ndarray
        Array of positions for :code:`N` particles in :code:`D` dimensions.
        Local for each MPI rank.
    config : Config
        Configuration object.
    compute_energy : bool, optional
        Computes the electrostatic energy if :code:`True`, otherwise only
        computes the electrostatic forces.
    comm : mpi4py.Comm
        MPI communicator to use for rank commuication.

    Returns
    -------
    elec_field_energy : float
        Total electrostatic energy.
    """
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
    """Calculate the particle-field potential and force density

    If :code:`compute_potential` is :code:`True`, the energy may subsequently
    be computed by calling :code:`compute_field_and_kinetic_energy`.

    Computes the particle-field external potential :math:`V_\\text{ext}` from
    particle number densities through the smoothed density field,
    :math:`\\phi(\\mathbf{r})`. With :math:`P` being the cloud-in-cell (CIC)
    window function, the density and filtered densities are computed as

    .. math::

        \\phi(\\mathbf{r}) = \\sum_i P(\\mathbf{r}-\\mathbf{r}_i),

    and

    .. math::

        \\tilde\\phi(\\mathbf{r}) = \\int\\mathrm{x}\\mathbf{r}\\,
            \\phi(\\mathbf{x})H(\\mathbf{r}-\\mathbf{x}),

    where :math:`H` is the grid-independent filtering function. The
    external potential is computed in reciprocal space as

    .. math::

        V_\\text{ext} = \\mathrm{FFT}^{-1}\\left[
            \\mathrm{FFT}
                \\left(
                    \\frac{\\partial w}{\\partial \\tilde\\phi}
                \\right)
            \\mathrm{FFT}(H)
        \\right],

    where :math:`w` is the interaction energy functional. Differentiating
    :math:`V_\\text{ext}` is done by simply applying :math:`i\\mathbf{k}` in
    Fourier space, and the resulting forces are back-transformed to direct
    space and interpolated to particle positions by

    .. math::

        \\mathbf{F}_j = -\\sum_{j_k} \\nabla V_{j_k}
            P(\\mathbf{r}_{j_k} - \\mathbf{r}_j)h^3,

    where :math:`j_k` are the neighbouring vertices of particle :math:`j` at
    position :math:`\\mathbf{r}_j`, and :math:`h^3` is the volume of each grid
    voxel.

    Parameters
    ----------
    phi : list[pmesh.pm.RealField]
        Pmesh :code:`RealField` objects containing discretized particle number
        density values on the computational grid; one for each particle type.
        Pre-allocated, but empty; any values in this field are discarded.
        Changed in-place. Local for each MPI rank--the full computaional grid
        is represented by the collective fields of all MPI ranks.
    layouts : list[pmesh.domain.Layout]
        Pmesh communication layout objects for domain decompositions of each
        particle type. Used as blueprint by :code:`pmesh.pm.readout` for
        exchange of particle information across MPI ranks as necessary.
    force_mesh : list[pmesh.pm.RealField]
        Pmesh :code:`RealField` objects containing discretized particle-field
        force density values on the computational grid; :code:`D` fields in D
        dimensions for each particle type. Pre-allocated, but empty; any values
        in this field are discarded. Changed in-place. Local for each MPI
        rank--the full computaional grid is represented by the collective
        fields of all MPI ranks.
    hamiltonian : Hamiltonian
        Particle-field interaction energy handler object. Defines the
        grid-independent filtering function, :math:`H`.
    pm : pmesh.pm.ParticleMesh
        Pmesh :code:`ParticleMesh` object interfacing to the CIC window
        function and the PFFT discrete Fourier transform library.
    positions : (N,D) numpy.ndarray
        Array of positions for :code:`N` particles in :code:`D` dimensions.
        Local for each MPI rank.
    types : (N,) numpy.ndarray
        Array of type indices for each of :code:`N` particles. Local for each
        MPI rank.
    config : Config
        Configuration object.
    v_ext : list[pmesh.pm.RealField]
        Pmesh :code:`RealField` objects containing discretized particle-field
        external potential values on the computational grid; one for each
        particle type. Pre-allocated, but empty; any values in this field are
        discarded Changed in-place. Local for each MPI rank--the full
        computaional grid is represented by the collective fields of all MPI
        ranks.
    phi_fourier : list[pmesh.pm.ComplexField]
        Pmesh :code:`ComplexField` objects containing discretized particle
        number density values in reciprocal space on the computational grid;
        one for each particle type. Pre-allocated, but empty; any values in
        this field are discarded Changed in-place. Local for each MPI rank--the
        full computaional grid is represented by the collective fields of all
        MPI ranks.
    v_ext_fourier : list[pmesh.pm.ComplexField]
        Pmesh :code:`ComplesField` objects containing discretized
        particle-field external potential values in reciprocal space on the
        computational grid; :code:`D+1` fields in D dimensions for each
        particle type. :code:`D` copies are made after calculation for later
        use in force calculation, because the `force transfer function`
        application differentiates the field in-place, ruining the contents
        for differentiation in the remaining :code:`D-1` spatial directions.
        Pre-allocated, but empty; any values in this field are discarded.
        Changed in-place. Local for each MPI rank--the full computaional grid
        is represented by the collective fields of all MPI ranks.
    compute_potential : bool, optional
        If :code:`True`, a :code:`D+1`-th copy of the Fourier transformed
        external potential field is made to be used later in particle-field
        energy calculation. If :code:`False`, only :code:`D` copies are made.

    See also
    --------
    compute_field_and_kinetic_energy :
        Compute the particle-field energy after the external potential is
        calculated.
    """
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
    """Compute the particle-field and kintic energy contributions

    Calculates the kinetic energy through

    .. math::

        E_k = \\sum_j \\frac{1}{2}m_j\\mathbf{v}_j\\cdot\\mathbf{v}_j,

    and the particle-field energy by

    .. math::

        E_\\text{field} = \\int\\mathrm{d}\\mathbf{r}\\,
            w[\\tilde\\phi],

    where :math:`w` is the interaction energy functional `density`.

    Parameters
    ----------
    phi : list[pmesh.pm.RealField]
        Pmesh :code:`RealField` objects containing discretized particle number
        density values on the computational grid; one for each particle type.
        Local for each MPI rank--the full computaional grid is represented by
        the collective fields of all MPI ranks.
    velocity : (N,D) numpy.ndarray
        Array of velocities for :code:`N` particles in :code:`D` dimensions.
        Local for each MPI rank.
    hamiltonian : Hamiltonian
        Particle-field interaction energy handler object. Defines the
        grid-independent filtering function, :math:`H`.
    positions : (N,D) numpy.ndarray
        Array of positions for :code:`N` particles in :code:`D` dimensions.
        Local for each MPI rank.
    types : (N,) numpy.ndarray
        Array of type indices for each of :code:`N` particles. Local for each
        MPI rank.
    v_ext : list[pmesh.pm.RealField]
        Pmesh :code:`RealField` objects containing discretized particle-field
        external potential values on the computational grid; one for each
        particle type. Local for each MPI rank--the full computaional grid is
        represented by the collective fields of all MPI ranks.
    config : Config
        Configuration object.
    layouts : list[pmesh.domain.Layout]
        Pmesh communication layout objects for domain decompositions of each
        particle type. Used as blueprint by :code:`pmesh.pm.readout` for
        exchange of particle information across MPI ranks as necessary.
    comm : mpi4py.Comm
        MPI communicator to use for rank commuication.

    See also
    --------
    update_field :
        Computes the up-to-date external potential for use in calculating the
        particle-field energy.
    """
    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells

    w = hamiltonian.w(phi) * volume_per_cell
    field_energy = w.csum()
    kinetic_energy = comm.allreduce(0.5 * config.mass * np.sum(velocity ** 2))
    return field_energy, kinetic_energy

def compute_field_energy_q_GPE(
    config,phi_eps_fourier,elec_field_contrib,phi_q_effective_fourier,
    phi_q_fourier,elec_energy_field, field_q_energy, comm=MPI.COMM_WORLD,
):
    """
    - added for general poisson equation (GPE)
    - used phi_q_effective_fourier (from iterative method) to calculate E_field, thus needed
    - multiply with actual charge density
    - PS: One can use the potential directly --> U = g*elec_potential
    """

    #COULK_GMX = 138.935458 # 1/(4pi eps0) config.dielectric_const

    #intermediate = np.abs(phi_q_fourier)*np.abs(phi_q_effective_fourier)
    ## ^ ----- For some reason it won't do this inside the transfer function
    ### even though they have the same shape

    #def transfer_energy(k,v):  ### potential field is electric field / (-ik)  --> potential field * q -->
    #    return 4.0 * np.pi * config.coulomb_constant * v / k.normp(p=2,zeromode=1)

    #intermediate.apply(transfer_energy,  kind='wavenumber', out=elec_energy_field)

    V = np.prod(config.box_size)

    #field_q_energy = 0.5 * V * comm.allreduce(np.sum(elec_energy_field.value))
    #print("field_q_E",field_q_energy)
    ### Add contribution from polarization
    #pol_energy = -0.5*(dielectric-phi_eps.readout(positions, layout=layout_q))*(elec_field_contrib.readout(positions, layout=layout_q))
    #pol_energy = comm.allreduce(np.sum(pol_energy))
    ### ^---- Alternative 1: From V_ext
    #print(pol_energy)

    pol_energy = 0.0
    eps_check = 1e-5 # to avoid invalid value in divide
    eps_0 = 1.0/config.coulomb_constant
    if eps_check < np.abs(np.min(phi_eps_fourier)):
        def transfer_pol_energy(k,v,epsk = phi_eps_fourier):
            return np.abs(v)**2  / (k.normp(p=2,zeromode=1))
        ##  ^----- Alternative 2: From its functional form W
        phi_q_fourier.apply(transfer_pol_energy,out = elec_energy_field)
        elec_energy_field = elec_energy_field/phi_eps_fourier
        elec_energy_field.c2r
        pol_energy = (0.5/eps_0) * V * comm.allreduce(np.sum(elec_energy_field.value))
    ### ^---- Alternative 2: From W_{ext}
        #print("pol_E",pol_energy.real)
    return pol_energy.real # field_q_energy.real + pol_energy.real

def update_field_force_q_GPE(conv_fun,phi, types, charges, dielectric, phi_q,
    phi_q_fourier,phi_eps, phi_eps_fourier,phi_q_eps, phi_q_eps_fourier,
    phi_q_effective_fourier,phi_eta,  phi_eta_fourier, phi_pol_prev,
    phi_pol, phi_pol_fourier,sum_fourier,phi_pol_temp,elec_field_fourier,
    elec_field,elec_forces,elec_field_contrib, layout_q, layout,pm,positions,
    config,comm = MPI.COMM_WORLD, compute_potential = False,
    E_field_from_potential = False,
):
    """
    - added for the general poisson equation (GPE) eletrostatics (follow PIC_Spectral_GPE)
    - this function get the electrostatic forces
    - refering to the test-pure-sphere-new.py, this inlcudes:
        [O] hpf_init_simple_gmx_units(grid_num,box_size,coords,charges,masses)
        ## ^----- define the pm and layout, already out outside this fucntion
        [Y] gen_qe_hpf_use_self(out_phiq_paraview_file)
        [Y] calc_phiq_fft_use_self_applyH_checkq(grid_num, out_phiq_paraview_file2 )
        [Y] poisson_solver(calc_energy, out_elec_field_paraview_file)
        [Y] compute_electric_field_on_particle()
        [Y] compute_electric_force_on_particle()
        [Y] compute electrostatic potential
    """
    ## basic setup
    V = np.prod(config.box_size)
    n_mesh_cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh_cells
    ## paint  ## pm.paint(positions[types == t], layout=layouts[t], out=phi[t])
    ## old protocol in gen_qe_hpf_use_self
    pm.paint(positions, layout=layout_q, mass=charges, out=phi_q) ##
    ## scale and fft
    ## old protocol in gen_qe_hpf_use_self
    phi_q /= volume_per_cell
    phi_q.r2c(out=phi_q_fourier)

    def phi_transfer_function(k, v):
        return v * np.exp(-0.5*config.sigma**2*k.normp(p=2, zeromode=1))

    phi_q_fourier.apply(phi_transfer_function, out=phi_q_fourier)
    ## ^------ use the same gaussian as the \kai interaciton
    ## ^------ tbr; phi_transfer_funciton by hamiltonian.H ??
    phi_q_fourier.c2r(out=phi_q) ## this phi_q is after applying the smearing function

    denom_phi_tot =  pm.create("real", value=0.0)
    num_types =  pm.create("real", value=0.0)

    #print("rank {:d} and phi shape {:}".format(comm.Get_rank(), np.shape(phi)))
    #for t_ in range(config.n_types):
    #    num_types = num_types  + (dielectric[t_])*phi[t_]
    #    denom_phi_tot = denom_phi_tot + phi[t_]
    #phi_eps = num_types/denom_phi_tot
    ### ^ ----- Calculate the relative dielectric (permittivity) to field
    ### ------- from a mean contribution of particle number densities

    for t_ in range(config.n_types): # do this with painting? (possible?)
        num_types = num_types  + (config.dielectric_type[t_])*phi[t_] #+ phi[t_].apply(dielectric_transfer_function, out = phi_eps)
        denom_phi_tot = denom_phi_tot + phi[t_]
    phi_eps = num_types/denom_phi_tot

    #print("shapes: denom, phi", np.shape(denom_phi_tot), np.shape(phi))
    #print("rank {:d} : max eps {:2f} , min eps {:2f}".format(comm.Get_rank(), np.max(phi_eps), np.min(phi_eps)))
    #print("mean", np.mean(phi_eps), "rank ", comm.Get_rank())
    phi_eps.r2c(out=phi_eps_fourier)
    ###  ^ ------  Alternatively with dielectric given by id list

    phi_q_eps = (phi_q/phi_eps)
    phi_q_eps.r2c(out = phi_q_eps_fourier)
    ##^ Get effective charge densities

    _SPACE_DIM = 3
    #COULK_GMX = 138.935458 # the 1/(4pi eps0) Gromacs units
    ##^--------- constants needed throughout the calculations

    ### method for finding the gradient (fourier space), using the spatial dimension of k
    for _d in np.arange(_SPACE_DIM):
        def gradient_transfer_function(k,x, d =_d):
            return  1j*k[_d]*x

        phi_eps_fourier.apply(gradient_transfer_function, out = phi_eta_fourier[_d])
        phi_eta_fourier[_d].c2r(out = phi_eta[_d])
        phi_eta[_d] = phi_eta[_d]/phi_eps # the eta param used in the iterative method

    ### iterative GPE solver ###
    ### ----------------------------------------------
    def iterate_apply_k_vec(k,additive_terms,d = _d):
        return additive_terms * (- 1j * k[_d]) / k.normp(p=2, zeromode=1)

    max_iter = 100; i = 0; delta = 1.0
    #phi_pol_prev = pm.create("real", value = 0.0)
    ### ^------ set to zero before each iterative procedure or soft start
    conv_criteria = config.conv_crit # conv. criteria (default 1e-6)
    w = config.pol_mixing # polarization mixing param (default 0.6)
    while (i < max_iter and delta > conv_criteria):
        (phi_q_eps + phi_pol_prev).r2c(out=sum_fourier)
        for _d in np.arange(_SPACE_DIM):
            sum_fourier.apply(iterate_apply_k_vec,out = phi_pol_fourier[_d])
            phi_pol_fourier[_d].c2r(out = phi_pol_temp[_d])

        phi_pol = -(phi_eta[0]*phi_pol_temp[0] + \
                     phi_eta[1]*phi_pol_temp[1] +  phi_eta[2]*phi_pol_temp[2]);
        ### ^-- Following a negative sign convention (-ik) of the FT, a neg sign is
        ### --- mathematically correct by the definition of the GPE
        phi_pol = w*phi_pol + (1.0-w)*phi_pol_prev
        diff = np.abs(phi_pol - phi_pol_prev)
        delta = conv_fun(comm,diff) # decided from toml input
        phi_pol_prev = phi_pol.copy()
        i = i + 1
    #print("Stopping after iteration {:d} with stop crit {:.2e}, delta {:.2e}".format(i,conv_criteria,delta))

    (phi_q_eps + phi_pol).r2c(out = phi_q_effective_fourier)
    for _d in np.arange(_SPACE_DIM):
        def poisson_transfer_function(k, v, d=_d): # fourier solution
            return - 1j * k[_d] * 4.0 * np.pi * config.coulomb_constant * v / k.normp(p=2,zeromode=1)
            ######return - 1j * k[_d] * 4.0 * np.pi * v /k.normp(p=2) #hymd.py:173: RuntimeWarning: invalid value encountered in true_divide
        phi_q_effective_fourier.apply(poisson_transfer_function, out = elec_field_fourier[_d])
        elec_field_fourier[_d].c2r(out=elec_field[_d])
    ##^--------- method 1: Solving the differential form of Gauss law (Coloumb) directly
    ###  with modified charge density
    ## electric field via solving poisson equation
    ## old protol in poisson_solver
    #compute_potential = True
    if compute_potential == True:
        def k_norm_divide(k, potential):
            return potential/k.normp(p=2, zeromode = 1)

        ## > Electrostatic potential
        eps0_inv = config.coulomb_constant*4*np.pi
        ## ^ the 1/(4pi eps0)*4*pi = 1/eps0
        elec_potential_fourier =  pm.create("complex", value = 0.0)
        elec_potential = pm.create("real", value = 0.0)
        ((eps0_inv)*(phi_q_eps + phi_pol)).r2c(out = elec_potential_fourier)
        elec_potential_fourier.apply(k_norm_divide, out = elec_potential_fourier)
        elec_potential_fourier.c2r(out = elec_potential)
        ### ^ electrostatic potential for the GPE

        ## calculate the electric field --> forces on the particles
        #E_field_from_potential = True
        if E_field_from_potential == True:
            for _d in np.arange(_SPACE_DIM):
                def gradient_transfer_function(k,x, d =_d):
                    return  -1j*k[_d]*x         ## negative sign relation here due to psi = - nabla E relation

                elec_potential_fourier.apply(gradient_transfer_function, out = elec_field_fourier[_d])
                elec_field_fourier[_d].c2r(out=elec_field[_d])
            ## ^-------- Method 2: Solving the poisson equation i.e from electrostatic potential
            ## Assuming the electric field is conserved.
            ## If we assume no magnetic flux (magnetic induced fields)


    ## calculate electric forces on particles
    elec_field_contrib_fourier = pm.create("complex", value=0.0)
    elec_field_contrib_fgrad = [pm.create("complex",value = 0.0) for _ in range(_SPACE_DIM)]
    phi_eps_fgrad = [pm.create("complex",value = 0.0) for _ in range(_SPACE_DIM)]
    phi_eps_grad =  [pm.create("real",value = 0.0) for _ in range(_SPACE_DIM)]
    elec_field_contrib_grad = [pm.create("real",value = 0.0) for _ in range(_SPACE_DIM)]

    elec_field_contrib = (elec_field[0]*elec_field[0] + \
                 elec_field[1]*elec_field[1] +  elec_field[2]*elec_field[2])/denom_phi_tot;
    #print("max E field val {:.2f} rank {:d}".format(np.max(elec_field),comm.Get_rank()))
    elec_field_contrib.r2c(out = elec_field_contrib_fourier)

    for _d in np.arange(_SPACE_DIM):
        def gradient_transfer_function(k,x, d =_d):
            return  1j * k[_d] * x        ## derivative
        elec_field_contrib_fourier.apply(gradient_transfer_function,out = elec_field_contrib_fgrad[_d])
        phi_eps_fourier.apply(gradient_transfer_function,out = phi_eps_fgrad[_d])
        elec_field_contrib_fgrad[_d].c2r(out = elec_field_contrib_grad[_d])
        phi_eps_fgrad[_d].c2r(phi_eps_grad[_d])


    eps0_inv = config.coulomb_constant*4*np.pi
    #sums = np.zeros((config.n_types,3))
    #elec_forces2 = np.zeros_like(positions) # check difference
    #if comm.Get_rank() == 0:

    for _d in np.arange(_SPACE_DIM):
        elec_forces[:,_d] =  charges*(elec_field[_d].readout(positions, layout=layout_q)) \
                                + (0.5 / eps0_inv) * (- ((phi_eps_grad[_d].readout(positions, layout=layout_q)) \
                                   * (elec_field_contrib).readout(positions, layout=layout_q)) \
                                    +  ((dielectric - (phi_eps.readout(positions, layout=layout_q))) \
                                    * (elec_field_contrib_grad[_d]).readout(positions, layout=layout_q)))

    #print(np.shape(dielectric[types == 3]))
    #print(np.shape(charges[types == 3]))
    #print(types == 4)
    #print(config.name_to_type_map)
    #for t_ in range(config.n_types):
    #    sum_elec_mesh = np.zeros(3)
    #    rank = comm.Get_rank()
    #    comm.Reduce(sums[t_,:], sum_elec_mesh,
    #    op=MPI.SUM, root=0)
    #    #print("sums", sums)
    #    if rank == 0:
    #        print("type",t_)
    #        print("total sums", sum_elec_mesh)

    #print(np.shape(elec_forces[:,2]))
    ###^------ here the use the column, as the elec_forces are defined as (N,3) dimension
    #print("field",elec_forces[:,2])
    #print(config.particles_id)

    return elec_field_contrib, phi_eps_fourier



def domain_decomposition(
    positions, pm, *args, molecules=None, bonds=None, verbose=0,
    comm=MPI.COMM_WORLD
):
    """Performs domain decomposition

    Rearranges the MPI rank memory layout into one that better mimics the PFFT
    pencil grid 2D decomposition. As molecules are always required to be fully
    contained on a single MPI rank, a perfect decomposition is not always
    possible. The best decomposition which leaves the center of mass of all
    molecules in their respective correct domains is used, with possible
    overlap into neighbouring domains if the spatial extent of any molecule
    crosses domain boundaries.

    Parameters
    ----------
    positions : (N,D) numpy.ndarray
        Array of positions for :code:`N` particles in :code:`D` dimensions.
        Local for each MPI rank.
    pm : pmesh.pm.ParticleMesh
        Pmesh :code:`ParticleMesh` object interfacing to the CIC window
        function and the PFFT discrete Fourier transform library.
    *args
        Variable length argument list containing arrays to include in the
        domain decomposition.
    molecules : (N,) numpy.ndarray, optional
        Array of integer molecule affiliation for each of :code:`N` particles.
        Global (across all MPI ranks) or local (local indices on this MPI rank
        only) may be used, both, without affecting the result.
    bonds : (N,M) numpy.ndarray, optional
        Array of :code:`M` bonds originating from each of :code:`N` particles.
    verbose : int, optional
        Specify the logging event verbosity of this function.
    comm : mpi4py.Comm
        MPI communicator to use for rank commuication.

    Returns
    -------
    Domain decomposed :code:`positions` and any array specified in
    :code:`*args`, in addition to :code:`bonds` and :code:`molecules` if these
    arrays were provided as input.
    """
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
