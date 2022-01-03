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
