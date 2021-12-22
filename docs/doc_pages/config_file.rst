.. _config-label:

Configuration file
##################
The HyMD configuration file specifies all aspects of the simulation, including
temperature coupling, bond specifications and strengths, the hPF interaction
energy potential, the integrator, and more. The keywords may be given in any
order.

The configuration file is a `toml`_ format file with keywords (strings) and
values (strings, numbers, arrays, etc.) which is serialised and parsed to
configure simulation runs.

.. _`toml`:
   https://github.com/toml-lang/toml

.. role:: required

Configuration file keywords
---------------------------

The following constitutes every possible keyword specification in HyMD
configuration files:

Metadata keywords
^^^^^^^^^^^^^^^^^
Configuration keywords specifying metadata for the H5MD output trajectory.

:name:
   :code:`string` [**optional**, default: :code:`None`]

   Name for the simulation. Optional keyword specifying a name in the metadata for the H5MD output trajectory/energy file.


:tags:
   :code:`array` [:code:`string`] [**optional**, default: :code:`None`]

   Tags for the simulation. Optional keyword specifying metadata tags for the H5MD output trajectory/energy file.


Particle keywords
^^^^^^^^^^^^^^^^^
Configuration keywords specifying properties of particles and molecules in the
system.

:n_particles:
   :code:`integer` [**optional**, default: :code:`None`]

   Specifies the total number of particles in the input. Optional keyword for validation, ensuring the input HDF5 topology has the correct number of particles and molecules.

:mass:
   :code:`float` [**optional**, default: :code:`72.0`] {units: :math:`\text{u}=\text{g}\,\text{mol}^{-1}`}

   Mass of the particles in the simulation. Unless masses are provided in the HDF5 topology/structure file, all masses are assumed equal.

:max_molecule_size:
   :code:`integer` [**optional**, default: :code:`201`]

   Maximum size of any single molecule in the system. Used to speed up distribution of particles onto MPI ranks in a parallel fashion. The code **will** crash if there are molecules larger than :code:`max_molecule_size` in the topology/structure input file.


Simulation keywords
^^^^^^^^^^^^^^^^^^^
Configuration keywords specifying simulation parameters.

:n_steps:
   :code:`integer` [:required:`required`]

   Total number of integration steps to perform for this simulation.

:n_print:
   :code:`integer` or :code:`boolean` [**optional**, default: :code:`False`]

   Frequency of trajectory/energy output to the H5MD trajectory/energy output file (in units of *number of time steps*).

:n_flush:
   :code:`integer` [**optional**, default: :code:`None`]

   Frequency of HDF5 write buffer flush, forcing trajectory/energy to be written to disk (in units of *number of* :code:`n_print`, i.e. :code:`n_print = 13`, :code:`n_flush = 3` would result in the trajectory/energy being written to disk every :code:`13 x 3 = 39` *simulation steps*)

:time_step:
   :code:`float` [:required:`required`] {units: :math:`\text{ps}=10^{-12}~\text{s}`}

   The time step per integration step. If the rRESPA multiple time step integrator is used (:code:`integrator = "respa"`), this specifies the **inner** time step, i.e. the step size used for the intramolecular force integration. In that case, the step size for the field force impulses is :code:`respa_inner` times the :code:`time_step` , and the total simulation time is :code:`n_steps` times :code:`time_step` times :code:`respa_inner` .

:box_size:
   :code:`array` [:code:`float`]  [:required:`required`] {units: :math:`\text{nm}=10^{-9}~\text{m}`}

   Size of the simulation box. Any particle outside the box is placed inside before the first time step is integrated, meaning no particles will be lost if the box size is specified too small, but the system may nevertheless be unstable.

:integrator:
   :code:`string` [:required:`required`] (options: :code:`velocity-verlet` or :code:`respa`)

   Specifies the time integrator used in the simulation. If :code:`respa`, the *reversible reference system propagator algorithm (rRESPA)* :cite:`tuckerman1992reversible` integrator is used, with :code:`respa_inner` number of inner (intramolecular force) time steps. If :code:`velocity-verlet`, a normal Velocity Verlet integrator is used. If :code:`respa` and :code:`respa_inner = 1`, the rRESPA integrator is equivalent to the Velocity Verlet.

:respa_inner:
   :code:`integer` [**optional**, default: :code:`1`]

   The number of inner time steps in the rRESPA integrator. This denotes the number of intramolecular force calculations (stretching, bending, torsional) are performed between each impulse applied from the field forces.

:domain_decomposition:
   :code:`integer` or :code:`boolean` [**optional**, default: :code:`False`]

   Specifies the interval (in time steps) of *domain decomposition exchange*, involving all MPI ranks sending and receiving particles according to the particles' positions in the integration box and the MPI ranks' assigned domain. Performing the decomposition is expensive in terms of MPI communication cost, but may reduce the communication of particle positions across MPI rank boundaries for some time during simulation. If :code:`True`, the decomposition is performed once (before starting the simulation). If :code:`False`, no decomposition is performed.

:cancel_com_momentum:
   :code:`integer` or :code:`boolean` [**optional**, default: :code:`False`]

   If :code:`True`, the total linear momentum of the center of mass is removed before starting the simulation. If an integer is specifed, the total linear momentum of the center of mass is removed every :code:`remove_com_momentum` time steps. If :code:`False`, the linear momentum is never removed.

:start_temperature:
   :code:`float` or :code:`boolean` [**optional**, default: :code:`False`] {units: :math:`\text{K}`}

   Generate starting temperature by assigning all particle velocities randomly according to the Maxwell-Boltzmann distribution at :code:`start_temperature` Kelvin prior to starting the simulation. If :code:`False`, the velocities are not changed before starting the simulation.

:target_temperature:
   :code:`float` or :code:`boolean` [**optional**, default: :code:`False`] {units: :math:`\text{K}`}

   Couple the system to a heat bath at :code:`target_temperature` Kelvin by applying a *Canonical sampling through velocity rescaling* :cite:`Bussi2007JCP` thermostat with coupling strength :code:`tau`. If :code:`False`, no temperature control is applied.

:tau:
   :code:`float` [**optional**, default: :code:`0.7`] {units: :math:`\text{ps}=10^{-12}~\text{s}`}

   The time scale of the CSVR thermostat coupling. In the limit of :code:`tau → ∞`, the Hamiltonian dynamics are preserved and no temperature coupling takes place.

:thermostat_coupling_groups:
   :code:`array` [:code:`array` [:code:`string`]] [**optional**, default: :code:`[]`]

   Specifies individual groups coupling independently to the CSVR thermostat. E.g. in a system containing :code:`"A"`, :code:`"B"`, and :code:`"C"` type particles, :code:`thermostat_coupling_groups = [["A", "B"], ["C"],]` would thermalise types :code:`"A"` and :code:`"B"` together and couple :code:`"C"` type particles to a different thermostat (all individual thermostats are at the same temperature, i.e. :code:`target_temperature` Kelvin).
:hamiltonian:
   :code:`string` [**optional**, default: :code:`"DefaultNohChi"`] (options: :code:`SquaredPhi`, :code:`DefaultNohChi`, or :code:`DefaultWithChi`)

   Specifies the interaction energy functional :math:`W[\tilde\phi]` for use with the particle-field interactions. See :ref:`functionals-label` for details.


Field keywords
^^^^^^^^^^^^^^
Configuration keywords specifying field parameters.

:mesh_size:
   :code:`array` [:code:`integer`] or :code:`integer` [:required:`required`]

   Either an integer or an array of three integers specifying the mesh grid size to use in each of the three spatial directions for the FFT operations. The grid spacing is :code:`box_size / mesh_size`.

:kappa:
   :code:`float` [:required:`required`] {units: :math:`\text{kJ}^{-1}\text{mol}`}

   Compressibility parameter used in the relaxed incompressibility term in the interaction energy functional :math:`W[\tilde\phi]`. See :ref:`functionals-label` for more details.

:sigma:
   :code:`float` [:required:`required`] {units: :math:`\text{nm}=10^{-9}~\text{m}`}

   Filter width, representing the effective coarse-graining level of the particles in the simulation. If a Gaussian filter is used (by default), this specifies the standard deviation. See :ref:`filtering-label` for more details.

:chi:
   :code:`array` [:code:`array` [:code:`string`, :code:`float`]] [**optional**, default: :code:`[]`] {units: :math:`\text{kJ}\,\text{mol}^{-1}`}

   Array of :math:`\tilde\chi_\text{AB}`-parameters indicating the strength of the repulsive or attractive interaction between particles of type :code:`"A"` and the number density due to particles of type :code:`"B"`, and vice versa. Example: :code:`chi = [ ["A", "B", 7.28], ["C", "D", -1.32] ]` would specify a repulsive :code:`"A"`--:code:`"B"` type interaction, and a weakly attractive :code:`"C"`--:code:`"D"` interaction. Self-interaction :math:`\tilde\chi`-terms are always zero. See :ref:`functionals-label` for more details.


Bond keywords
^^^^^^^^^^^^^
Configuration keywords specifying bonds and bonds parameters.

:bonds:
   :code:`array` [:code:`array` [2 :code:`string`, 2 :code:`float`]] [**optional**, default: :code:`[]`] {units: :math:`\text{nm}=10^{-9}~\text{m}` and :math:`\text{kJ}\,\text{mol}^{-1}`}

   Specifies harmonic stretching potentials between particles in the same molecule. Each entry in the array specifies one bond between two types of particles, followed by the equilibrium distance and the bond stiffness. Example: :code:`bonds = [ ["A", "A", 0.47, 980.0], ["A", "B", 0.31, 1250.0] ]` indicates a :code:`"A"`--:code:`"A"` bond of equilibrium length :code:`0.47` :math:`\text{nm}` with a bond strength of :code:`980.0` :math:`\text{kJ}\,\text{mol}^{-1}` and a corresponding bond between :code:`"A"` and :code:`"B"` type particles with equilibrium length :code:`0.31` :math:`\text{nm}` and a strength of :code:`1250.0` :math:`\text{kJ}\,\text{mol}^{-1}`. See :ref:`bonds-label` for details about how to specify stretching bonds.

:angle_bonds:
  :code:`array` [:code:`array` [3 :code:`string`, 2 :code:`float`]] [**optional**, default: :code:`[]`] {units: :math:`{}^\circ` and :math:`\text{kJ}\,\text{mol}^{-1}`}

   Specifies harmonic angular bending potentials between particles in the same molecule. Each entry in the array specifies one bond between three types of particles, followed by the equilibrium angle (in degrees) and the angle bond stiffness. Example: :code:`bonds = [ ["A", "A", "A", 180.0, 55.0], ["A", "B", "B", 120.0, 25.0] ]` indicates a :code:`"A"`--:code:`"A"`--:code:`"A"` angular bond that tries to keep the bond linear at :code:`180` degrees with a bending strength of :code:`55.0` :math:`\text{kJ}\,\text{mol}^{-1}` and a corresponding angle bond between :code:`"A"`--:code:`"B"`--:code:`"B"` prefering the angle at :code:`120` degrees and a strength of :code:`25.0` :math:`\text{kJ}\,\text{mol}^{-1}`. See :ref:`bonds-label` for details about how to specify angular bonds.

:dihedrals:
   :code:`array` [:code:`array` [4 :code:`string`, :code:`integer`, :code:`COSINE SERIES` ]] [**optional**, default: :code:`[]`] {units: :math:`\text{kJ}\,\text{mol}^{-1}\text{rad}^{-2}`}

   Specifies four-particle torsional potentials by cosine series. See :ref:`bonds-label` for details about how to specify dihedrals.



Electrostatic keywords
^^^^^^^^^^^^^^^^^^^^^^
Configuration keywords specifying electrostatics and electrostatic parameters.

:coulombtype:
   :code:`string` [**optional**, default: :code:`None`] (options: :code:`PIC_Spectral`)

   Specifies the type of electrostatic Coulomb interactions in the system. The strength of the electrostatic forces is modulated by the relative dielectric constant of the simulation medium, specified with the :code:`dielectric_const` keyword. Charges for individual particles are specified in the structure/topology HDF5 input file, *not* in the configuration file. If no charges (or peptide backbone dipoles) are present, the electrostatic forces will not be calculated even if this keyword is set to `PIC_Spectral`.

:dielectric_const:
   :code:`float` [**optional**, default: :code:`None`]

   Specifies the relative dielectric constant of the simulation medium which regulates the strength of the electrostatic interactions. When using helical propensity dihedrals, this keyword must be specified---even if electrostatics are not included with the :code:`coulombtype` keyword.
