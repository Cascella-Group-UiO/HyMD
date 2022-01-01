.. _examples-label:

Examples
########
The following examples are ordered in *more or less* order of increasing
complexity, adding pieces at each step until we arrive at models for realistic
soft matter systems. All files (input and simulation output trajectories) are
available in the `HyMD-tutorial`_ github repository.

.. _HyMD-tutorial:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial

Ideal gas
=========
.. image:: https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/ideal_gas/ideal_gas.png?raw=true
   :align: right
   :width: 200

The simplest possible system is one with no interactions at all---intramolecular
or intermolecular. A minimal configuration file for a non-interacting system
may look like:


.. code-block:: toml
   :caption: **ideal_gas.toml**

   [simulation]
   integrator = "velocity-verlet"
   n_steps = 100                      # total simulation steps
   n_print = 10                       # output trajectory data every n_print steps
   time_step = 0.01                   # time step in ps
   box_size = [5.0, 5.0, 5.0]         # simulation box size in nm
   start_temperature = 300            # generate starting temperature in Kelvin
   target_temperature = false         # thermostat temperature in Kelvin
   tau = 0.1                          # thermostat coupling strength in ps
   hamiltonian = "defaultnochi"       # interaction energy functional
   kappa = 0.05                       # compressibility in mol/kJ

   [field]
   mesh_size = [1, 1, 1]              # FFT grid size
   sigma = 1.0                        # filtering length scale in nm

A simple input structure/topology file :code:`ideal_gas.HDF5` is available in
the `HyMD-tutorial/ideal_gas`_ github repository (along with the code to
generate it). Run the (very) short simulation by

.. code:: bash

   python3 -m hymd ideal_gas.toml ideal_gas.HDF5 --disable-field   \
                                                 --velocity-output \
                                                 --force-output

with the :code:`--disable-field` argument to completely turn off all
particle--field interactions. Adding the options to output the velocities and
forces to the trajectory file, enables examining the ballistic motion of the
particles. See the accompanying Jupyter notebook `ideal_gas.ipynb`_ for details.


.. _HyMD-tutorial/ideal_gas:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/tree/main/ideal_gas
.. _ideal_gas.ipynb:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/ideal_gas/ideal_gas.ipynb


Ideal chains
============
.. image:: https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/ideal_chain/100ps.png?raw=true
   :align: right
   :width: 200

The simplest available interaction is the harmonic two-particle bond (see
:ref:`bonds-label`),

.. math::

   V_2(r) = \frac{k}{2}(r-r_0)^2.

Extending the configuration file from the ideal gas case, we need to add a
:code:`bonds` specification. Note that the :code:`[bonds]` meta specifier is not
necessary, but may be used to help organise the input file.

.. code-block:: toml
   :caption: **ideal_chain.toml**

   [simulation]
   integrator = "velocity-verlet"
   n_steps = 10000                    # total simulation steps
   n_print = 200                      # output trajectory data every n_print steps
   time_step = 0.01                   # time step in ps
   box_size = [30.0, 30.0, 30.0]      # simulation box size in nm
   start_temperature = 50             # generate starting temperature in Kelvin
   target_temperature = 300           # thermostat temperature in Kelvin
   tau = 0.1                          # thermostat coupling strength in ps
   hamiltonian = "defaultnochi"       # interaction energy functional
   kappa = 0.05                       # compressibility in mol/kJ

   [field]
   mesh_size = [1, 1, 1]              # FFT grid size
   sigma = 1.0                        # filtering length scale in nm

   [bonds]
   bonds = [
     ["A", "A", 0.5, 1000.0],         # (i name, j name, equilibrium length, strength)
   ]

A simple input structure/topology file :code:`ideal_chain.HDF5` with a few
coiled up ideal chain polymers is available in the
`HyMD-tutorial/ideal_chain`_ github repository (along with the code to
generate it). Running the simulation with

.. code:: bash

   python3 -m hymd ideal_chain.toml ideal_chain.HDF5 --disable-field

we may examine the radius of gyration of the individual polymer chains. An
example of this is shown in the accompanying Jupyter notebook
`ideal_chain.ipynb`_.


.. _HyMD-tutorial/ideal_chain:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/tree/main/ideal_chain
.. _ideal_chain.ipynb:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/ideal_chain/ideal_chain.ipynb


Stiff rods
==========
.. image:: https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/rods/rods.png?raw=true
   :align: right
   :width: 200

Having considered two-particle bonds, the next step is three-particle angular
bonds depending on the :code:`i--j--k` angle :math:`\theta` (see
:ref:`bonds-label`),

.. math::

   V_3(\theta) = \frac{k}{2}(\theta-\theta_0)^2.

Extending the configuration file from the ideal chain case, we need to add a
:code:`angle_bonds` specification. Note that the :code:`[bonds]` meta specifier
is not necessary, but may be used to help organise the input file.

.. code-block:: toml
   :caption: **rods.toml**

   [simulation]
   integrator = "velocity-verlet"
   n_steps = 10000                    # total simulation steps
   n_print = 200                      # output trajectory data every n_print steps
   time_step = 0.01                   # time step in ps
   box_size = [30.0, 30.0, 30.0]      # simulation box size in nm
   start_temperature = 50             # generate starting temperature in Kelvin
   target_temperature = 300           # thermostat temperature in Kelvin
   tau = 0.1                          # thermostat coupling strength in ps
   hamiltonian = "defaultnochi"       # interaction energy functional
   kappa = 0.05                       # compressibility in mol/kJ

   [field]
   mesh_size = [1, 1, 1]              # FFT grid size
   sigma = 1.0                        # filtering length scale in nm

   [bonds]
   bonds = [
     ["A", "A", 0.5, 1000.0],         # (i name, j name, equilibrium length, strength)
   ]

   angle_bonds = [
     ["A", "A", "A", 180.0, 100.0],   # (i, j, k, equilibrium angle, strength)
   ]


A simple input structure/topology file :code:`rods.HDF5` with a few
coiled up polymer chains is available in the `HyMD-tutorial/rods`_ github
repository (along with the code to generate it). Running the simulation with

.. code:: bash

   python3 -m hymd rods.toml rods.HDF5 --disable-field

the polymer chains extend into rod-like conformations. We may examine the
radius of gyration of the individual polymer chains, and compare it to the
gyration radii of the ideal chain case. An example of this is shown in the
accompanying Jupyter notebook `rods.ipynb`_.


.. _HyMD-tutorial/rods:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/tree/main/rods
.. _rods.ipynb:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/rods/rods.ipynb



Helixes
=======
.. image:: https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/helixes/helixes.png?raw=true
   :align: right
   :width: 200

Having considered two- and three-particle bonds, the next step is dihedral
four-particle angular bonds, depending on the angle :math:`\phi` between the
:code:`i--j--k` plane and the :code:`j--k--l` plane (see :ref:`bonds-label`),

.. math::

   V_4(\phi) = \frac{1}{2}\sum_n^M c_n\cos(n\phi+\phi_n).

The Cosine series defining the strength and equilibrium conditions of the bond
are given as input in a :code:`dihedrals` keyword in the configuration file.
The helical dihedral bond is designed for use with peptides and topological
dipole reconstruction, so we need to specify the :code:`dielectric_const`
keyword even though we are not including electrostatic forces in the current
simulation. Note that the :code:`[bonds]` meta specifier is not necessary, but
may be used to help organise the input file.

.. code-block:: toml
   :caption: **helixes.toml**

   [simulation]
   integrator = "velocity-verlet"
   n_steps = 10000                    # total simulation steps
   n_print = 200                      # output trajectory data every n_print steps
   time_step = 0.01                   # time step in ps
   box_size = [30.0, 30.0, 30.0]      # simulation box size in nm
   start_temperature = 50             # generate starting temperature in Kelvin
   target_temperature = 300           # thermostat temperature in Kelvin
   tau = 0.1                          # thermostat coupling strength in ps
   hamiltonian = "defaultnochi"       # interaction energy functional
   kappa = 0.05                       # compressibility in mol/kJ
   dielectric_const = 15.0

   [field]
   mesh_size = [1, 1, 1]              # FFT grid size
   sigma = 1.0                        # filtering length scale in nm

   [bonds]
   bonds = [
     ["A", "A", 0.31, 10000.0],       # (i name, j name, equilibrium length, strength)
   ]

   dihedrals = [
     [
       ["A", "A", "A", "A"],
       [
         [-1],
         [449.08790868, 610.2408724, -544.48626121, 251.59427866, -84.9918564],
         [0.08, 0.46, 1.65, -0.96, 0.38],
       ],
       [1.0]
     ],
   ]

A simple input structure/topology file :code:`helixes.HDF5` with a few
coiled up polymer chains is available in the `HyMD-tutorial/rods`_ github
repository (along with the code to generate it). Running the simulation with

.. code:: bash

   python3 -m hymd rods.toml rods.HDF5 --disable-field

the polymer chains extend into helical conformations. An example of this is
shown in the accompanying Jupyter notebook `helixes.ipynb`_.


.. _HyMD-tutorial/helixes:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/tree/main/helixes
.. _helixes.ipynb:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/helixes/helixes.ipynb


Phase separation
================
.. image:: https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/phase_separation/chi=40_final.png?raw=true
   :align: right
   :width: 200

The simplest field interaction available in HyMD is the interaction between two
monoatomic particles of types :code:`A` and :code:`B`. Using the Hamiltonian

.. math::

   \mathcal{H}=H_0+W

with :math:`\tilde\chi`--dependent interactions defined by (specified in the
configuration file by :code:`hamiltonian = DefaultWithChi`):

.. math::

   W=\frac{1}{\phi_0}\int\mathrm{d}\mathbf{r}\tilde\chi_{\text{A}-\text{B}}\tilde\phi_\text{A}(\mathbf{r})\tilde\phi_\text{B}(\mathbf{r}) + \frac{1}{2\kappa}\left(\tilde\phi_\text{A}(\mathbf{r})+\tilde\phi_\text{B}\mathbf{r}-\phi_0\right)^2.


.. code-block:: toml
   :caption: **phase_separation.toml**

   [simulation]
   integrator = "velocity-verlet"
   n_steps = 10000                    # total simulation steps
   n_print = 2000                     # output trajectory data every n_print steps
   time_step = 0.01                   # time step in ps
   box_size = [5.0, 5.0, 5.0]         # simulation box size in nm
   start_temperature = 300            # generate starting temperature in Kelvin
   target_temperature = 300           # thermostat temperature in Kelvin
   tau = 0.1                          # thermostat coupling strength in ps
   hamiltonian = "defaultwithchi"     # interaction energy functional

   [field]
   mesh_size = [20, 20, 20]           # FFT grid size
   sigma = 1.0                        # filtering length scale in nm
   kappa = 0.05                       # compressibility in mol/kJ
   chi = [
     ['A', 'B', 5.0],                 # (name i, name j, strength)
   ]

A simple input structure/topology file :code:`phase_separation.HDF5` containing
an equal number of :code:`A` type and :code:`B` type particles is available in
the `HyMD-tutorial/phase_separation`_ github repository (along with the code to
generate it). Running the simulation with

.. code:: bash

   python3 -m hymd phase_separation.toml phase_separation.HDF5

we may examine the resulting trajectory. In the case of a low interaction
strength :math:`\tilde\chi` of :code:`5.0` (below the critical value of
separation) the system remains mixed. However, raising the interaction strength
to :code:`40.0` (well above the critical value) yields a phase separated system.
This may be elucidated by considering the radial distribution function in each
case. An example of this is shown in the accompanying Jupyter notebook
`phase_separation.ipynb`_.


.. _HyMD-tutorial/phase_separation:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/tree/main/phase_separation
.. _phase_separation.ipynb:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/phase_separation/phase_separation.ipynb


Diblock copolymers
==================
.. image:: https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/copolymer/copolymer_final.png?raw=true
   :align: right
   :width: 200

Having introduced particle--field interactions, we may now combine bonded and
field terms and make a model of diblock copolymers phase separating under
positive :math:`\tilde\chi`--interactions. This simple model contains two- and
three-particle harmonic bonds as well. With the combination of bonded and field
interactions, we may also introduce the rRESPA multiple time step integator with
the :code:`integrator = "respa"` keyword. Putting the pieces together, the
configuration file may look like the following:

.. code-block:: toml
   :caption: **copolymer.toml**

   [simulation]
   integrator = "respa"
   respa_inner = 15
   n_steps = 2000                     # total simulation steps
   n_print = 2000                     # output trajectory data every n_print steps
   time_step = 0.01                   # time step in ps
   box_size = [10.0, 10.0, 10.0]      # simulation box size in nm
   start_temperature = 50             # generate starting temperature in Kelvin
   target_temperature = 300           # thermostat temperature in Kelvin
   tau = 0.1                          # thermostat coupling strength in ps
   hamiltonian = "defaultwithchi"     # interaction energy functional
   kappa = 0.05                       # compressibility in mol/kJ

   [field]
   mesh_size = [50, 50, 50]           # FFT grid size
   sigma = 0.5                        # filtering length scale in nm
   chi = [
     ["A", "B", 30.0],                # (i, j, strength)
   ]

   [bonds]
   bonds = [
     ["A", "A", 0.25, 1500.0],        # (i, j, equilibrium length, strength)
     ["A", "B", 0.25, 1500.0],
     ["B", "B", 0.25, 1500.0],
   ]
   angle_bonds = [
     ["A", "A", "A", 180.0, 25.0],    # (i, j, k, equilibrium angle, strength)
     ["B", "B", "B", 180.0, 25.0],
   ]

A simple input structure/topology file :code:`copolymer.HDF5` with a few
coiled up polymer chains is available in the `HyMD-tutorial/copolymer`_ github
repository (along with the code to generate it). Each polymer chain is 20
particles long, with ten :code:`A` followed by ten :code:`B` type particles.
Running the simulation with

.. code:: bash

   python3 -m hymd copolymer.toml copolymer.HDF5

the polymer chains extend into rod-like conformations in a phase-separating
manner. An example of this is shown in the accompanying Jupyter notebook
`copolymer.ipynb`_.


.. _HyMD-tutorial/copolymer:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/tree/main/copolymer
.. _copolymer.ipynb:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/copolymer/copolymer.ipynb


Lipid bilayer self-assembly
===========================
.. image:: https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/lipid_self_assembly/bilayer.png?raw=true
   :align: right
   :width: 300

Putting together the same pieces as in the diblock copolymer case, we may setup
a model of a phospholipid (DPPC) which self-assembles into a bilayer
conformation. The same ingredients (as in the copolymer system) are present in
the configuration file; harmonic bonds, angular bonds, and field--particle
interactions. In this case, we couple a different thermostat to the solvent and
the lipids via the :code:`thermostat_coupling_groups` keyword. With parameters
optimised in :cite:`ledum2020automated`, the configuration file looks like:


.. code-block:: toml
   :caption: **lipid_self_assembly.toml**

   [simulation]
   integrator = "respa"
   respa_inner = 25
   n_steps = 3000
   n_print = 200
   n_flush = 1
   time_step = 0.01
   box_size = [9.96924, 9.96924, 10.03970]
   start_temperature = 323
   target_temperature = 323
   tau = 0.1
   hamiltonian = "defaultwithchi"
   kappa = 0.05
   domain_decomposition = 10001
   cancel_com_momentum = true
   max_molecule_size = 15
   thermostat_coupling_groups = [
     ["N", "P", "G", "C"],
     ["W"],
   ]

   [field]
   mesh_size = [25, 25, 25]
   sigma = 1.0
   chi = [
     ["C", "W", 42.24],
     ["G", "C", 10.47],
     ["N", "W", -3.77],
     ["G", "W", 4.53],
     ["N", "P", -9.34],
     ["P", "G", 8.04],
     ["N", "G", 1.97],
     ["P", "C", 14.72],
     ["P", "W", -1.51],
     ["N", "C", 13.56],
   ]

   [bonds]
   bonds = [
     ["N", "P", 0.47, 1250.0],
     ["P", "G", 0.47, 1250.0],
     ["G", "G", 0.37, 1250.0],
     ["G", "C", 0.47, 1250.0],
     ["C", "C", 0.47, 1250.0],
   ]
   angle_bonds = [
     ["P", "G", "G", 120.0, 25.0],
     ["P", "G", "C", 180.0, 25.0],
     ["G", "C", "C", 180.0, 25.0],
     ["C", "C", "C", 180.0, 25.0],
   ]


A randomly mixed input structure/topology file
:code:`lipid_self_assembly.HDF5` with a few hundred lipids in a roughy
:code:`10 x 10 x 10nm` box is available in the
`HyMD-tutorial/lipid_self_assembly`_ github repository. The simulation box was
previously equilibrated in the Martini model before subsequent mixing of the
contents. Running the simulation

.. code:: bash

   python3 -m hymd lipid_self_assembly.toml lipid_self_assembly.HDF5

we can observe spontaneous aggregation into a bilayer conformation in the
sub-nanosecond regime. An example of this is shown in the accompanying Jupyter
notebook `lipid_self_assembly.ipynb`_.


.. _HyMD-tutorial/lipid_self_assembly:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/tree/main/lipid_self_assembly
.. _lipid_self_assembly.ipynb:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/lipid_self_assembly/lipid_self_assembly.ipynb


Peptide in lipid bilayer
========================
.. image:: https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/peptide/peptide.png?raw=true
   :align: right
   :width: 300

Next, we combine the dihedral helical propensity and the lipid model, and model
a single homopolypeptide consisting of alanine amino acids embedded inside a
DOPC phospholipid bilayer.


The `configuration file`_ is available in the `HyMD-tutorial/peptide`_ github
repository (omitted here for brevity).

A pre-equilibrated input structure/topology file :code:`peptide.HDF5` with a
few hundred lipids in a roughy :code:`20 x 20 x 20nm` box is available in the
`HyMD-tutorial/peptide`_ github repository. Running the simulation

.. code:: bash

   python3 -m hymd peptide.toml peptide.HDF5

we observe the peptide embedded laterally in the membrane in a stable
configuration.  An example of this is shown in the accompanying Jupyter
notebook `peptide.ipynb`_.


.. _HyMD-tutorial/peptide:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/tree/main/peptide
.. _configuration file:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/peptide/peptide.toml
.. _peptide.ipynb:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/peptide/peptide.ipynb


SDS
===
.. image:: https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/sds/oblate.png?raw=true
   :align: right
   :width: 300

The last part we introduce is explicit electrostatic interactions through
particle--mesh Ewald. Enabling the electrostatic calculations is done via the
:code:`coulombtype` keyword. :code:`"PIC_Spectral"` is the only supported
value in the released version of the code (more general variants are in
development). We specify the relative dielectric constant of the simulation
medium via the :code:`dielectric_const` keyword. As an example system, we use
a model for sodium dodecyl sulfate (SDS), consiting of short four-particle
chains with the head group carrying charge of negative one. We add sodium
counter-ions to ensure stability of the Ewald scheme by neutralising the total
system charge.



.. code-block:: toml
   :caption: **sds.toml**

   [simulation]
   integrator = "respa"
   respa_inner = 10
   n_steps = 3000
   n_print = 1500
   n_flush =
   time_step = 0.01
   box_size = [8.3, 8.3, 8.3]
   start_temperature = 298
   target_temperature = 298
   tau = 0.1
   hamiltonian = "defaultwithchi"
   kappa = 0.1
   domain_decomposition = 10000
   cancel_com_momentum = true
   max_molecule_size = 5
   thermostat_coupling_groups = [
     ["S", "C"],
     ["W", "Na"],
   ]
   dielectric_const = 45.0              # dielectric constant for the simulation medium
   coulombtype = "PIC_Spectral"         # particle in cloud, spectral method

   [field]
   mesh_size = [64, 64, 64]
   sigma = 0.5
   chi = [
     ["S", "C",  13.50],
     ["S", "Na",  0.00],
     ["S", "W",  -3.60],
     ["C", "Na", 13.50],
     ["C", "W",  33.75],
     ["W", "Na",  0.00],
   ]

   [bonds]
   bonds = [
     ["S",   "C",   0.50, 1250.0],
     ["C",   "C",   0.50, 1250.0],
   ]
   angle_bonds = [
     ["S", "C", "C", 170.0, 25.0],
     ["C", "C", "C", 180.0, 25.0],
   ]


An input structure/topology file :code:`sds.HDF5` with a couple thousand
randomly dispersed SDS chains is available in the `HyMD-tutorial/sds`_ github
repository (along with the code to generate it). Note that the charges are
specified in the structure/topology file through a :code:`/charge` dataset.
Running the simulation with

.. code:: bash

   python3 -m hymd sds.toml sds.HDF5

we observe the aggregation of the SDS into micellular structures---first oblate
spheroid shaped and later fully spherical. An example of this is shown in the
accompanying Jupyter notebook `sds.ipynb`_.


.. _HyMD-tutorial/sds:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/tree/main/sds
.. _sds.ipynb:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/blob/main/sds/sds.ipynb
