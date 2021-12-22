.. _overview-label:

================================
 Usage overview for HylleraasMD
================================

HyMD is a Python package that allows large scale parallel hybrid particle-field
molecular dynamics (hPF-MD) simulations. The main simulation driver takes as
input a `toml`_ format configuration file, along with a topology and structure
file specifying the input system in HDF5 format.

.. _`toml`:
   https://github.com/toml-lang/toml

**Configuration file**

The configuration file specifies the type and length of simulation to run. A
simple example of a configuration file which performs a short simulation of a
system contained in a :code:`[5, 5, 5]` nm simulation box at 300 K may look like
the following:

.. code-block:: toml
   :caption: **config_simple_1.toml**

   [simulation]
   integrator = "velocity-verlet"
   n_steps = 100                      # total simulation steps
   n_print = 10                       # output trajectory data every n_print steps
   time_step = 0.01                   # time step in ps
   box_size = [5.0, 5.0, 5.0]         # simulation box size in nm
   target_temperature = 300           # thermostat temperature in Kelvin
   tau = 0.1                          # thermostat coupling strength in ps
   hamiltonian = "SquaredPhi"         # interaction energy functional
   kappa = 0.05                       # compressibility in mol/kJ

   [field]
   mesh_size = [30, 30, 30]           # FFT grid size
   sigma = 0.5                        # filtering length scale in nm

Note that for the version 1.0 release of HyMD, the :code:`[particles]`,
:code:`[simulation]`, and :code:`[field]` groups are optional specifiers for
structuring the configuration file.

For a full specification of every recognised configuration keyword in HyMD,
please refer to :ref:`config-label`.

**Topology and structure file**

The topology and structure file specifies the positions, momenta, types, and
molecular structure of the particles in the system. In order to allow highly
parallel file read during program initialisation, the structure file format is a
simple flat HDF5 structure. An example structure file for a system containing
:code:`N` particles may look like the following:

.. code-block:: bash
   :caption: **example_structure.HDF5**

   group   /
      # root group

   dataset /coordinates
      # float32, shape (T, N, D,)
      # Time steps of N rows of D dimensional coordinate data
      # The last frame is used by default if multiple time steps are provided
      # T is the number of time steps, D is the spatial dimension

   dataset /velocities
      # float32, shape (T, N, D,) OPTIONAL
      # Time steps of N rows of D dimensional velocity data
      # The last frame is used by default if multiple time steps are provided
      # T is the number of time steps, D is the spatial dimension

   dataset /bonds
      # int32,   shape (N, B,)    OPTIONAL
      # Each row i contains indices of the particles bonds from/to i
      # B denotes the maximum bonds per particle

   dataset /indices
      # int32,   shape (N,)
      # Index of each particle, 0--(N-1)

   dataset /molecules
      # int32,   shape (N,)
      # Index of molecule each particle belongs to
      # Used to construct bond network locally on MPI ranks

   dataset /names
      # str10,   shape (N,)
      # Fixed size string array containing particle type names

   dataset /types
      # int32,   shape (N,)       OPTIONAL
      # Mapping of particle names to type indices


For a full specification of the topology and structure input file, see
:ref:`topology-label`.

**Command line arguments**

A select few settings for HyMD are specified on the command line, most notably
output files, output directories, output specifications, and random seeds. In
addition, the configuration and structure files are provided as the first and
second required positional arguments. An example for running the simulation
specified by the :code:`config_simple_1.toml` and :code:`example_structure.HDF5`
files may look like the following:

.. code-block:: bash

   mpirun -n ${MPI_NUM_RANKS} python3 -m hymd         \  # HyMD executable
                              config_simple_1.toml    \  # configuration file
                              example_structure.HDF5  \  # structure file
                              --logfile=log_out.txt   \  # set logfile path
                              --seed 123456           \  # set random seed
                              --verbose 2             \  # set logging verbosity
                              --velocity-output          # enable velocity in trajectory output

For a full specification of all command line arguments, see
:ref:`commandline-label`.

**More examples**

For more thorough rundown and concrete usage examples, see
:ref:`examples-label`.

Running parallel simulations
============================
Executing the HyMD code in parallel is fundamentally no different from running
it in serial, with the **only** difference being how the python3 interpreter is
invoked. Prepending :code:`mpirun -n ${NPROCS}` to the usual
:code:`python3 -m hymd` runs the simulation with :code:`${NPROCS}` MPI ranks.

Inputting

.. code:: bash

   mpirun -n 6 python3 -m ideal_gas.toml ideal_gas.HDF5 --disable-field

sets up an example simulation of the `HyMD-tutorial/ideal_gas`_ system. For
more details about this system in particular, or other example simulations, see
the `HyMD-tutorial`_ repository.

.. _HyMD-tutorial:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial
.. _HyMD-tutorial/ideal_gas:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial/tree/main/ideal_gas


Running tests
=============
Run serial code tests by

.. code-block:: bash

   git clone https://github.com/Cascella-Group-UiO/HyMD.git hymd
   cd hymd/
   python3 -m pip install pytest pytest-mpi
   python3 -m pytest

There is a small convenience script in :code:`hymd/` for running the MPI-enabled
tests,

.. code-block:: bash

   chmod +x pytest-mpi
   ./pytest-mpi --nprocs 5 --order-output --no-summary --capture=no
