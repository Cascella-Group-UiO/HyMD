.. _commandline-label:

Command line arguments
######################
Certain options are provided to HyMD as command line arguments. Besides input
and output file specifications, these options mostly constitute debugging
options such as disabling all bonded or particle--field interactions.


Required arguments
==================
:configuration file:
   :code:`type` positional

   Specifies the :code:`.toml` format configuration file containing simulation details. See :ref:`config-label` for details.

:structure and topology file:
   :code:`type` positional

   Specifies the positions, indices, names (and optionally molecular affiliation, bond structure, velocities). See :ref:`topology-label` for details.

Optional arguments
==================
:code:`-v  --verbose`
   :code:`type` optional

   Number of following arguments: :code:`0` or :code:`1`

   Determines the verbosity of the simulation logger. More verbose means more text output during runs. Verbosity may be increased by specifying :code:`-v` or :code:`--verbose` with no additional specification. This increases verbosity level by :code:`1`. Alternatively, it maye be specified by :code:`--verbose V` with :code:`V` being :code:`0`, :code:`1`, or :code:`2`.

:code:`--profile`
   :code:`type` optional

   Number of following arguments: :code:`0`

   Enables code profiling using :code:`cProfile`. This will output one profiling file per MPI rank invoked in the simulation.

:code:`--double-precision`
   :code:`type` optional

   Number of following arguments: :code:`0`

   Specify usage of double precision internally in HyMD (including in the FFTs).

:code:`--double-output`
   :code:`type` optional

   Number of following arguments: :code:`0`

   Specify usage of double precision in the output trajectory from HyMD.

:code:`--velocity-output`
   :code:`type` optional

   Number of following arguments: :code:`0`

   Specify that velocities should be output to the HyMD trajectory.

:code:`--force-output`
   :code:`type` optional

   Number of following arguments: :code:`0`

   Specify that forces should be output to the HyMD trajectory.

:code:`--disable-mpio`
   :code:`type` optional

   Number of following arguments: :code:`0`

   Specify that a non-MPI-enabled HDF5 library is to be used, foregoing the parallel file IO capabilities of h5py. This is normally used as a compatibility option for machines on which installing MPI-enabled HDF5 is difficult.

:code:`--logfile`
   :code:`type` optional

   Number of following arguments: :code:`1`

   Specifies the path of a plain text log file in which stdout is mirrored during simulation runs.

:code:`--seed`
   :code:`type` optional

   Number of following arguments: :code:`1`

   Specifies the random seed used in the :code:`numpy` random library to generate velocities, thermostatting, etc.

:code:`--disable-bonds`
   :code:`type` optional

   Number of following arguments: :code:`0`

   Disable any two-particle bonds present in the system.

:code:`--disable-angle-bonds`
   :code:`type` optional

   Number of following arguments: :code:`0`

   Disable any three-particle bonds present in the system.

:code:`--disable-dihedrals`
   :code:`type` optional

   Number of following arguments: :code:`0`

   Disable any four-particle bonds present in the system.

:code:`--disable-dipole`
   :code:`type` optional

   Number of following arguments: :code:`0`

   Disable any topological reconstruction of peptide backbone dipoles present in the system.

:code:`--disable-field`
   :code:`type` optional

   Number of following arguments: :code:`0`

   Disable any particle--field interactions present in the system.
