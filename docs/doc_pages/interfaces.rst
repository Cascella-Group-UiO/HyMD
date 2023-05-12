.. _interfaces-label:

PLUMED interface
################

HyMD is interfaced with `PLUMED <https://www.plumed.org/>` for simulations using enhanced sampling or free-energy methods.
To use the interface, the Python package of PLUMED must be installed, and the user must provide a working :code:`PLUMED_KERNEL` by setting this environment variable to the kernel path.
In the current version, only simulations using a single replica are supported.

.. warning::
   PLUMED versions prior to :code:`2.8.1` did not have a working Python interface with MPI. Therefore, you **must** use PLUMED :code:`2.8.1` or greater when installing the PLUMED library in Python. Older versions for the kernel are supported if the Python interface was correctly installed with version :code:`2.8.1`.

Running simulations using PLUMED
================================

To run your simulations using PLUMED you must first set the environment variable :code:`PLUMED_KERNEL` and then call HyMD with the :code:`--plumed` option, passing the PLUMED input file to HyMD.
Optionally, you can also set the output file name with the :code:`--plumed-outfile` option (the default it :code:`plumed.out`).

If your PLUMED input (see the `PLUMED manual <>` for details) is called, e.g., :code:`plumed_input.dat`, you can run a simulation using the PLUMED interface with:

.. code:: bash

   python3 -m hymd ideal_chain.toml ideal_chain.HDF5 --disable-field --plumed plumed_input.dat
