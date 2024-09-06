.. _api-label:


API reference
=============

Integrator
----------

.. automodule:: hymd.integrator
   :members:


Thermostat
----------

.. automodule:: hymd.thermostat
   :members: csvr_thermostat, _random_chi_squared, _random_gaussian
   :undoc-members:


Barostat
--------

Berendsen Barostat
^^^^^^^^^^^^^^^^^^

.. automodule:: hymd.barostat
   :members: isotropic, semiisotropic
   :undoc-members:

SCR Barostat
^^^^^^^^^^^^^^^^^^

.. automodule:: hymd.barostat_bussi
   :members: isotropic, semiisotropic
   :undoc-members:


Hamiltonian
-----------

.. autoclass:: hymd.hamiltonian.Hamiltonian
   :members:

   .. automethod:: __init__

   .. automethod:: _setup

.. autoclass:: hymd.hamiltonian.SquaredPhi
   :members:

   .. automethod:: __init__


.. autoclass:: hymd.hamiltonian.DefaultNoChi
   :members:

   .. automethod:: __init__


.. autoclass:: hymd.hamiltonian.DefaultWithChi
   :members:

   .. automethod:: __init__


Force
-----

.. automodule:: hymd.force
   :members:
   :undoc-members:


Pressure
--------

.. automodule:: hymd.pressure
   :members: comp_pressure
   :undoc-members:


Logger
------

.. autoclass:: hymd.logger.Logger
   :members:

.. autoclass:: hymd.logger.MPIFilterRoot
   :members:

.. autoclass:: hymd.logger.MPIFilterAll
   :members:


Input parser
------------

.. automodule:: hymd.input_parser
   :members:
   :undoc-members: propensity_potential_coeffs


Field
-----

.. automodule:: hymd.field
   :members:


File input/output
-----------------

.. automodule:: hymd.file_io
   :members:

PLUMED
------

.. automodule:: hymd.plumed
   :members:
