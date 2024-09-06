.. hymd documentation master file, created by
   sphinx-quickstart on Thu Dec 16 13:31:43 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

###############################
HylleraasMD documentation
###############################

:Release:
   |release|
:Date:
   |today|

**HylleraasMD** (HyMD) is a massively parallel Python package for Hamiltonian hybrid
particle-field molecular dynamics (HhPF-MD) simulations of coarse-grained bio-
and soft-matter systems.

HyMD can run canonical hPF-MD simulations, or filtered density HhPF-MD simulations :cite:`hymd_domain2023,hymd_massive2023,hymd_pressure2023,hymd_alias2020`
with or without explicit PME electrostatic interactions. It includes all standard intramolecular interactions,
including stretching, bending, torsional, and combined bending-dihedral potentials. Additionally, topological reconstruction of permanent peptide chain backbone
dipoles is possible for accurate recreation of protein conformational dynamics. It can run simulations in constant energy (NVE), constant volume (NVT) :cite:`hymd_domain2023,hymd_massive2023`
or constant pressure (NPT) conditions :cite:`hymd_pressure2023`.
HyMD is also interfaced with :doc:`PLUMED </doc_pages/interfaces>` and can perform simulations using enhanced sampling methods.

HyMD uses the pmesh library for particle-mesh operations, with the PPFT :cite:`pippig2013` backend for FFTs through the pfft-python bindings.
File IO is done via HDF5 formats to allow MPI parallel reads.

User Guide
==========
The HylleraasMD :doc:`User Guide </doc_pages/overview>` provides comprehensive information on how to run
simulations. Selected :doc:`Examples </doc_pages/examples>` are available to guide new users.

.. _`github.com/rainwoodman/pmesh`:
   https://github.com/rainwoodman/pmesh
.. _`github.com/rainwoodman/pfft-python`:
   https://github.com/rainwoodman/pfft-python

Installing HyMD
===============
The easiest approach is to install using pip_:

.. code-block:: bash

   python3 -m pip install --upgrade pip
   python3 -m pip install --upgrade numpy mpi4py cython
   python3 -m pip install hymd

For more information and **required dependencies**, see
:ref:`installation-label`.

.. _pip:
   http://www.pip-installer.org/en/latest/index.html


Run in Google colab
-------------------
Run HyMD interactively in `Google Colaboratory`_ jupyter notebook `here`_.

.. _`Google colaboratory` :
   https://colab.research.google.com/
.. _`here` :
   https://colab.research.google.com/drive/1jfzRaXjL3q53J4U8OrCgADepmf_HuCOh?usp=sharing


Source Code
===========
**Source code** is available from
https://github.com/Cascella-Group-UiO/HyMD/ under the `GNU Lesser General Public
License v3.0`_. Obtain the source code with `git`_:

.. code-block:: bash

   git clone https://github.com/Cascella-Group-UiO/HyMD.git

.. _GNU Lesser General Public License v3.0:
   https://www.gnu.org/licenses/lgpl-3.0.html
.. _git:
   https://git-scm.com/


Development
===========
HyMD is developed and maintained by researchers at the `Hylleraas Centre for
Quantum Molecular Sciences`_ at the `University of Oslo`_.

|pic1| |pic2|

.. _`Hylleraas Centre for Quantum Molecular Sciences`:
   https://www.mn.uio.no/hylleraas/english/
.. _`University of Oslo`:
   https://www.uio.no/

.. |pic1| image:: img/hylleraas_centre_logo_black.png
   :target: img/hylleraas_centre_logo_black.png
   :width: 250 px

.. |pic2| image:: img/uio_full_logo_eng_pos.png
   :target: img/uio_full_logo_eng_pos.png
   :width: 325 px



References
==========
.. bibliography::
  :all:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. Contents
.. ========

.. toctree::
   :maxdepth: 2
   :numbered:
   :hidden:
   :caption: Getting started

   ./doc_pages/installation
   ./doc_pages/overview
   ./doc_pages/config_file
   ./doc_pages/topology_input
   ./doc_pages/command_line
   ./doc_pages/constants_and_units


.. toctree::
   :maxdepth: 2
   :numbered:
   :hidden:
   :caption: Examples

   ./doc_pages/examples

.. toctree::
   :maxdepth: 2
   :numbered:
   :hidden:
   :caption: Theory

   ./doc_pages/theory
   ./doc_pages/intramolecular_bonds
   ./doc_pages/electrostatics
   ./doc_pages/pressure
   ./doc_pages/interfaces

.. toctree::
   :maxdepth: 2
   :numbered:
   :hidden:
   :caption: Developer documentation

   ./doc_pages/interaction_energy_functionals
   ./doc_pages/filtering
   ./doc_pages/benchmarks
   ./doc_pages/api
