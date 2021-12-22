.. _electrostatics-label:

Electrostatic interactions
##########################
In the filtered Hamiltonian hPF formalism :cite:`bore2020hamiltonian` the
particles are intrinsically smeared, filtered density distributions. The
electrostatic interactions between such smeared densities takes on the form of
the long-range part of ordinary particle--mesh Ewald.

The charge density is projected onto the computational grid by use of the CIC
window function, and subsequently filtered

.. math::

   \tilde\rho =\int\mathrm{d}\mathbf{x}\,H(\mathbf{r}-\mathbf{x})\sum_{i=1}^Nq_i P(\mathbf{r}-\mathbf{r}_i),

with :math:`q_i` being the charge of particle :math:`i` and :math:`H` is the
filtering function (see :ref:`filtering-label`). The value of the charge grid at
vertex :math:`(i,j,k)` is found by

.. math::

   \tilde\rho_{ijk}=\text{FFT}^{-1}[\text{FFT}(\rho)\text{FFT}(H)]

and the electrostatic potential :math:`\Psi_{ijk}`

.. math::

   \Psi_{ijk}=\text{FFT}^{-1}\left[\frac{k_\text{e}}{|\mathbf{k}^2|}\text{FFT}(\rho)\text{FFT}(H)\right],

where :math:`k_\text{e}` is the Coulomb constant :math:`1/4\pi\varepsilon_0`.
The *electric field* is obtained by differentiation of the electrostatic
potential in Fourier space,

.. math::

   \mathbf{E}_{ijk}=\text{FFT}^{-1}[-i\mathbf{k}\text{FFT}(\Psi)]

from which the forces are interpolated back to the particle positions.

Specifying electrostatics
^^^^^^^^^^^^^^^^^^^^^^^^^
In HyMD, electrostatics are specified by the :code:`coulombtype` and
:code:`dielectric_const` keywords in the configuration file (see
:ref:`config-label`) and the :code:`/charge` dataset in the HDF5 format
structure/topology input file (see :ref:`topology-label`). In addition, the
helical propensity peptide dihedral type induces topological reconstruction
of peptide dipoles which adds electrostatic interactions.
