.. _theory-label:

Theory
######
Hybrid particle--field simulations switch out the ordinary particle--particle
`Lennard-Jones`_ interactions with interactions between particles and a slowly
varying density field. In this way, the most expensive part of normal molecular
dynamics simulations is circumvented. Hybrid particle--field densities are
defined as

.. math::

   \phi(\mathbf{r}) = \sum_{i=1}^NP(\mathbf{r}-\mathbf{r}_i),

where :math:`\mathbf{r}` is a spatial coordinate, :math:`P` is a window function
used to distribute particle number densities onto a computational grid, and
:math:`\mathbf{r}_i` is the position of particle :math:`i` (of total :math:`N`).
By default, HyMD uses a Cloud-In-Cell (CIC) window function. A Hamiltonian form
for a system of :math:`N` particles in :math:`M` molecules is

.. math::

   \mathcal{H}(\{\mathbf{r}\})=\sum_{m=1}^MH_0(\{\mathbf{r},\mathbf{v}\}_m)+W[\phi(\mathbf{r})]

with :math:`H_0` being a standard intramolecular Hamiltonian form (see
:ref:`bonds-label`) including kinetic terms, while :math:`W` is a density
dependent *interaction energy functional*.

In the Hamiltonian hPF-MD formalism :cite:`bore2020hamiltonian`, the density
field is filtered using a grid-independent filtering function :math:`H`,

.. math::

   \tilde\phi(\mathbf{r})=\int\mathrm{d}\mathbf{x}\,\phi(\mathbf{x})H(\mathbf{r}-\mathbf{x}).

The filter smooths the density, ensuring that :math:`\tilde\phi` and
:math:`W[\tilde\phi([\phi])]` both converge as the grid size is reduced.

External potential
==================
The external potential acting on a particle is defined as the functional
derivative of :math:`W` with respect to :math:`\phi`. In the filtered formalism,
the potential takes the form

.. math::

   V(\mathbf{r}) &= \int\mathrm{d}\mathbf{y}\,\frac{\delta w}{\delta\phi(\mathbf{r})} \\
   &= \int\mathrm{d}\mathbf{y}\,\frac{\delta w}{\delta\tilde\phi(\mathbf{y})}\frac{\delta\tilde\phi(\mathbf{y})}{\delta\phi(\mathbf{r})},

under the assumption of a *local* form of the interaction energy functional,
:math:`W[\tilde\phi]=\int\mathrm{d}\mathbf{r}\,w[\tilde\phi(\mathbf{r})]`. Note
that

.. math::

   \frac{\delta \tilde\phi(\mathbf{y})}{\delta \phi(\mathbf{r})} = H(\mathbf{y}-\mathbf{r}).

Force interpolation
===================
The forces on particle :math:`i` are obtained by differentiation of the external
potential,

.. math::

   \mathbf{F}_i=-\int\mathrm{d}\mathbf{r}\,\nabla V(\mathbf{r})P(\mathbf{r}-\mathbf{r}_i).


Reciprocal space calculations
=============================
The field operations in HyMD are discretised and performed on a grid in
reciprocal space using (discrete) fast Fourier transform algorithms. After
interpolating the density :math:`\phi_{ijk}` with CIC, we apply the filtering
and obtain the discrete version of the external potential by

.. math::

   \tilde\phi_{ijk}=\mathrm{FFT}^{-1}\big[\mathrm{FFT}(\phi)\mathrm{FFT}(H)\big]

and

.. math::

   V_{ijk}=\mathrm{FFT}^{-1}\left[\mathrm{FFT}\left(\frac{\delta w(\tilde\phi)}{\delta \tilde\phi}\right)\mathrm{FFT}(H)\right].

The forces are obtained by differentiation of :math:`V` in Fourier space as

.. math::

   \nabla V_{ijk} = \mathrm{FFT}^{-1}\left[i\mathbf{k}\mathrm{FFT}\left(\frac{\delta w(\tilde\phi)}{\delta \tilde\phi}\right)\mathrm{FFT}(H)\right].

Filter
======
By default, the filter used in HyMD is a simple Gaussian of the form

.. math::

   H(x) &= \frac{1}{\sqrt{2\pi}\sigma}\exp\left[\frac{-x^2}{2\sigma^2}\right] \\
   \hat{H}(k) &= \exp\left[\frac{-\sigma^2k^2}{2}\right].

For more details about the filtering, see :ref:`filtering-label`.

Hamiltonian form
================
The default form of the interaction energy functional in HyMD is

.. math::

   W=\frac{1}{2\phi_0}\int\mathrm{d}\mathbf{r}\sum_{\text{i},\text{j}}\tilde\chi_{\text{i}-\text{j}}\tilde\phi_\text{i}(\mathbf{r})\tilde\phi_\text{j}(\mathbf{r}) + \frac{1}{2\kappa}\left(\sum_\text{k}\tilde\phi_\text{k}(\mathbf{r})-\phi_0\right)^2.

See :ref:`functionals-label` for details.


.. _`Lennard-Jones`:
   https://en.wikipedia.org/wiki/Lennard-Jones_potential
