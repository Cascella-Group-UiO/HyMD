.. _filtering-label:

Filtering
#########
The default grid-independent filter function used in HyMD is a Gaussian

.. math::

   H(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left[\frac{-x^2}{2\sigma^2}\right]

with reciprocal space representation

.. math::

   \hat{H}(k) = \exp\left[\frac{-\sigma^2k^2}{2}\right].

The coarse-graining parameter :math:`\sigma` determines a length scale of
particle extent and is given as configuration file input by

.. code:: toml

   sigma = 0.75


Implementing new filters
^^^^^^^^^^^^^^^^^^^^^^^^
Implementing new interaction energy functions is straightforward by hijacking
the :code:`setup` method of the Hamiltonian superclass. The new filter is
automatically applied and differentiated through the potential and interaction
energy functional.
