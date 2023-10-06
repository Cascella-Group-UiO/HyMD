.. _functionals-label:

Interaction energy functionals
##############################
A few interaction energy functionals :math:`W` are defined in HyMD through
the :code:`Hamiltonian` class and associated subclasses.

:math:`\chi`--interaction and :math:`\kappa`--incompressibility
===============================================================
The most common hPF energy functional is specified by

.. code:: toml

   hamiltonian = "DefaultWithChi"

in the configuration file (see :ref:`config-label`). It takes the form

.. math::

   W=\frac{1}{2\phi_0}\int\mathrm{d}\mathbf{r}\sum_{\text{i}, \text{j}}\tilde\chi_{\text{i}-\text{j}}\tilde\phi_\text{i}(\mathbf{r})\tilde\phi_\text{j}(\mathbf{r}) + \frac{1}{2\kappa}\left(\sum_\text{k}\tilde\phi_\text{k}(\mathbf{r})-\phi_0\right)^2,

where :math:`\tilde\chi_{\text{A}-\text{B}}` is the interaction energy between
overlapping densities of particle types :math:`\text{A}` and :math:`\text{B}`.
The incompressibility term is governed by the compressibility parameter
:math:`\kappa`. The average density of the full system is denoted
:math:`\phi_0`.

Using this energy functional necessitates specification of :math:`\tilde\chi`
and :math:`\kappa` in the configuration file (see :ref:`config-label`)

.. code-block:: toml

   kappa = 0.05
   chi = [
      ["A", "B", 15.85],
      ["B", "C", -5.70],
      ...
   ]

Only :math:`\kappa`--incompressibility
======================================
The only :math:`\kappa` interactions energy functional is specified by

.. code:: toml

   hamiltonian = "DefaultNoChi"

in the configuration file (see :ref:`config-label`). It takes the form

.. math::

   W=\int\mathrm{d}\mathbf{r}\frac{1}{2\kappa}\left(\sum_\text{k}\tilde\phi_\text{k}(\mathbf{r})-\phi_0\right)^2,

where the incompressibility term is governed by the compressibility parameter
:math:`\kappa`. The average density of the full system is denoted
:math:`\phi_0`.

Using this energy functional necessitates specification of :math:`\kappa` in
the configuration file (see :ref:`config-label`)

.. code-block:: toml

   kappa = 0.05


Only :math:`\phi^2`
===================
The :math:`\phi^2` interactions energy functional is specified by

.. code:: toml

   hamiltonian = "SquaredPhi"

in the configuration file (see :ref:`config-label`). It takes the form

.. math::

   W=\int\mathrm{d}\mathbf{r}\frac{1}{2\kappa\phi_0}\left(\sum_\text{k}\tilde\phi_\text{k}(\mathbf{r})\right)^2,

where the :math:`\phi` squared term is governed by the compressibility parameter
:math:`\kappa`. The average density of the full system is denoted
:math:`\phi_0`.

Using this energy functional necessitates specification of :math:`\kappa` in
the configuration file (see :ref:`config-label`)

.. code-block:: toml

   kappa = 0.05


Implementing new interaction energy forms
=========================================
Implementing new interaction energy functions is straightforward by subclassing
:code:`Hamiltonian` and applying :code:`sympy` differentiation to the symbolic
field objects. The :code:`sympy.lamdify` function creates vectorised numpy
functions automatically from the differentiation result which is used to
transform the density fields.
