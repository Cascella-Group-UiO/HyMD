.. _pressure-label:

Pressure
########
Internal pressure is calculated from internal energy according to

.. math::

    P_a = \frac{1}{\mathcal{V}} \left( 2T_a - \text{Vir}_a \right) \\
    \text{Vir}_a = L_a \frac{\partial \mathcal{U}}{\partial L_a} \\
    \mathcal{U} = \sum_{i=1}^M \mathcal{U}_0( \{ \mathbf{r}\}_i ) + W[\{ \tilde\phi \} ]

where
:math:`\mathcal{V}` is the simulation volume,
:math:`{T_a}` is the kinetic energy
and :math:`L_a` the length of the box in the Cartesian direction :math:`a`,
Vir is the virial of the total interaction energy :math:`\mathcal{U}`.

:math:`\mathcal{U}` comprises intramolecular bonded terms :math:`\mathcal{U}_0` (see :ref:`bonds-label` for details),
and field terms :math:`W[\{ \tilde\phi \} ]` (see :ref:`theory-label` for details).

Using the above expressions, the following form for internal pressure is obtained:

.. math::

    P_a = \frac{2 T_a}{\mathcal{V}} -\frac{L_a}{\mathcal{V}} \sum_{i=1}^N \frac{\partial \mathcal{U}_{0i}}{\partial L_a} + P^{(3)}_a \\

.. P^{(3)}_a = - \frac{L_a}{\mathcal{V}} \frac{\partial W[\{ \tilde\phi \} ]}{\partial L_a} \\

.. math::

    P^{(3)}_a = \frac{1}{\mathcal{V}}\left ( -W[\{ \tilde\phi(\mathbf{r}) \}] + \int \sum_t \bar{V}_t(\mathbf{r})\tilde\phi_t(\mathbf{r})d\mathbf{r} 
                + \int \sum_t \sigma^2\bar{V}_t(\mathbf{r})\nabla_a^2\tilde\phi_t(\mathbf{r}) d\mathbf{r} \right)

where :math:`\bar{V}_t(\mathbf{r}) = \frac{\partial w(\{\tilde\phi\})}{\partial\tilde\phi_t}` 
and :math:`σ` is a coarse-graining parameter (see :ref:`filtering-label` for details).
Note that the above expression is obtained for a Gaussian filter which is the most natural choice in HhPF theory.
