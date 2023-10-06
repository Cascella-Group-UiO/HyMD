.. _bonds-label:

Intramolecular bonds
####################

Intramolecular bonds types in HyMD are specified in the configuration file.
Once a simulation starts, the specific indices of bonded particles are inferred
from the type information in the configuration file, see :ref:`config-label`.
The construction of the bonds internally happens locally on each MPI rank after
every explicit domain decomposition transfer, which is the only time during
simulation when particles are permanently re-assigned to new CPUs. Ideally, you
should be running HyMD with relatively few particles per MPI rank (see
:ref:`benchmarks-label`), around 200 being the optimal value for maximum
efficiency. In this case, the computational cost involved in reconstructing the
bond network subsequent to every domain decomposition swap is minuscule and not
measurable.

All intramolecular potentials are computed in highly optimised Fortran kernels,
and require no MPI communication (except for explicit domain decomposition
swaps which are performed very rarely in practice [e.g. every hundreds of
thousands time steps]).


Two-particle bonds
^^^^^^^^^^^^^^^^^^
Stretching bonds in HyMD are implemented as harmonic spring potentials,

.. math::

   V_2(r) = \frac{k}{2}(r-r_0)^2,

where :math:`r=|\mathbf{r}|` is the inter-particle distance, :math:`k` is a
constant of dimension energy (units :math:`\text{kJ}\,\text{mol}^{-1}`), and
:math:`r_0` is the equilibrum length of the bond (units: :math:`\text{nm}`). The
force is calculated as

.. math::

   F_{i\rightarrow j}(\mathbf{r}) = k(r-r_0)\frac{\mathbf{r}}{r},

where :math:`\mathbf{r}` is the vector pointing *from* :math:`i` *to* :math:`j`.

In the configuration file, two-particle bonds are specified per particle type:

.. code-block:: toml
   :caption: configuration_two_particle_example.toml

   [bonds]
   bonds = [
      ["A", "A", 0.47, 1250.0],
      ["A", "B", 0.37,  940.0],
      ["B", "C", 0.50, 1010.0],
      ["C", "A", 0.42,  550.0],
      ...
   ]

The order of the specified names does not matter, but the order of the length
and energy scales do.

Three-particle bonds
^^^^^^^^^^^^^^^^^^^^
Bending bonds in HyMD are implemented as harmonic angular bonds, depending on
the particle--particle--particle angle :math:`\theta`,

.. math::

   V_3(\theta) = \frac{k}{2}(\theta-\theta_0)^2,

where :math:`k` is a constant of dimension energy (units
:math:`\text{kJ}\,\text{mol}^{-1}`), and :math:`\theta_0` is the equilibrum
angle of the bond (units: :math:`{}^\circ`). Defining the particles with labels
:math:`a`, :math:`b`, and :math:`c`, let :math:`\mathbf{r}_a` denote the vector
pointing from :math:`b` to :math:`a`, and correspondingly let
:math:`\mathbf{r}_c` point from :math:`b` to :math:`c`. The
:math:`a`--:math:`b`--:math:`c` may be computed through the law of Cosines,

.. math::

   \theta = \cos^{-1}\left[\frac{\mathbf{r}_a\cdot \mathbf{r}_c}{r_a r_c}\right]

Then the force acting on :math:`a` and :math:`c` is

.. math::

   \mathbf{F}_a = -\frac{\mathrm{d}V_3(\theta)}{\mathrm{d}r_a}\frac{\mathrm{d}r_a}{\mathrm{d}\mathbf{r}_a},

   \mathbf{F}_c = -\frac{\mathrm{d}V_3(\theta)}{\mathrm{d}r_c}\frac{\mathrm{d}r_c}{\mathrm{d}\mathbf{r}_c},

and

.. math::

   \mathbf{F}_b = - \mathbf{F}_a - \mathbf{F}_c

In the configuration file, three-particle bonds are specified per particle type:

.. code-block:: toml
   :caption: configuration_three_particle_example.toml

   [bonds]
   angle_bonds = [
      ["A", "A", "A", 180.0, 90.0],
      ["A", "B", "A", 120.0, 55.0],
      ["B", "C", "C",  30.0, 10.0],
      ["C", "A", "B", 110.0, 25.5],
      ...
   ]

The order of the specified names does not matter, but the order of the length
and energy scales do.

Four-particle bonds
^^^^^^^^^^^^^^^^^^^
Torsional dihedral potentials in HyMD are implemented as Cosine series
potentials, depending on the angle :math:`\phi` between the
:math:`a`--:math:`b`--:math:`c` and :math:`b`--:math:`c`--:math:`d` planes, for
a dihedral bond quartet :math:`a`, :math:`b`, :math:`c`, and :math:`d`. The
potential takes the form

.. math::

   V_4(\phi) = \frac{1}{2}\sum_n^M c_n\cos(n\phi+\phi_n),

where :math:`c_n` is a constant of dimension energy (units
:math:`\text{kJ}\,\text{mol}^{-1}\text{rad}^{-2}`), and :math:`\phi_n` are
specified phase angles in the cosine series.

In the configuration file, four-particle bonds are specified per particle type:

.. code-block:: toml
   :caption: configuration_four_particle_example.toml

   [bonds]
   dihedrals = [
      ["A", "B", "C", "D"],
      [
        [-1],
        [449.08790868, 610.2408724, -544.48626121, 251.59427866, -84.9918564],
        [0.08, 0.46, 1.65, -0.96, 0.38],
      ],
      [1.0],
   ]
